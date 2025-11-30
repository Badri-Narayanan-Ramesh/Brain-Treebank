# src/data/dataset_popt_speech.py
"""
Dataset and dataloader helpers for **PopT-style speech decoding**.

This reads the artifacts written by the official PopT scripts, e.g.:

    saved_examples/all_test_word_onset/
        sub_1/
            trial000/
                0.npy
                1.npy
                ...
            trial002/
                ...
        manifest.tsv
        labels.tsv

Assumptions:

- `manifest.tsv` has one row per window:
      <windows_path>\t<subject>

  where the first column ends with something like:
      ...\\all_test_word_onset\\sub_1\\trial002\\0.npy

- `labels.tsv` has the same number of rows, with:
      <bool_label>\t<onset_sample_index>

  where bool_label is "True" or "False".

We assume row i in `labels.tsv` corresponds to row i in `manifest.tsv`.
We ignore the onset sample index and only keep the boolean label.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SpeechSample:
    subject: str
    trial: str
    idx: int
    label: int


class PopTSpeechDataset(Dataset):
    """
    Pure brain-only speech decoding dataset using PopT's own
    `all_test_word_onset` windows + labels.

    Returns:
        brain: (n_electrodes, 768) float32 tensor
        label: scalar 0/1 tensor (long)
        meta: dict with keys: subject, trial, idx
    """

    def __init__(
        self,
        brain_root: Path,
        manifest_tsv: Path,
        labels_tsv: Path,
        split: str = "train",            # "train" | "val"
        split_mode: str = "by_trial",    # "by_trial" | "random"
        train_trials: Optional[List[Tuple[str, str]]] = None,
        val_trials: Optional[List[Tuple[str, str]]] = None,
        train_frac: float = 0.9,         # used if split_mode="random"
        seed: int = 42,
        only_subjects: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.brain_root = Path(brain_root)
        self.split = split
        self.split_mode = split_mode
        self.train_frac = float(train_frac)
        self.seed = int(seed)
        self.only_subjects = set(only_subjects) if only_subjects is not None else None

        manifest_tsv = Path(manifest_tsv)
        labels_tsv = Path(labels_tsv)

        if not manifest_tsv.is_file():
            raise FileNotFoundError(f"manifest.tsv not found at: {manifest_tsv}")
        if not labels_tsv.is_file():
            raise FileNotFoundError(f"labels.tsv not found at: {labels_tsv}")

        # -------------------------
        # 1) Load manifest + labels
        # -------------------------
        paths: List[str] = []
        subjects_from_manifest: List[str] = []
        with manifest_tsv.open("r", encoding="utf-8") as f:
            tsv_reader = csv.reader(f, delimiter="\t")
            for row in tsv_reader:
                if not row:
                    continue
                path_str = row[0]
                subj = row[1] if len(row) > 1 else ""
                paths.append(path_str)
                subjects_from_manifest.append(subj)

        labels: List[int] = []
        with labels_tsv.open("r", encoding="utf-8") as f:
            tsv_reader = csv.reader(f, delimiter="\t")
            for row in tsv_reader:
                if not row:
                    continue
                flag = row[0].strip()  # "True" / "False"
                label = 1 if flag.lower() == "true" else 0
                labels.append(label)

        if len(paths) != len(labels):
            raise ValueError(
                f"manifest.tsv rows ({len(paths)}) != labels.tsv rows ({len(labels)})"
            )

        # -------------------------
        # 2) Build structured samples
        # -------------------------
        raw_samples: List[SpeechSample] = []
        for i, (p_str, subj_from_manifest, lab) in enumerate(
            zip(paths, subjects_from_manifest, labels)
        ):
            # Example tail: "sub_1\\trial002\\0.npy"
            parts = p_str.split("\\")
            if len(parts) < 3:
                raise ValueError(f"Unexpected path format in manifest row {i}: {p_str}")

            subj = parts[-3]   # e.g. "sub_1"
            trial = parts[-2]  # e.g. "trial002"
            fname = parts[-1]  # e.g. "0.npy"

            # Optional consistency check
            if subj_from_manifest and subj_from_manifest != subj:
                print(
                    f"[PopTSpeechDataset] WARNING: subject mismatch at row {i}: "
                    f"path says {subj}, manifest col says {subj_from_manifest}"
                )

            try:
                idx = int(Path(fname).stem)
            except ValueError as e:
                raise ValueError(f"Could not parse index from filename: {fname}") from e

            if self.only_subjects is not None and subj not in self.only_subjects:
                continue

            raw_samples.append(
                SpeechSample(subject=subj, trial=trial, idx=idx, label=lab)
            )

        if not raw_samples:
            raise ValueError("No samples found after filtering by subjects.")

        # -------------------------
        # 3) Train/val split
        # -------------------------
        if split_mode == "by_trial":
            # Here, train_trials and val_trials are already lists of (subject, trial) tuples
            train_set = set(train_trials or [])
            val_set = set(val_trials or [])

            if split == "train":
                self.samples = [
                    s for s in raw_samples if (s.subject, s.trial) in train_set
                ]
            elif split == "val":
                self.samples = [
                    s for s in raw_samples if (s.subject, s.trial) in val_set
                ]
            else:
                raise ValueError(f"Invalid split '{split}' for by_trial mode")

        elif split_mode == "random":
            rng = np.random.RandomState(self.seed)
            indices = np.arange(len(raw_samples))
            rng.shuffle(indices)
            split_idx = int(self.train_frac * len(indices))
            if split == "train":
                chosen = indices[:split_idx]
            elif split == "val":
                chosen = indices[split_idx:]
            else:
                raise ValueError(f"Invalid split '{split}' for random mode")
            self.samples = [raw_samples[i] for i in chosen]

        else:
            raise ValueError(f"Unknown split_mode '{split_mode}'")

        if not self.samples:
            raise ValueError(
                f"No samples remaining for split={split} with split_mode={split_mode}"
            )

        print(
            f"[PopTSpeechDataset] split={split}, split_mode={split_mode}, "
            f"num_samples={len(self.samples)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        brain_path = self.brain_root / s.subject / s.trial / f"{s.idx}.npy"
        if not brain_path.is_file():
            raise FileNotFoundError(f"Brain window not found: {brain_path}")

        brain = np.load(brain_path).astype(np.float32)  # (n_elec, 768)
        brain_t = torch.from_numpy(brain)               # (n_elec, 768)
        label_t = torch.tensor(s.label, dtype=torch.long)

        meta = {
            "subject": s.subject,
            "trial": s.trial,
            "idx": s.idx,
        }
        return brain_t, label_t, meta


def _expand_subject_trials(cfg_list):
    """
    Expand a config list into a list of (subject, trial) pairs.

    Supports:
        - [{'subject': 'sub_1', 'trials': ['trial000', 'trial002']}]
        - [('sub_1', 'trial000'), ('sub_1', 'trial002')]
    """
    pairs = []
    if cfg_list is None:
        return pairs

    for item in cfg_list:
        # Case 1: already a tuple (subject, trial)
        if isinstance(item, tuple) and len(item) == 2:
            pairs.append(item)

        # Case 2: dictionary with "subject" and "trials"
        elif isinstance(item, dict):
            subj = item["subject"]
            for t in item.get("trials", []):
                pairs.append((subj, t))

        else:
            raise TypeError(f"Unsupported train/val trials entry: {item}")

    return pairs


def make_popt_speech_dataloaders_from_config(config: dict):
    """
    Build train/val dataloaders for PopT speech decoding baseline
    from the global YAML config.

    Expected in config["data"]:
        brain_root_speech: path to all_test_word_onset root
        speech_manifest_tsv: path to manifest.tsv
        speech_labels_tsv: path to labels.tsv

        split_mode: "by_trial" | "random"
        train_subject_trials: [...]
        val_subject_trials: [...]

        train_frac: float (for random split)
        only_subjects: [list of subjects] (optional)

    And in config["training"]:
        batch_size, num_workers
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    brain_root = Path(data_cfg["brain_root_speech"])
    manifest_tsv = Path(data_cfg["speech_manifest_tsv"])
    labels_tsv = Path(data_cfg["speech_labels_tsv"])

    split_mode = data_cfg.get("split_mode", "by_trial")
    only_subjects = data_cfg.get("only_subjects", None)

    train_trials_cfg = data_cfg.get("train_subject_trials", None)
    val_trials_cfg = data_cfg.get("val_subject_trials", None)

    train_trials = _expand_subject_trials(train_trials_cfg)
    val_trials = _expand_subject_trials(val_trials_cfg)

    common_kwargs = dict(
        brain_root=brain_root,
        manifest_tsv=manifest_tsv,
        labels_tsv=labels_tsv,
        split_mode=split_mode,
        train_trials=train_trials,
        val_trials=val_trials,
        train_frac=data_cfg.get("train_frac", 0.9),
        seed=train_cfg.get("seed", 42),
        only_subjects=only_subjects,
    )

    ds_train = PopTSpeechDataset(split="train", **common_kwargs)
    ds_val = PopTSpeechDataset(split="val", **common_kwargs)

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 4))

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
