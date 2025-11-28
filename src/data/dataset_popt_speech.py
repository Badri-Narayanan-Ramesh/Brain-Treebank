# src/data/dataset_popt_speech.py

import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _expand_subject_trials(cfg_list):
    """
    Expand:
      - subject: "sub_1"
        trials: ["trial000", "trial002"]
    into list[ (sub, trial), ... ]
    """
    pairs = []
    if cfg_list is None:
        return pairs
    for item in cfg_list:
        sub = item["subject"]
        for t in item["trials"]:
            pairs.append((sub, t))
    return pairs


class PopTSpeechDataset(Dataset):
    """
    Dataset for PopT's *speech vs non-speech* decoding task,
    using the word-onset aligned windows written by the official PopT repo.

    Expected directory layout (brain_root):

        brain_root/
            sub_1/
                trial000/
                    0.npy
                    1.npy
                    ...
                trial002/
                    ...
            sub_2/
                trial000/
                trial001/
                ...

    Expected label manifest (CSV, one file for all subjects/trials):

        subject,trial,idx,label
        sub_1,trial000,0,1
        sub_1,trial000,1,0
        ...

    where:
        - subject: string (e.g. "sub_1")
        - trial:   string (e.g. "trial000")
        - idx:     integer, matching <idx>.npy file
        - label:   0 or 1 (non-speech / speech)
    """

    def __init__(
        self,
        brain_root: Path,
        manifest_path: Path,
        split: str = "train",           # "train" / "val" / "test"
        split_mode: str = "by_trial",   # "by_trial" or "random"
        train_trials: Optional[List[Tuple[str, str]]] = None,
        val_trials: Optional[List[Tuple[str, str]]] = None,
        train_frac: float = 0.9,
        seed: int = 42,
        only_subjects: Optional[List[str]] = None,
    ):
        super().__init__()

        self.brain_root = Path(brain_root)
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.split_mode = split_mode
        self.train_frac = train_frac
        self.seed = seed

        if only_subjects is not None:
            self.only_subjects = set(only_subjects)
        else:
            self.only_subjects = None

        # -------------------------
        # 1) Load manifest
        # -------------------------
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Label manifest not found: {self.manifest_path}")

        entries = []
        with open(self.manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = {"subject", "trial", "idx", "label"}
            if not required_cols.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Manifest {self.manifest_path} must have columns: "
                    f"{required_cols}, got {reader.fieldnames}"
                )

            for row in reader:
                subject = row["subject"]
                trial = row["trial"]
                idx = int(row["idx"])
                label = int(row["label"])

                if self.only_subjects is not None and subject not in self.only_subjects:
                    continue

                entries.append(
                    {
                        "subject": subject,
                        "trial": trial,
                        "idx": idx,
                        "label": label,
                    }
                )

        if not entries:
            raise ValueError(f"No entries found in {self.manifest_path} after filtering")

        # Optional: sanity check that .npy files exist
        missing_count = 0
        for e in entries:
            npy_path = self.brain_root / e["subject"] / e["trial"] / f"{e['idx']}.npy"
            if not npy_path.exists():
                missing_count += 1
        if missing_count > 0:
            print(
                f"[PopTSpeechDataset] WARNING: {missing_count} entries "
                f"have missing .npy files under {self.brain_root}"
            )

        # -------------------------
        # 2) Split into train / val
        # -------------------------
        if self.split_mode == "by_trial":
            # group by (subject, trial)
            all_trials = sorted({(e["subject"], e["trial"]) for e in entries})
            train_set = set(train_trials) if train_trials else set()
            val_set = set(val_trials) if val_trials else set()

            if self.split == "train":
                self.samples = [
                    e for e in entries if (e["subject"], e["trial"]) in train_set
                ]
            elif self.split == "val":
                self.samples = [
                    e for e in entries if (e["subject"], e["trial"]) in val_set
                ]
            elif self.split == "test":
                # Optionally, anything not in train_set/val_set
                test_set = all_trials - train_set - val_set
                self.samples = [
                    e for e in entries if (e["subject"], e["trial"]) in test_set
                ]
            else:
                raise ValueError(f"Invalid split: {self.split}")

        elif self.split_mode == "random":
            # random subject-agnostic split
            rng = np.random.RandomState(self.seed)
            indices = np.arange(len(entries))
            rng.shuffle(indices)

            split_idx = int(self.train_frac * len(entries))
            if self.split == "train":
                chosen_idx = indices[:split_idx]
            elif self.split in ("val", "test"):
                chosen_idx = indices[split_idx:]
            else:
                raise ValueError(f"Invalid split: {self.split}")

            self.samples = [entries[i] for i in chosen_idx]

        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

        print(
            f"[PopTSpeechDataset] brain_root = {self.brain_root}\n"
            f"[PopTSpeechDataset] manifest  = {self.manifest_path}\n"
            f"[PopTSpeechDataset] split     = {self.split} "
            f"(mode={self.split_mode}), samples = {len(self.samples)}"
        )

        if not self.samples:
            raise ValueError(f"No samples after split={self.split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        e = self.samples[i]
        subject = e["subject"]
        trial = e["trial"]
        idx = e["idx"]
        label = e["label"]

        # ---- Load brain window ----
        brain_path = self.brain_root / subject / trial / f"{idx}.npy"
        if not brain_path.exists():
            raise FileNotFoundError(f"Brain embedding not found: {brain_path}")

        brain = np.load(brain_path).astype(np.float32)  # (n_electrodes, 768)

        brain_t = torch.from_numpy(brain)  # (n_elec, d_model_input)
        label_t = torch.tensor(label, dtype=torch.long)

        meta = {
            "subject": subject,
            "trial": trial,
            "idx": idx,
        }

        # Note: no audio here (brain-only speech decoding)
        return brain_t, label_t, meta


def make_popt_speech_dataloaders_from_config(config: dict):
    """
    Helper to build train/val dataloaders for PopT speech decoding
    given a config dict.

    Expects in config["data"]:
        brain_root_speech: path to .../saved_examples/all_test_word_onset
        speech_manifest:    path to speech_labels.csv
        split_mode:         "by_trial" or "random"
        train_subject_trials: list of {subject, trials:[...]} for train
        val_subject_trials:   same for val

    and in config["training"]:
        batch_size, num_workers, seed, train_frac (if random split)
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    brain_root = Path(data_cfg["brain_root_speech"])
    manifest_path = Path(data_cfg["speech_manifest"])

    split_mode = data_cfg.get("split_mode", "by_trial")
    only_subjects = data_cfg.get("only_subjects", None)

    train_subject_trials = _expand_subject_trials(
        data_cfg.get("train_subject_trials", None)
    )
    val_subject_trials = _expand_subject_trials(
        data_cfg.get("val_subject_trials", None)
    )

    common_kwargs = dict(
        brain_root=brain_root,
        manifest_path=manifest_path,
        split_mode=split_mode,
        train_trials=train_subject_trials,
        val_trials=val_subject_trials,
        train_frac=data_cfg.get("train_frac", 0.9),
        seed=train_cfg.get("seed", 42),
        only_subjects=only_subjects,
    )

    dataset_train = PopTSpeechDataset(split="train", **common_kwargs)
    dataset_val = PopTSpeechDataset(split="val", **common_kwargs)

    batch_size = train_cfg.get("batch_size", 64)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
