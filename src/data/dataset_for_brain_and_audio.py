# src/data/dataset_for_brain_and_audio.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


TRIAL_TO_MOVIE = {
    "sub_1": {
        "trial000": "fantastic-mr-fox",
        "trial002": "thor-ragnarok",
    },
    "sub_2": {
        "trial000": "aquaman",
        "trial001": "avengers-infinity-war",
    },
    # extend as needed
}


class BrainAudioDataset(Dataset):
    """
    Dataset that pairs PopT NSP brain windows (.npy) with
    proxy 'audio' features (precomputed numpy arrays).

    Brain:
        brain_root / sub_X / trialYYY / k.npy
        each k.npy: (n_electrodes, 768)

    Audio proxy:
        audio_root / {movie_key}_proxy_features.npy
        where movie_key ∈ TRIAL_TO_MOVIE[sub][trial]
        shape: (T, feat_dim)  e.g. feat_dim = 3

    Window assumptions:
        brain window duration ~ 5.0s, step ~ 0.2s
        audio proxy fps ~ 100
        audio_window_sec ~ 5.0s
    """

    def __init__(
        self,
        brain_root: Path,
        audio_root: Path,
        duration_sec: float = 5.0,
        step_sec: float = 0.2,
        audio_fps: int = 100,
        audio_window_sec: float = 5.0,
        audio_offset_sec: float = 0.15,   # lag for iEEG delay
        split: str = "train",             # "train" / "val"
        split_mode: str = "random",       # "random" / "by_trial"
        train_trials: Optional[List[Tuple[str, str]]] = None,
        val_trials: Optional[List[Tuple[str, str]]] = None,
        train_frac: float = 0.9,
        seed: int = 42,
        only_subjects: Optional[List[str]] = None,
        speech_threshold: float = 0.5,
        min_speech_fraction: float = 0.0,
    ) -> None:
        super().__init__()

        print(f"[BrainAudioDataset] brain_root = {brain_root}")
        print(f"[BrainAudioDataset] audio_root = {audio_root}")

        self.brain_root = Path(brain_root)
        self.audio_root = Path(audio_root)
        self._proxy_cache: Dict[str, np.ndarray] = {}

        self.duration_sec = float(duration_sec)
        self.step_sec = float(step_sec)
        self.audio_fps = int(audio_fps)
        self.audio_window_sec = float(audio_window_sec)
        self.audio_offset_sec = float(audio_offset_sec)

        self.split = split
        self.split_mode = split_mode
        self.train_frac = float(train_frac)
        self.seed = int(seed)

        self.speech_threshold = float(speech_threshold)
        self.speech_min_fraction = float(min_speech_fraction)

        if only_subjects is None:
            self.only_subjects = set(TRIAL_TO_MOVIE.keys())
        else:
            self.only_subjects = set(only_subjects)

        # -------------------------
        # 1) Collect all brain windows
        # -------------------------
        self.samples: List[Dict] = []

        for subject in self.only_subjects:
            subj_dir = self.brain_root / subject
            if not subj_dir.is_dir():
                continue

            for trial in os.listdir(subj_dir):
                trial_dir = subj_dir / trial
                if not trial_dir.is_dir():
                    continue

                npy_files = list(trial_dir.glob("*.npy"))
                if not npy_files:
                    continue

                for f in npy_files:
                    try:
                        idx = int(f.stem)
                    except ValueError:
                        continue
                    self.samples.append(
                        {"subject": subject, "trial": trial, "idx": idx}
                    )

        print(f"[BrainAudioDataset] Found {len(self.samples)} brain windows")
        if not self.samples:
            raise ValueError("No brain .npy files found in brain_root")

        # -------------------------
        # 2) Split
        # -------------------------
        if self.split_mode == "random":
            np.random.seed(self.seed)
            np.random.shuffle(self.samples)
            split_idx = int(self.train_frac * len(self.samples))
            if self.split == "train":
                self.samples = self.samples[:split_idx]
            elif self.split == "val":
                self.samples = self.samples[split_idx:]
            else:
                raise ValueError(f"Invalid split '{self.split}'")

        elif self.split_mode == "by_trial":
            train_set = set(train_trials) if train_trials else set()
            val_set = set(val_trials) if val_trials else set()

            if self.split == "train":
                self.samples = [
                    s for s in self.samples if (s["subject"], s["trial"]) in train_set
                ]
            elif self.split == "val":
                self.samples = [
                    s for s in self.samples if (s["subject"], s["trial"]) in val_set
                ]
            else:
                raise ValueError(f"Invalid split '{self.split}'")
        else:
            raise ValueError(f"Unknown split_mode '{self.split_mode}'")

        print(f"[BrainAudioDataset] {self.split} samples: {len(self.samples)}")

    def _center_time(self, idx: int) -> float:
        return idx * self.step_sec + self.duration_sec / 2.0

    def _get_audio_window(self, proxy: np.ndarray, t_center: float) -> np.ndarray:
        """
        Extract a window centered at t_center + offset.
        """
        fps = self.audio_fps
        win_frames = int(round(self.audio_window_sec * fps))
        c = int(round((t_center + self.audio_offset_sec) * fps))
        s = max(0, c - win_frames // 2)
        e = s + win_frames

        if e > proxy.shape[0]:
            pad = np.zeros((e - proxy.shape[0], proxy.shape[1]), dtype=proxy.dtype)
            proxy_padded = np.concatenate([proxy, pad], axis=0)
        else:
            proxy_padded = proxy

        return proxy_padded[s:e]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        subject = s["subject"]
        trial = s["trial"]
        idx = s["idx"]

        # ---- brain ----
        brain_path = self.brain_root / subject / trial / f"{idx}.npy"
        if not brain_path.is_file():
            raise FileNotFoundError(f"Brain embedding not found: {brain_path}")

        brain = np.load(brain_path).astype(np.float32)  # (n_elec, 768)

        # ---- movie ----
        if subject not in TRIAL_TO_MOVIE:
            raise KeyError(f"Subject {subject} not in TRIAL_TO_MOVIE mapping")
        sub_map = TRIAL_TO_MOVIE[subject]
        if trial not in sub_map:
            raise KeyError(
                f"Trial {trial} not in TRIAL_TO_MOVIE mapping for {subject}"
            )
        movie_key = sub_map[trial]

        # ---- audio proxy ----
        if movie_key not in self._proxy_cache:
            proxy_path = self.audio_root / f"{movie_key}_proxy_features.npy"
            if not proxy_path.is_file():
                print(
                    "DEBUG — missing proxy:",
                    "subject:", subject,
                    "trial:", trial,
                    "movie:", movie_key,
                    "path:", proxy_path,
                )
                raise FileNotFoundError(f"Proxy features not found: {proxy_path}")
            proxy = np.load(proxy_path).astype(np.float32)
            self._proxy_cache[movie_key] = proxy
            self.audio_feat_dim = proxy.shape[-1]
        else:
            proxy = self._proxy_cache[movie_key]

        t_center = self._center_time(idx)
        audio_win = self._get_audio_window(proxy, t_center)  # (T_win, feat_dim)

        # derive label from first channel (speech mask)
        speech_chan = audio_win[:, 0]  # (T_win,)
        speech_frac = speech_chan.mean()
        if speech_frac < self.speech_min_fraction:
            speech_label = 0
        else:
            speech_label = int(speech_frac >= self.speech_threshold)

        brain_t = torch.from_numpy(brain)      # (n_elec, 768)
        audio_t = torch.from_numpy(audio_win)  # (T_win, feat_dim)
        label_t = torch.tensor(speech_label, dtype=torch.long)

        meta = {
            "subject": subject,
            "trial": trial,
            "idx": idx,
            "movie": movie_key,
            "t_center": t_center,
        }

        return brain_t, audio_t, label_t, meta


def _expand_subject_trials(cfg_list):
    pairs: List[Tuple[str, str]] = []
    if cfg_list is None:
        return pairs
    for item in cfg_list:
        subj = item["subject"]
        for t in item["trials"]:
            pairs.append((subj, t))
    return pairs


def make_dataloaders_from_config(config: dict):
    data_cfg = config["data"]
    train_cfg = config["training"]

    brain_root = Path(data_cfg["brain_root"])
    audio_root = Path(data_cfg["audio_proxy_root"])

    split_mode = data_cfg.get("split_mode", "random")
    only_subjects = data_cfg.get("only_subjects", None)

    train_subject_trials = _expand_subject_trials(
        data_cfg.get("train_subject_trials", None)
    )
    val_subject_trials = _expand_subject_trials(
        data_cfg.get("val_subject_trials", None)
    )

    speech_threshold = float(data_cfg.get("speech_threshold", 0.5))
    min_speech_fraction = float(data_cfg.get("speech_min_fraction", 0.0))
    audio_offset_sec = float(data_cfg.get("audio_offset_sec", 0.15))

    common_kwargs = dict(
        brain_root=brain_root,
        audio_root=audio_root,
        duration_sec=data_cfg.get("brain_window_sec", 5.0),
        step_sec=data_cfg.get("brain_window_step", 0.2),
        audio_fps=data_cfg.get("audio_fps", 100),
        audio_window_sec=data_cfg.get("audio_window_sec", 5.0),
        audio_offset_sec=audio_offset_sec,
        train_frac=data_cfg.get("train_frac", 0.9),
        seed=train_cfg.get("seed", 42),
        only_subjects=only_subjects,
        split_mode=split_mode,
        train_trials=train_subject_trials,
        val_trials=val_subject_trials,
        speech_threshold=speech_threshold,
        min_speech_fraction=min_speech_fraction,
    )

    dataset_train = BrainAudioDataset(split="train", **common_kwargs)
    dataset_val = BrainAudioDataset(split="val", **common_kwargs)

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 4))

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
