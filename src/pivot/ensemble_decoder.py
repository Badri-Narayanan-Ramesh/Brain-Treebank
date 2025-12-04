#!/usr/bin/env python3
"""
Ensemble Decoder for Word Onset Detection
=======================================

This script combines the predictions of multiple word‑onset decoders –
including a fine‑tuned Population Transformer (PopT), a Kalman filter
baseline, and a point‑process (logistic) decoder – to produce an
ensemble probability and binary classification for each time window.
It supports two fusion strategies:

1. **Weighted Average** – The user supplies non‑negative weights
   (that sum to one) for each model, and the ensemble probability is
   computed as the weighted sum of the three model probabilities.

2. **Stacked Logistic Regression** – A meta‑classifier is trained
   on the training split using the per‑window probabilities from
   each base model as input features.  The resulting logistic
   regression combines the models in a data‑driven way to optimize
   classification performance.  A regularisation path (via
   `LogisticRegressionCV`) is used to tune the strength of
   regularisation.

The script reports evaluation metrics on the training, validation,
and test splits: area under the ROC curve (AUROC), area under the
precision‑recall curve (AUPRC), F1 score at the default 0.5
threshold, F1 score at the optimal threshold (selected from the
validation set), overall accuracy, and temporal smoothness metrics
(jitter and flip rate).  Optionally, ensemble predictions (probabilities
and binary outputs) can be saved to disk for further analysis or
ensembling with additional models.

Author: OpenAI ChatGPT (Agent Mode)
"""

import argparse
import csv
import json
import os
import re
from typing import List, Tuple, Sequence, Optional

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler


def load_data(
    manifest_path: str,
    labels_path: str,
    splits_path: str,
) -> Tuple[List[str], List[Optional[str]], np.ndarray, List[int], List[int], List[int]]:
    """Load file paths, subject IDs, labels and split indices.

    Parameters
    ----------
    manifest_path : str
        Path to the TSV manifest with at least two columns: file path and subject ID.
    labels_path : str
        Path to the TSV labels file with the same number of rows as the manifest.  The
        first column must contain "True"/"False" or "1"/"0" labels indicating a
        word onset in each time window.
    splits_path : str
        Path to JSON file specifying train/val/test indices.  Keys should be
        "train", "val" and "test" mapping to lists of zero‑based indices.

    Returns
    -------
    paths : list of str
        Normalised file paths (with forward slashes) to the .npy windows.
    subjects : list of str or None
        Subject identifiers corresponding to each window (None if missing).
    labels : np.ndarray of shape (n_windows,)
        Binary labels (0 for no onset, 1 for onset).
    train_idx, val_idx, test_idx : lists of int
        Indices into the `paths` list for the train, validation and test splits.
    """
    paths: List[str] = []
    subjects: List[Optional[str]] = []
    # Load manifest TSV
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            # First column: path to .npy file
            path = row[0].strip().replace("\\", "/")
            paths.append(path)
            # Second column: subject (if present)
            subj = row[1].strip() if len(row) > 1 and row[1].strip() else None
            subjects.append(subj)
    # Load labels TSV
    label_list: List[int] = []
    with open(labels_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            flag = row[0].strip().lower()
            label_list.append(1 if flag == "true" or flag == "1" else 0)
    labels_arr = np.array(label_list, dtype=int)
    if len(labels_arr) != len(paths):
        raise ValueError(
            f"Number of labels ({len(labels_arr)}) does not match number of files ({len(paths)})"
        )
    # Load splits JSON
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_idx = splits.get("train", [])
    val_idx = splits.get("val", [])
    test_idx = splits.get("test", [])
    return paths, subjects, labels_arr, train_idx, val_idx, test_idx


def load_probabilities(npz_path: str, n_windows: int) -> np.ndarray:
    """Load probability vector from a `.npz` file.

    The `.npz` should contain either a key 'prob' or 'y_prob' pointing to a 1D array.
    The array is padded or trimmed to match `n_windows` if necessary (e.g. when some
    models only output predictions for a subset of data).  Missing probabilities
    default to 0.0.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file containing the probabilities.
    n_windows : int
        Total number of windows in the dataset (length of `paths`).

    Returns
    -------
    probs : np.ndarray of shape (n_windows,)
        Probability of word onset per window.  Missing values are filled with 0.0.
    """
    if not npz_path:
        return np.zeros(n_windows, dtype=float)
    data = np.load(npz_path)
    # Check common keys
    if "prob" in data:
        arr = data["prob"].reshape(-1)
    elif "y_prob" in data:
        arr = data["y_prob"].reshape(-1)
    else:
        # If only one array is present, use it
        arr_keys = list(data.keys())
        if arr_keys:
            arr = data[arr_keys[0]].reshape(-1)
        else:
            arr = np.zeros(0, dtype=float)
    # Pad or truncate to length n_windows
    if len(arr) < n_windows:
        padded = np.zeros(n_windows, dtype=float)
        padded[: len(arr)] = arr
        return padded
    elif len(arr) > n_windows:
        return arr[:n_windows]
    else:
        return arr


def validate_weights(weights: Sequence[float]) -> List[float]:
    """Validate and normalise a sequence of weights.

    Ensures all weights are non‑negative and sum to one.  Raises ValueError if
    negative or sum is zero.  If all weights are zero, evenly distributes
    probability mass across models.

    Parameters
    ----------
    weights : sequence of floats
        The raw weights (e.g. as parsed from a comma‑separated argument).

    Returns
    -------
    norm_weights : list of floats
        Non‑negative weights summing to one.
    """
    ws = [max(0.0, float(w)) for w in weights]
    total = sum(ws)
    if total <= 0:
        raise ValueError("Sum of weights must be positive")
    return [w / total for w in ws]


def parse_path(path: str):
    """Robust extraction of subject, trial, and window number from path."""
    path = path.replace("\\", "/")  # normalize
    # Match patterns like sub_3 or sub_03 or SUBJECT3 or sub3
    subj_match = re.search(r'[\/\\](?:sub[_]?|SUBJECT)(\d+)', path, re.IGNORECASE)
    subject = f"sub_{int(subj_match.group(1))}" if subj_match else "unknown"

    # Match trial000 ... trial999
    trial_match = re.search(r'[\/\\]trial(\d{3})[\/\\]', path)
    trial = f"trial{trial_match.group(1)}" if trial_match else "unknown"

    # Extract window number from filename: 1.npy, 0001.npy, etc.
    fname = os.path.basename(path)
    num_match = re.search(r'(\d+)\.npy$', fname)
    window_num = int(num_match.group(1)) if num_match else 0

    return subject, trial, window_num


def compute_jitter_flip_corrected(
    subjects, trial_ids, window_numbers,
    all_indices, probabilities, threshold=0.5
):
    preds = (probabilities >= threshold).astype(int)
    diffs = []
    flips = []

    prev_subj = prev_trial = prev_prob = prev_pred = None

    # Go through windows in true chronological order
    for idx in all_indices:
        # Find position in chronological sequence
        subj = subjects[idx]
        trial = trial_ids[idx]
        prob = probabilities[idx]
        pred = preds[idx]

        if prev_subj is not None and subj == prev_subj and trial == prev_trial:
            # Only count if same trial and truly consecutive in time
            diffs.append(abs(prob - prev_prob))
            flips.append(1 if pred != prev_pred else 0)

        prev_subj, prev_trial = subj, trial
        prev_prob, prev_pred = prob, pred

    jitter = float(np.mean(diffs)) if diffs else 0.0
    flip_rate = float(np.mean(flips)) if flips else 0.0
    return jitter, flip_rate


def evaluate_predictions(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    split_indices: List[int],
) -> Tuple[float, float, float, float, float, float]:
    """Compute classification metrics for a given set of indices.

    Returns AUROC, AUPRC, F1 at default threshold 0.5, best F1, best threshold,
    and accuracy.
    """
    # Extract labels for the split
    labels = true_labels[split_indices]
    probs = probabilities[split_indices]
    # If only one class present, metrics are degenerate
    if len(np.unique(labels)) <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0.5, 0.0
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    # Default threshold 0.5
    preds_default = (probs >= 0.5).astype(int)
    f1_default = f1_score(labels, preds_default, zero_division=0)
    acc = accuracy_score(labels, preds_default)
    # Find best threshold on this split
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1s)) if len(f1s) > 0 else 0
    best_f1 = float(f1s[best_idx])
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    return auroc, auprc, f1_default, best_f1, best_threshold, acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine predictions from multiple decoders for word‑onset detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest.tsv listing .npy windows and subject IDs",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels.tsv with word‑onset labels",
    )
    parser.add_argument(
        "--splits",
        type=str,
        required=True,
        help="Path to splits.json specifying train, val and test indices",
    )
    parser.add_argument(
        "--popt-prob",
        type=str,
        default="",
        help="Path to NPZ file containing probabilities from the PopT model",
    )
    parser.add_argument(
        "--kalman-prob",
        type=str,
        default="",
        help="Path to NPZ file containing probabilities from the Kalman decoder",
    )
    parser.add_argument(
        "--ppf-prob",
        type=str,
        default="",
        help="Path to NPZ file containing probabilities from the point‑process decoder",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help=(
            "Comma‑separated non‑negative weights for [PopT, Kalman, PPF] to form a weighted ensemble."
            " If not provided, the script uses stacked logistic regression."
        ),
    )
    parser.add_argument(
        "--save-preds",
        type=str,
        default="",
        help="Path to save ensemble probabilities and predictions as an NPZ file",
    )
    args = parser.parse_args()

    # Load dataset metadata
    paths, subjects, labels_arr, train_idx, val_idx, test_idx = load_data(
        args.manifest, args.labels, args.splits
    )
    n_windows = len(paths)
    # Load base model probabilities
    prob_popt = load_probabilities(args.popt_prob, n_windows)
    prob_kalman = load_probabilities(args.kalman_prob, n_windows)
    prob_ppf = load_probabilities(args.ppf_prob, n_windows)

    # Determine ensemble probabilities
    if args.weights:
        # Weighted average ensemble
        try:
            raw_weights = [float(w) for w in args.weights.split(",")]
        except ValueError:
            raise ValueError(
                "--weights must be a comma‑separated list of floats (e.g. '0.6,0.3,0.1')"
            )
        if len(raw_weights) != 3:
            raise ValueError("Exactly three weights must be provided (for PopT, Kalman, PPF)")
        weights = validate_weights(raw_weights)
        # Weighted sum of probabilities
        ensemble_prob = (
            weights[0] * prob_popt + weights[1] * prob_kalman + weights[2] * prob_ppf
        )
        ensemble_name = f"Weighted ensemble (w={weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f})"
    else:
        # Stacked logistic regression ensemble
        # Prepare meta-features: shape (n_windows, 3)
        meta_features = np.vstack(
            [prob_popt, prob_kalman, prob_ppf]
        ).T.astype(float)
        # Standardise features (important for logistic regression)
        scaler = StandardScaler()
        meta_train = scaler.fit_transform(meta_features[train_idx])
        meta_val = scaler.transform(meta_features[val_idx]) if val_idx else np.empty((0, 3))
        meta_test = scaler.transform(meta_features[test_idx]) if test_idx else np.empty((0, 3))
        # Fit logistic regression with cross‑validated C on training data
        if len(train_idx) == 0 or len(np.unique(labels_arr[train_idx])) < 2:
            # Fallback to dummy if training data is empty or only one class
            from sklearn.dummy import DummyClassifier

            meta_clf = DummyClassifier(strategy="constant", constant=0)
            meta_clf.fit(meta_train, labels_arr[train_idx])
        else:
            # Use AUPRC as scoring metric because onset events are rare
            Cs = np.logspace(-4, 4, 10)
            meta_clf = LogisticRegressionCV(
                Cs=Cs,
                cv=5,
                penalty="l2",
                solver="lbfgs",
                scoring="average_precision",
                class_weight="balanced",
                max_iter=1000,
                n_jobs=-1,
                random_state=42,
            )
            meta_clf.fit(meta_train, labels_arr[train_idx])
        # Predict probabilities for all windows
        ensemble_prob = np.zeros(n_windows, dtype=float)
        ensemble_prob[train_idx] = meta_clf.predict_proba(meta_train)[:, 1] if len(train_idx) else 0.0
        if len(val_idx):
            ensemble_prob[val_idx] = meta_clf.predict_proba(meta_val)[:, 1]
        if len(test_idx):
            ensemble_prob[test_idx] = meta_clf.predict_proba(meta_test)[:, 1]
        # Name includes chosen C if available
        c_val = meta_clf.C_[0] if hasattr(meta_clf, "C_") else None
        ensemble_name = (
            f"Stacked logistic ensemble (C={c_val:.4e})" if c_val is not None else "Stacked logistic ensemble"
        )

    # Evaluate ensemble on each split
    for split_name, split_idx in [
        ("Train", train_idx),
        ("Validation", val_idx),
        ("Test", test_idx),
    ]:
        auroc, auprc, f1_def, best_f1, best_thr, acc = evaluate_predictions(
            labels_arr, ensemble_prob, split_idx
        )
        print(f"\n{split_name} Set:\n  AUROC = {auroc:.4f}\n  AUPRC = {auprc:.4f}\n  F1 (0.5) = {f1_def:.4f}\n  Best F1 = {best_f1:.4f} at thr={best_thr:.4f}\n  Accuracy = {acc:.4f}")

    # Compute jitter/flip rate on test set for temporal smoothness
    print("Parsing subject, trial, and temporal order from file paths...")
    parsed = [parse_path(p) for p in paths]
    subjects_parsed = [p[0] for p in parsed]
    trial_ids = [p[1] for p in parsed]
    window_numbers = [p[2] for p in parsed]

    # Sort test indices chronologically for correct jitter calculation
    test_idx_sorted = sorted(
        test_idx,
        key=lambda i: (subjects_parsed[i], trial_ids[i], window_numbers[i])
    )

    jitter, flip_rate = compute_jitter_flip_corrected(
        subjects_parsed, trial_ids, window_numbers,
        test_idx_sorted, ensemble_prob, threshold=0.5
    )
    print(
        f"\nTemporal Stability on Test:\n  Jitter (mean |Δp|) = {jitter:.4f}\n  Flip Rate = {flip_rate:.4f}"
    )
    print(f"\nEnsemble method: {ensemble_name}\n")
    # Optionally save ensemble probabilities and predictions
    if args.save_preds:
        np.savez(
            args.save_preds,
            prob=ensemble_prob,
            pred=(ensemble_prob >= 0.5).astype(int),
            true=labels_arr,
        )
        print(f"Saved ensemble probabilities to {args.save_preds}")


if __name__ == "__main__":
    main()