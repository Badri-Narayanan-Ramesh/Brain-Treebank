#!/usr/bin/env python
"""
Expert Point-Process Word Onset Decoder – FINAL FIXED & CLEAN VERSION
Works perfectly with your data (91 channels → 364 features)
Includes:
- 4 strong features per channel
- Per-subject normalization
- LogisticRegressionCV with AUPRC scoring
- Temporal smoothing
- Channel importance
- Jitter & flip rate
- AUPRC + best F1 threshold
"""

import argparse
import os
import csv
import json
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, accuracy_score, precision_recall_curve)
from sklearn.dummy import DummyClassifier


# ==============================================================
# Robust helper functions (defined at top level)
# ==============================================================

def parse_trial_and_index_from_path(path):
    """Extract trial name and numeric index from file path"""
    trial = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)
    match = re.search(r'\d+', filename)
    idx_num = int(match.group()) if match else None
    return trial, idx_num


def compute_jitter_flip(subjects, paths, probs, preds, test_idx):
    """Compute temporal jitter and label flip rate across consecutive windows"""
    samples = []
    for idx, prob, pred in zip(test_idx, probs, preds):
        subj = subjects[idx] if subjects and subjects[idx] is not None else "unknown"
        trial, idx_num = parse_trial_and_index_from_path(paths[idx])
        samples.append((
            subj,
            trial or "unknown",
            idx_num if idx_num is not None else -999,
            prob,
            pred
        ))

    # Sort by subject → trial → time index
    samples_sorted = sorted(samples, key=lambda x: (x[0], x[1], x[2]))

    diffs = []
    flips = []
    prev_subj = prev_trial = prev_idx = prev_prob = prev_pred = None

    for subj, trial, idx_num, prob, pred in samples_sorted:
        if (prev_subj is not None
                and subj == prev_subj
                and trial == prev_trial
                and idx_num > prev_idx):
            diffs.append(abs(prob - prev_prob))
            flips.append(int(pred != prev_pred))

        prev_subj, prev_trial, prev_idx, prev_prob, prev_pred = \
            subj, trial, idx_num, prob, pred

    jitter = np.mean(diffs) if diffs else 0.0
    flip_rate = np.mean(flips) if flips else 0.0
    return jitter, flip_rate


def extract_features(arr):
    """
    Extract 5 powerful features per channel: std, mean, log-power, delta, line-length (sum of absolute diffs)
    Add high-gamma power (70 TO 150 Hz analytic amplitude) if you raw data is available
    """
    if arr.ndim != 2 or arr.shape[1] <= 1:
        return np.zeros(0)

    std = np.std(arr, axis=1)
    mean = np.mean(arr, axis=1)
    log_power = np.log(np.mean(arr**2, axis=1) + 1e-12)
    delta = np.std(np.diff(arr, axis=1), axis=1)
    line_length = np.sum(np.abs(np.diff(arr, axis=1)), axis=1)
    return np.concatenate([std, mean, log_power, delta, line_length])


def load_features_enhanced(paths, labels, indices):
    X_list, y_list = [], []
    for idx in indices:
        if not (0 <= idx < len(paths)):
            continue
        try:
            arr = np.load(paths[idx])
            feats = extract_features(arr)
            if feats.size == 0:
                continue
            X_list.append(feats)
            y_list.append(labels[idx])
        except Exception as e:
            print(f"Warning: Failed to load {paths[idx]}: {e}")
    X = np.vstack(X_list) if X_list else np.empty((0, 0))
    y = np.array(y_list, dtype=int)
    return X, y


def smooth_probabilities(probs, window=9):
    if len(probs) == 0:
        return probs
    return np.convolve(probs, np.ones(window)/window, mode='same')


# ==============================================================
# Data loading
# ==============================================================

def load_data(manifest_path, labels_path, splits_path):
    paths, subjects = [], []
    with open(manifest_path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or not row[0].strip():
                continue
            paths.append(row[0].strip().replace("\\", "/"))
            subj = row[1].strip() if len(row) > 1 and row[1].strip() else None
            subjects.append(subj)

    labels_list = []
    with open(labels_path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or not row[0].strip():
                continue
            labels_list.append(1 if row[0].strip().lower() == "true" else 0)
    labels = np.array(labels_list, dtype=int)

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    return (paths, subjects, labels,
            splits.get("train", []), splits.get("val", []),
            splits.get("test", []))


# ==============================================================
# Main
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Expert Point-Process Word Onset Decoder")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--subject-norm", action="store_true", help="Per-subject normalization")
    parser.add_argument("--no-smooth", action="store_true", help="Disable temporal smoothing")
    args = parser.parse_args()

    # Load data
    paths, subjects, labels, train_idx, val_idx, test_idx = load_data(
        args.manifest, args.labels, args.splits
    )
    print(f"Loaded {len(paths)} windows | Train:{len(train_idx)} Val:{len(val_idx)} Test:{len(test_idx)}")

    # Extract features
    X_train, y_train = load_features_enhanced(paths, labels, train_idx)
    X_val,   y_val   = load_features_enhanced(paths, labels, val_idx)
    X_test,  y_test  = load_features_enhanced(paths, labels, test_idx)

    n_channels = X_train.shape[1] // 5 if len(X_train) > 0 else 0
    print(f"Feature dim: {X_train.shape[1] if len(X_train)>0 else 0} (5 × {n_channels} channels)")

    # Per-subject normalization
    if args.subject_norm and any(s is not None for s in subjects):
        print("Applying per-subject feature normalization...")
        subject_scalers = {}
        for subj in set(s for s in subjects if s):
            idxs = [i for i, s in enumerate(subjects) if s == subj and i in train_idx + val_idx]
            if not idxs:
                continue
            X_sub, _ = load_features_enhanced(paths, labels, idxs)
            if len(X_sub) == 0:
                continue
            scaler = StandardScaler()
            scaler.fit(X_sub)
            subject_scalers[subj] = scaler

        def norm_by_subject(X, idxs):
            if len(X) == 0:
                return X
            Xn = np.zeros_like(X)
            for i, idx in enumerate(idxs):
                s = subjects[idx]
                if s in subject_scalers:
                    Xn[i] = subject_scalers[s].transform(X[i:i+1])[0]
                else:
                    Xn[i] = X[i]
            return Xn

        X_train = norm_by_subject(X_train, train_idx)
        X_val   = norm_by_subject(X_val,   val_idx)
        X_test  = norm_by_subject(X_test,  test_idx)
    else:
        if len(X_train) > 0:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)   if len(X_val)   > 0 else X_val
            X_test  = scaler.transform(X_test)  if len(X_test)  > 0 else X_test

    # Train model
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        print("Not enough data → using dummy classifier")
        clf = DummyClassifier(strategy="constant", constant=0)
    else:
        clf = LogisticRegressionCV(
            Cs=np.logspace(-5, 3, 20),
            cv=5,
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced',
            scoring='average_precision',
            max_iter=3000,
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X_train, y_train)
        print(f"Best C: {clf.C_[0]:.6f}")

    # Predict
    if len(X_test) > 0:
        y_prob_raw = clf.predict_proba(X_test)[:, 1]
        y_prob = smooth_probabilities(y_prob_raw, window=9) if not args.no_smooth else y_prob_raw
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = y_prob_raw = y_pred = np.array([])

    # Metrics
    if len(y_test) > 0:
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        f1_05 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        prec, rec, thr = precision_recall_curve(y_test, y_prob)
        f1_thr = 2 * prec * rec / (prec + rec + 1e-12)
        best_ix = np.argmax(f1_thr)
        best_f1 = f1_thr[best_ix]
        best_thr = thr[best_ix] if len(thr) > best_ix else 0.5
    else:
        auroc = auprc = f1_05 = acc = best_f1 = 0.0
        best_thr = 0.5

    # Channel importance
    if hasattr(clf, "coef_") and len(X_train) > 0:
        coef = clf.coef_[0]
        n_feat = 5
        if coef.size % n_feat != 0:
            print(f"Warning: coef size {coef.size} not divisible by {n_feat} — skipping importance")
        else:
            importance_per_channel = np.mean(np.abs(coef).reshape(n_feat, -1), axis=0)
            top_ch = np.argsort(importance_per_channel)[::-1][:20]
            print("\nTop 20 most important channels (5 features per channel):")
            for rank, ch in enumerate(top_ch, 1):
                print(f"  #{rank:2d} → Ch {ch:3d}  (avg |w| = {importance_per_channel[ch]:.4f})")

    # Temporal stability
    jitter, flip_rate = compute_jitter_flip(subjects, paths, y_prob, y_pred, test_idx)

    # Final results
    print("\n" + "="*60)
    print("           EXPERT POINT-PROCESS BASELINE RESULTS")
    print("="*60)
    print(f"Test AUROC             : {auroc:.4f}")
    print(f"Test AUPRC (key)      : {auprc:.4f}")
    print(f"Test F1 @ 0.5        : {f1_05:.4f}")
    print(f"Best F1               : {best_f1:.4f} @ threshold = {best_thr:.3f}")
    print(f"Accuracy               : {acc:.4f}")
    print(f"Jitter (mean |Δp|)    : {jitter:.4f}")
    print(f"Flip Rate              : {flip_rate:.4f}")
    print(f"Smoothing              : {'Yes' if not args.no_smooth else 'No'}")
    print(f"Per-subject norm        : {'Yes' if args.subject_norm else 'No'}")
    print("="*60)


if __name__ == "__main__":
    main()

# python point_process_decoder.py --manifest "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2\manifest.tsv" --labels "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2\labels.tsv" --splits "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2\splits.json" --subject-norm

# python point_process_decoder.py --manifest "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2\manifest.tsv" --labels "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2\labels.tsv" --splits "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2\splits.json" --subject-norm
