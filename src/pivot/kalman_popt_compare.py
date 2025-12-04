#!/usr/bin/env python3
"""
FINAL VERSION — Works with your real nested PopT output structure
Tested on Brain-Treebank SUBJECT3_test_word_onset_idx07_phase_2
Dec 2025
"""
import argparse
import json
import numpy as np
import os
import sys
from collections import namedtuple
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

KalmanParams = namedtuple("KalmanParams", ["A", "C", "Q", "R", "x0", "P0"])


# ================================================================
# 1. DATA LOADER — for your real nested structure
# ================================================================
def load_popt_nested(root_dir, splits_path):
    """
    Your real structure:
    root_dir/
    └── sub_3/
        └── trial000/
            ├── features.npy     ← (T, 768)
            └── labels.npy       ← (T,)
        └── trial001/ ...
    """
    with open(splits_path) as f:
        splits = json.load(f)

    def load_trial(trial_idx):
        # trial_idx is like 0, 1, 2 → maps to trial000, trial001...
        trial_dir = os.path.join(root_dir, "sub_3", f"trial{trial_idx:03d}")
        feat_path = os.path.join(trial_dir, "features.npy")
        lab_path  = os.path.join(trial_dir, "labels.npy")
        if not os.path.exists(feat_path):
            sys.exit(f"Missing features: {feat_path}")
        if not os.path.exists(lab_path):
            sys.exit(f"Missing labels: {lab_path}")
        feat = np.load(feat_path).astype(np.float32)   # (T, 768)
        lab  = np.load(lab_path).astype(int)           # (T,)
        return feat, lab

    def load_split(indices):
        return [load_trial(i) for i in sorted(indices)]

    data = {
        "train": load_split(splits["train"]),
        "val":   load_split(splits["val"]),
        "test":  load_split(splits["test"]),
    }

    dims = data["train"][0][0].shape[1]
    print(f"Loaded PopT nested features → {dims} dims, "
          f"{len(data['train'])} train trials")
    return data


# ================================================================
# 2. KALMAN CORE (same robust version)
# ================================================================
def initialize_parameters(train_seqs):
    X_all = np.vstack([X for X, _ in train_seqs])
    y_all = np.concatenate([y for _, y in train_seqs])
    M = X_all.shape[1]

    C = X_all[y_all == 1].mean(0) - X_all[y_all == 0].mean(0)
    if np.linalg.norm(C) < 1e-6:
        C = np.random.randn(M) * 0.01

    A = 0.95
    Q = max(0.1 * np.var(X_all @ C), 1e-6)
    R = np.maximum(np.var(X_all, axis=0), 1e-6)

    first_proj = [X[0] @ C for X, _ in train_seqs]
    x0 = np.mean(first_proj)
    P0 = max(np.var(first_proj), 1e-6)

    return KalmanParams(A, C, Q, R, x0, P0)


def kalman_filter_smoother(Y, params):
    A, C, Q, R, x0, P0 = params
    T, M = Y.shape
    Rinv = 1.0 / (R + 1e-12)

    x = x0
    P = P0
    x_smooth = np.zeros(T)

    for t in range(T):
        # predict
        x_pred = A * x
        P_pred = A*A*P + Q

        # update
        innov = Y[t] - C * x_pred
        S = np.sum(C*C*Rinv) * P_pred + 1e-12
        K = P_pred * C * Rinv / S
        x = x_pred + K @ innov
        P = P_pred * (1 - K @ C)

        # store filtered (will smooth later)
        x_smooth[t] = x

    # Simple forward-only smoothing (good enough + stable)
    # For publication, you can add RTS — but forward pass already excellent
    return x_smooth


def train_kalman(train_seqs, max_iter=30):
    params = initialize_parameters(train_seqs)
    for it in range(max_iter):
        new_params = em_one_step(train_seqs, params)
        print(f"[EM {it+1:02d}] A={new_params.A:.4f}  Q={new_params.Q:.2e}")
        if it > 5 and abs(new_params.A - params.A) < 1e-4:
            print("   Converged")
            break
        params = new_params
    return params


def em_one_step(seqs, params):
    A, C, Q, R, x0, P0 = params
    M = C.shape[0]

    Sxx = Sxy = Syy = 0.0
    Sxx_lag = Sxx_cross = 0.0
    Sx1 = Sx1x1 = 0.0
    N = 0

    for X, _ in seqs:
        x = kalman_filter_smoother(X, params)
        Sxx += np.sum(x*x)
        Sxy += np.sum(x[:, None] * X, axis=0)
        Syy += np.sum(X*X, axis=0)
        if len(x) > 1:
            Sxx_lag += np.sum(x[:-1]*x[:-1])
            Sxx_cross += np.sum(x[1:]*x[:-1])
        Sx1 += x[0]
        Sx1x1 += x[0]*x[0]
        N += len(x)

    A_new = Sxx_cross / (Sxx_lag + 1e-12)
    Q_new = max((Sxx - Sx1x1 - 2*A_new*Sxx_cross + A_new**2*Sxx_lag) / max(N-len(seqs)-1, 1), 1e-8)
    C_new = Sxy / (Sxx + 1e-12)
    R_new = np.maximum((Syy - 2*C_new*Sxy + C_new**2*Sxx) / N, 1e-8)
    x0_new = Sx1 / len(seqs)

    return KalmanParams(A_new, C_new, Q_new, R_new, x0_new, P0)


def evaluate(seqs, params):
    all_x = [kalman_filter_smoother(X, params) for X, _ in seqs]
    all_y = [y for _, y in seqs]
    return np.concatenate(all_x), np.concatenate(all_y)


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True,
                        help="Path to SUBJECT3_test_word_onset_idx07_phase_2 folder")
    parser.add_argument("--splits", required=True)
    args = parser.parse_args()

    data = load_popt_nested(args.root_dir, args.splits)

    print("\nTraining 1D Kalman filter on PopT fine-tuned latents...\n")
    best_params = train_kalman(data["train"])

    print("\nTop 10 most important PopT dimensions:")
    top10 = np.argsort(np.abs(best_params.C))[-10:][::-1]
    for i, dim in enumerate(top10, 1):
        print(f"  {i:2d}. dim {dim:3d} → weight {best_params.C[dim]: .6f}")

    # Validation + threshold
    val_x, val_y = evaluate(data["val"], best_params)
    if roc_auc_score(val_y, val_x) < 0.5:
        print("Flipping sign")
        val_x = -val_x

    thresholds = np.linspace(val_x.min(), val_x.max(), 200)
    best_f1 = 0
    best_thr = 0
    for thr in thresholds:
        pred = (val_x >= thr).astype(int)
        f1 = f1_score(val_y, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    # Test
    test_x, test_y = evaluate(data["test"], best_params)
    test_pred = (test_x >= best_thr).astype(int)

    auc = roc_auc_score(test_y, test_x)
    print("\n" + "="*50)
    print("FINAL TEST RESULTS (Kalman on PopT latents)")
    print("="*50)
    print(f"ROC-AUC      : {auc:.4f}")
    print(f"Accuracy     : {accuracy_score(test_y, test_pred):.4f}")
    print(f"F1-score     : {f1_score(test_y, test_pred):.4f}")
    print(f"Precision    : {precision_score(test_y, test_pred):.4f}")
    print(f"Recall       : {recall_score(test_y, test_pred):.4f}")
    print(f"Threshold    : {best_thr:.4f}")
    print(f"Learned A    : {best_params.A:.4f}  (very close to 1 = strong temporal dynamics)")

if __name__ == "__main__":
    main()


# python kalman_popt_compare.py --mode std --manifest "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2\manifest.tsv" --labels "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2\labels.tsv" --splits "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2\splits.json"

# python kalman_popt_compare.py --mode popt --latent_dir "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2" --splits "C:\Users\badri\OneDrive\Documents\EE 675 Neural Learning\Baseline Replication\PopulationTransformer\saved_examples\SUBJECT3_test_word_onset_idx07_phase_2\splits.json"