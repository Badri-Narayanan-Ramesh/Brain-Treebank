# src/training/metrics.py

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    scores: (N,) predicted probabilities
    labels: (N,) binary 0/1
    """
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    if labels_np.min() == labels_np.max():
        # all labels same -> undefined AUC; return 0.5
        return 0.5

    if not _HAS_SK:
        # Very simple fallback: approximate via ranking
        order = np.argsort(scores_np)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(scores_np))
        pos = labels_np == 1
        neg = ~pos
        if pos.sum() == 0 or neg.sum() == 0:
            return 0.5
        sum_ranks_pos = ranks[pos].sum()
        n_pos = pos.sum()
        n_neg = neg.sum()
        auc = (sum_ranks_pos - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

    return float(roc_auc_score(labels_np, scores_np))


def compute_embedding_jitter(
    z_brain: torch.Tensor,
    meta: Optional[dict] = None,
) -> float:
    """
    Simple jitter metric: L2 difference between consecutive embeddings.

    z_brain: (N, d_model)
    """
    if z_brain.size(0) < 2:
        return 0.0

    diffs = z_brain[1:] - z_brain[:-1]  # (N-1, d)
    jitter = torch.sqrt((diffs ** 2).sum(dim=-1)).mean()
    return float(jitter.item())


def compute_retrieval_at_k_sampled(
    u: torch.Tensor,
    v: torch.Tensor,
    k: int = 2,
    num_queries: int = 1000,
    pool_size: int = 256,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Approximate retrieval@k via random subsampling.

    u, v: (N, d) brain/audio embeddings (normalized or not)

    Returns:
        (retrieval_at_k, chance_level)
    """
    assert u.shape == v.shape, f"Shape mismatch: {u.shape} vs {v.shape}"
    N = u.size(0)
    if N == 0:
        return 0.0, 0.0

    device = u.device
    rng = np.random.RandomState(seed)

    num_queries = min(num_queries, N)
    successes = 0
    total = 0

    for _ in range(num_queries):
        q_idx = int(rng.randint(0, N))
        # Choose pool_size-1 negatives + the positive index
        if N <= pool_size:
            pool_indices = np.arange(N)
        else:
            neg_indices = rng.choice(
                np.delete(np.arange(N), q_idx),
                size=pool_size - 1,
                replace=False,
            )
            pool_indices = np.concatenate([[q_idx], neg_indices])

        pool_indices_t = torch.from_numpy(pool_indices).to(device)
        u_q = u[q_idx : q_idx + 1]          # (1, d)
        v_pool = v[pool_indices_t]         # (P, d)

        scores = (u_q @ v_pool.t()).squeeze(0)  # (P,)
        # higher is better; rank the true index
        _, sorted_idx = torch.sort(scores, descending=True)
        # position of q_idx in sorted order
        # find index where pool_indices[sorted_idx[pos]] == q_idx
        ordered_pool = pool_indices_t[sorted_idx]
        pos = (ordered_pool == q_idx).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        rank = int(pos[0, 0])  # 0-based

        if rank < k:
            successes += 1
        total += 1

    if total == 0:
        return 0.0, 0.0

    retrieval_at_k = successes / total
    chance = float(min(k, pool_indices.shape[0]) / pool_indices.shape[0])
    return float(retrieval_at_k), float(chance)
