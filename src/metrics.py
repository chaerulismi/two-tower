"""
src/metrics.py
--------------
Retrieval evaluation metrics for the two-tower model.

Workflow
--------
1. Build a FAISS index over all item embeddings (build_faiss_index).
2. For each user query, retrieve top-K candidates (retrieve_top_k).
3. Compute ranking metrics against the ground-truth positives.

Metrics implemented
-------------------
- Recall@K   : fraction of relevant items that appear in the top-K retrieved.
- NDCG@K     : Normalised Discounted Cumulative Gain (binary relevance).
- MRR        : Mean Reciprocal Rank of the first relevant item in the ranking.

All metric functions accept pre-retrieved ranked lists (lists of item indices)
and a set of ground-truth positives per user so they are FAISS-agnostic and
easy to unit-test.

Usage:
    index, id_map = build_faiss_index(item_embeddings)
    top_k_ids = retrieve_top_k(index, id_map, user_embeddings, k=10)

    recall = recall_at_k(top_k_ids, ground_truth, k=10)
    ndcg   = ndcg_at_k(top_k_ids, ground_truth, k=10)
    mrr    = mrr_at_k(top_k_ids, ground_truth)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# FAISS index utilities
# ---------------------------------------------------------------------------

def build_faiss_index(
    embeddings: np.ndarray,
    item_ids: Optional[np.ndarray] = None,
) -> Tuple["faiss.Index", np.ndarray]:
    """
    Build a flat L2 (exact) FAISS index over the provided embeddings.

    Because embeddings from the towers are L2-normalised, L2 distance and
    cosine similarity are equivalent (argmin L2 == argmax cosine).

    Args:
        embeddings: (N, D) float32 array of item embeddings.
        item_ids:   (N,) int array mapping index positions → item ids.
                    If None, positions 0..N-1 are used as ids.

    Returns:
        (index, id_map) where id_map[i] is the item id for position i.
    """
    if not _FAISS_AVAILABLE:
        raise ImportError("faiss is required for index building. Install faiss-cpu.")

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if item_ids is None:
        item_ids = np.arange(len(embeddings), dtype=np.int64)

    D = embeddings.shape[1]
    index = faiss.IndexFlatL2(D)
    index.add(embeddings)
    return index, item_ids


def retrieve_top_k(
    index: "faiss.Index",
    id_map: np.ndarray,
    query_embeddings: np.ndarray,
    k: int = 10,
) -> List[List[int]]:
    """
    Retrieve top-K item ids for each query embedding.

    Args:
        index:            FAISS index built by build_faiss_index.
        id_map:           Mapping from index positions to item ids.
        query_embeddings: (Q, D) float32 array.
        k:                Number of neighbours to retrieve.

    Returns:
        List of length Q; each element is a list of k item ids ranked
        by ascending L2 distance (i.e. descending cosine similarity).
    """
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    _, indices = index.search(query_embeddings, k)  # (Q, k)
    return [id_map[row].tolist() for row in indices]


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved: List[List[int]],
    ground_truth: List[List[int]],
    k: int = 10,
) -> float:
    """
    Mean Recall@K over all queries.

    Recall@K for a single query = |retrieved[:k] ∩ relevant| / |relevant|.
    Queries with no relevant items are skipped.
    """
    scores = []
    for pred, gt in zip(retrieved, ground_truth):
        if not gt:
            continue
        top_k = set(pred[:k])
        relevant = set(gt)
        scores.append(len(top_k & relevant) / len(relevant))
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(
    retrieved: List[List[int]],
    ground_truth: List[List[int]],
    k: int = 10,
) -> float:
    """
    Mean NDCG@K over all queries (binary relevance).

    DCG@K  = Σ_{r=1}^{K} rel_r / log2(r + 1)
    IDCG@K = Σ_{r=1}^{min(|rel|,K)} 1 / log2(r + 1)
    NDCG@K = DCG@K / IDCG@K

    Queries with no relevant items are skipped.
    """
    scores = []
    for pred, gt in zip(retrieved, ground_truth):
        if not gt:
            continue
        relevant = set(gt)
        dcg = sum(
            1.0 / math.log2(r + 2)          # r is 0-indexed
            for r, item in enumerate(pred[:k])
            if item in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def mrr_at_k(
    retrieved: List[List[int]],
    ground_truth: List[List[int]],
    k: int = 10,
) -> float:
    """
    Mean Reciprocal Rank (within top-K) over all queries.

    MRR = (1/|Q|) Σ_q 1 / rank_q
    where rank_q is the position (1-indexed) of the first relevant item
    within the top-K list, or 0 if no relevant item appears.

    Queries with no relevant items are skipped.
    """
    scores = []
    for pred, gt in zip(retrieved, ground_truth):
        if not gt:
            continue
        relevant = set(gt)
        rr = 0.0
        for r, item in enumerate(pred[:k]):
            if item in relevant:
                rr = 1.0 / (r + 1)
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else 0.0


def compute_all_metrics(
    retrieved: List[List[int]],
    ground_truth: List[List[int]],
    k: int = 10,
) -> Dict[str, float]:
    """Convenience wrapper returning all three metrics in a single dict."""
    return {
        f"recall@{k}": recall_at_k(retrieved, ground_truth, k),
        f"ndcg@{k}":   ndcg_at_k(retrieved, ground_truth, k),
        f"mrr@{k}":    mrr_at_k(retrieved, ground_truth, k),
    }
