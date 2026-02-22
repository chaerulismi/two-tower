"""
src/loss.py
-----------
In-batch softmax loss (InfoNCE / NT-Xent variant) for two-tower retrieval.

Given a batch of B (user, item) positive pairs, every item in the batch
serves as a negative for every other user.  This avoids the need for
explicit negative sampling and scales well with batch size.

Loss formulation
----------------
    S[i, j] = dot(query[i], key[j]) / temperature

The diagonal S[i, i] is the positive score.  For each user i, we compute
a softmax over the full row and use cross-entropy with target class i.

    L = - (1/B) Σ_i log(softmax(S[i, :])[i])
      = - (1/B) Σ_i [ S[i,i] - log Σ_j exp(S[i,j]) ]

Because embeddings are L2-normalised (as produced by towers.py), the dot
product equals cosine similarity, so temperature directly controls the
sharpness of the distribution.

Usage:
    from src.loss import in_batch_softmax_loss

    q = model.encode_user(user_batch)   # (B, D)
    k = model.encode_item(item_batch)   # (B, D)
    loss = in_batch_softmax_loss(q, k, temperature=0.07)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def in_batch_softmax_loss(
    query: Tensor,
    key: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """
    Symmetric in-batch softmax (InfoNCE) loss.

    Args:
        query:       (B, D) L2-normalised user embeddings.
        key:         (B, D) L2-normalised item embeddings.
        temperature: Softmax temperature.  Lower = sharper distribution.

    Returns:
        Scalar loss tensor.
    """
    if query.shape != key.shape:
        raise ValueError(
            f"query and key must have the same shape, got {query.shape} vs {key.shape}"
        )

    B = query.size(0)
    # Similarity matrix: (B, B), entry [i,j] = cosine_sim(user_i, item_j)
    logits = torch.matmul(query, key.T) / temperature  # (B, B)

    # Targets: positive pair is always on the diagonal
    labels = torch.arange(B, device=query.device)

    # Symmetric loss: average user→item and item→user cross-entropy
    loss_q = F.cross_entropy(logits, labels)
    loss_k = F.cross_entropy(logits.T, labels)
    return (loss_q + loss_k) / 2
