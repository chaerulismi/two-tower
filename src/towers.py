"""
src/towers.py
-------------
Two-tower neural architecture for retrieval.

UserTower:  (user_idx, gender_idx, age_idx, occupation_idx) → query embedding
ItemTower:  (movie_idx, year_bucket, genres)                → candidate embedding

Both towers independently embed their inputs and project them to the same
output_dim so that dot-product similarity is well-defined.  Embeddings are
L2-normalised at the end so cosine similarity == dot product.

Usage (standalone):
    vocab = {"num_users": 6040, "num_movies": 3952,
             "num_genders": 2, "num_ages": 7,
             "num_occupations": 21, "num_year_buckets": 7, "num_genres": 18}

    model = TwoTowerModel(vocab)
    user_batch = {"user_idx": torch.zeros(4, dtype=torch.long),
                  "gender_idx": torch.zeros(4, dtype=torch.long),
                  "age_idx": torch.zeros(4, dtype=torch.long),
                  "occupation_idx": torch.zeros(4, dtype=torch.long)}
    item_batch = {"movie_idx": torch.zeros(4, dtype=torch.long),
                  "year_bucket": torch.zeros(4, dtype=torch.long),
                  "genres": torch.zeros(4, 18)}
    q, k = model(user_batch, item_batch)   # (4, output_dim), (4, output_dim)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Two-layer MLP with BatchNorm, ReLU, and Dropout."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# User tower
# ---------------------------------------------------------------------------

class UserTower(nn.Module):
    """
    Encodes a user feature dict into a fixed-size embedding.

    Input keys (all LongTensors of shape (B,)):
        user_idx, gender_idx, age_idx, occupation_idx

    Output: FloatTensor of shape (B, output_dim), L2-normalised.
    """

    def __init__(
        self,
        num_users: int,
        num_genders: int,
        num_ages: int,
        num_occupations: int,
        emb_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.gender_emb = nn.Embedding(num_genders, emb_dim)
        self.age_emb = nn.Embedding(num_ages, emb_dim)
        self.occupation_emb = nn.Embedding(num_occupations, emb_dim)

        # 4 embeddings concatenated
        self.mlp = _mlp(4 * emb_dim, hidden_dim, output_dim, dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_emb, self.gender_emb, self.age_emb, self.occupation_emb):
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, user: Dict[str, Tensor]) -> Tensor:
        x = torch.cat(
            [
                self.user_emb(user["user_idx"]),
                self.gender_emb(user["gender_idx"]),
                self.age_emb(user["age_idx"]),
                self.occupation_emb(user["occupation_idx"]),
            ],
            dim=-1,
        )  # (B, 4 * emb_dim)
        out = self.mlp(x)  # (B, output_dim)
        return F.normalize(out, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Item tower
# ---------------------------------------------------------------------------

class ItemTower(nn.Module):
    """
    Encodes an item feature dict into a fixed-size embedding.

    Input keys:
        movie_idx   : LongTensor  (B,)
        year_bucket : LongTensor  (B,)
        genres      : FloatTensor (B, num_genres)  multi-hot

    Output: FloatTensor of shape (B, output_dim), L2-normalised.
    """

    def __init__(
        self,
        num_movies: int,
        num_year_buckets: int,
        num_genres: int,
        emb_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.movie_emb = nn.Embedding(num_movies, emb_dim)
        self.year_emb = nn.Embedding(num_year_buckets, emb_dim)
        # Project multi-hot genre vector to emb_dim
        self.genre_proj = nn.Linear(num_genres, emb_dim, bias=False)

        # 3 emb_dim vectors concatenated
        self.mlp = _mlp(3 * emb_dim, hidden_dim, output_dim, dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.movie_emb, self.year_emb):
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.genre_proj.weight)

    def forward(self, item: Dict[str, Tensor]) -> Tensor:
        x = torch.cat(
            [
                self.movie_emb(item["movie_idx"]),
                self.year_emb(item["year_bucket"]),
                self.genre_proj(item["genres"]),
            ],
            dim=-1,
        )  # (B, 3 * emb_dim)
        out = self.mlp(x)  # (B, output_dim)
        return F.normalize(out, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Combined two-tower model
# ---------------------------------------------------------------------------

class TwoTowerModel(nn.Module):
    """
    Wraps UserTower and ItemTower.

    Forward returns (query_emb, key_emb) — both L2-normalised, shape (B, output_dim).
    At inference time call encode_user() or encode_item() independently.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        emb_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(
            num_users=vocab["num_users"],
            num_genders=vocab["num_genders"],
            num_ages=vocab["num_ages"],
            num_occupations=vocab["num_occupations"],
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.item_tower = ItemTower(
            num_movies=vocab["num_movies"],
            num_year_buckets=vocab["num_year_buckets"],
            num_genres=vocab["num_genres"],
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        user: Dict[str, Tensor],
        item: Dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Returns (query_emb, key_emb), both shape (B, output_dim)."""
        return self.user_tower(user), self.item_tower(item)

    def encode_user(self, user: Dict[str, Tensor]) -> Tensor:
        return self.user_tower(user)

    def encode_item(self, item: Dict[str, Tensor]) -> Tensor:
        return self.item_tower(item)
