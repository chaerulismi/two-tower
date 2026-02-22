"""
src/dataset.py
--------------
PyTorch Dataset and LightningDataModule for the two-tower retrieval model.

Each training sample is a positive (user, movie) pair.
Negatives are formed in-batch during the loss computation — no negative
sampling is required here.

Usage (standalone):
    dm = MovieLensDataModule(processed_dir="data/processed", batch_size=1024)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    print({k: v.shape for k, v in batch.items()})
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import lightning as L

log = logging.getLogger(__name__)

# Genre column prefix used during preprocessing
GENRE_PREFIX = "genre_"


# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------
# These must match the column names produced by src/preprocessing.py

USER_FEATURE_COLS = {
    "int": ["user_idx", "gender_idx", "age_idx", "occupation_idx"],
}

MOVIE_FEATURE_COLS = {
    "int": ["movie_idx", "year_bucket"],
    "float": [],          # genre columns are added dynamically at runtime
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MovieLensDataset(Dataset):
    """
    One sample = one positive (user, movie) interaction.

    Each __getitem__ returns a dict with two sub-dicts:
        {
          "user": { "user_idx": Tensor, "gender_idx": ..., ... },
          "item": { "movie_idx": Tensor, "year_bucket": ..., "genres": ... },
        }

    The genre multi-hot vector is stacked into a single float tensor of
    shape (num_genres,) under the key "genres".
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

        # Identify genre columns
        self.genre_cols = sorted(
            [c for c in df.columns if c.startswith(GENRE_PREFIX)]
        )

        # Pre-convert to numpy for fast indexing
        self._user_idx = df["user_idx"].to_numpy(dtype=np.int64)
        self._gender_idx = df["gender_idx"].to_numpy(dtype=np.int64)
        self._age_idx = df["age_idx"].to_numpy(dtype=np.int64)
        self._occupation_idx = df["occupation_idx"].to_numpy(dtype=np.int64)

        self._movie_idx = df["movie_idx"].to_numpy(dtype=np.int64)
        self._year_bucket = df["year_bucket"].to_numpy(dtype=np.int64)
        self._genres = df[self.genre_cols].to_numpy(dtype=np.float32)  # (N, G)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Tensor]]:
        return {
            "user": {
                "user_idx":       torch.tensor(self._user_idx[idx]),
                "gender_idx":     torch.tensor(self._gender_idx[idx]),
                "age_idx":        torch.tensor(self._age_idx[idx]),
                "occupation_idx": torch.tensor(self._occupation_idx[idx]),
            },
            "item": {
                "movie_idx":   torch.tensor(self._movie_idx[idx]),
                "year_bucket": torch.tensor(self._year_bucket[idx]),
                "genres":      torch.from_numpy(self._genres[idx].copy()),  # (G,)
            },
        }


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def collate_fn(samples: list) -> Dict[str, Dict[str, Tensor]]:
    """
    Merges a list of per-sample dicts into a batched dict.

    Output shapes (B = batch size, G = num genres):
        batch["user"]["user_idx"]       → (B,)
        batch["user"]["gender_idx"]     → (B,)
        batch["user"]["age_idx"]        → (B,)
        batch["user"]["occupation_idx"] → (B,)
        batch["item"]["movie_idx"]      → (B,)
        batch["item"]["year_bucket"]    → (B,)
        batch["item"]["genres"]         → (B, G)
    """
    user_batch: Dict[str, list] = {}
    item_batch: Dict[str, list] = {}

    for sample in samples:
        for k, v in sample["user"].items():
            user_batch.setdefault(k, []).append(v)
        for k, v in sample["item"].items():
            item_batch.setdefault(k, []).append(v)

    return {
        "user": {k: torch.stack(v) for k, v in user_batch.items()},
        "item": {k: torch.stack(v) for k, v in item_batch.items()},
    }


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------

class MovieLensDataModule(L.LightningDataModule):
    """
    LightningDataModule for MovieLens 1M two-tower training.

    Args:
        processed_dir: Path to the directory produced by src/preprocessing.py.
        batch_size:    Number of positive pairs per batch.
        num_workers:   DataLoader worker processes.
        pin_memory:    Pin CPU tensors to memory for faster GPU transfer.

    After calling .setup(), the following attributes are populated:
        .vocab        dict with cardinality of every embedding feature
        .genre_cols   list of genre column names (in order)
    """

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        batch_size: int = 1024,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: Optional[MovieLensDataset] = None
        self.val_dataset: Optional[MovieLensDataset] = None
        self.test_dataset: Optional[MovieLensDataset] = None

        self.vocab: Dict[str, int] = {}
        self.genre_cols: list[str] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called by Lightning before train/val/test loops.
        Loads parquet files and constructs Dataset objects.
        """
        vocab_path = self.processed_dir / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"vocab.json not found in {self.processed_dir}. "
                "Run `python -m src.preprocessing` first."
            )
        with open(vocab_path) as f:
            self.vocab = json.load(f)

        if stage in ("fit", None):
            train_df = pd.read_parquet(self.processed_dir / "train.parquet")
            val_df = pd.read_parquet(self.processed_dir / "val.parquet")
            self.train_dataset = MovieLensDataset(train_df)
            self.val_dataset = MovieLensDataset(val_df)
            self.genre_cols = self.train_dataset.genre_cols
            log.info(
                "Train: %d samples | Val: %d samples",
                len(self.train_dataset),
                len(self.val_dataset),
            )

        if stage in ("test", None):
            test_df = pd.read_parquet(self.processed_dir / "test.parquet")
            self.test_dataset = MovieLensDataset(test_df)
            if not self.genre_cols:
                self.genre_cols = self.test_dataset.genre_cols
            log.info("Test: %d samples", len(self.test_dataset))

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,           # shuffle within training epoch
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,         # ensures full in-batch negative matrices
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    # Convenience: full item catalogue loader
    # ------------------------------------------------------------------

    def item_catalogue_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """
        Returns a DataLoader over ALL unique movies (for building the FAISS
        index at evaluation time).  Each batch contains item features only.
        """
        movies_df = pd.read_parquet(self.processed_dir / "movies.parquet")

        # Build a dummy pairs df that only has item columns needed
        # (user columns are filled with zeros — they are unused here)
        dummy_df = movies_df.copy()
        for col in ["user_idx", "gender_idx", "age_idx", "occupation_idx"]:
            dummy_df[col] = 0

        dataset = MovieLensDataset(dummy_df)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )