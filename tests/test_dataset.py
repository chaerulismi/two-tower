"""
tests/test_dataset.py
---------------------
Unit tests for preprocessing and dataset components.
Run with: pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# ── helpers ──────────────────────────────────────────────────────────────────

def make_mock_raw_dir(tmp_path: Path) -> Path:
    """Write minimal MovieLens-style .dat files to a temp directory."""
    raw = tmp_path / "raw"
    raw.mkdir()

    # users.dat  (6 users)
    users = [
        "1::F::1::10::48067",
        "2::M::56::16::70072",
        "3::M::25::15::55117",
        "4::M::45::7::02460",
        "5::M::25::20::55455",
        "6::F::50::9::55117",
    ]
    (raw / "users.dat").write_text("\n".join(users))

    # movies.dat  (4 movies)
    movies = [
        "1::Toy Story (1995)::Animation|Children's|Comedy",
        "2::Jumanji (1995)::Adventure|Children's|Fantasy",
        "3::GoodFellas (1990)::Crime|Drama",
        "4::The Matrix (1999)::Action|Sci-Fi|Thriller",
    ]
    (raw / "movies.dat").write_text("\n".join(movies))

    # ratings.dat  (20 ratings spanning two timestamps)
    ratings = []
    ts_base = 978_300_000
    uid = 1
    for movie_id in [1, 2, 3, 4]:
        for user_id in range(1, 6):
            rating = 4 if (user_id + movie_id) % 2 == 0 else 2
            ratings.append(
                f"{user_id}::{movie_id}::{rating}::{ts_base + user_id * 100 + movie_id}"
            )
    (raw / "ratings.dat").write_text("\n".join(ratings))

    return raw


# ── preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_load_ratings(self, tmp_path):
        from src.preprocessing import load_ratings
        raw = make_mock_raw_dir(tmp_path)
        df = load_ratings(raw)
        assert set(df.columns) == {"user_id", "movie_id", "rating", "timestamp"}
        assert len(df) == 20

    def test_load_users(self, tmp_path):
        from src.preprocessing import load_users
        raw = make_mock_raw_dir(tmp_path)
        df = load_users(raw)
        assert "user_id" in df.columns
        assert len(df) == 6

    def test_load_movies(self, tmp_path):
        from src.preprocessing import load_movies
        raw = make_mock_raw_dir(tmp_path)
        df = load_movies(raw)
        assert "movie_id" in df.columns
        assert len(df) == 4

    def test_engineer_user_features(self, tmp_path):
        from src.preprocessing import load_users, engineer_user_features
        raw = make_mock_raw_dir(tmp_path)
        users_raw = load_users(raw)
        users, encoders = engineer_user_features(users_raw)

        assert "user_idx" in users.columns
        assert "gender_idx" in users.columns
        assert "age_idx" in users.columns
        assert "occupation_idx" in users.columns
        # zip_code should be dropped
        assert "zip_code" not in users.columns
        # indices must be contiguous non-negative ints
        assert users["user_idx"].min() == 0
        assert users["user_idx"].max() == len(users) - 1

    def test_engineer_movie_features(self, tmp_path):
        from src.preprocessing import load_movies, engineer_movie_features, ALL_GENRES
        raw = make_mock_raw_dir(tmp_path)
        movies_raw = load_movies(raw)
        movies, encoders = engineer_movie_features(movies_raw)

        assert "movie_idx" in movies.columns
        assert "year_bucket" in movies.columns
        # all genre columns present
        for genre in ALL_GENRES:
            safe = genre.replace("'", "").replace("-", "_").lower()
            assert f"genre_{safe}" in movies.columns, f"Missing genre col: genre_{safe}"
        # genre values are 0 or 1
        genre_cols = [c for c in movies.columns if c.startswith("genre_")]
        assert movies[genre_cols].isin([0.0, 1.0]).all().all()

    def test_build_interaction_pairs(self, tmp_path):
        from src.preprocessing import (
            load_ratings, load_users, load_movies,
            engineer_user_features, engineer_movie_features,
            build_interaction_pairs,
        )
        raw = make_mock_raw_dir(tmp_path)
        ratings = load_ratings(raw)
        users, _ = engineer_user_features(load_users(raw))
        movies, _ = engineer_movie_features(load_movies(raw))
        pairs = build_interaction_pairs(ratings, users, movies, threshold=4)

        # Every pair should have rating >= 4
        assert (pairs["rating"] >= 4).all()
        # User and movie index columns must be present
        assert "user_idx" in pairs.columns
        assert "movie_idx" in pairs.columns

    def test_temporal_split_ordering(self, tmp_path):
        from src.preprocessing import (
            load_ratings, load_users, load_movies,
            engineer_user_features, engineer_movie_features,
            build_interaction_pairs, temporal_split,
        )
        raw = make_mock_raw_dir(tmp_path)
        ratings = load_ratings(raw)
        users, _ = engineer_user_features(load_users(raw))
        movies, _ = engineer_movie_features(load_movies(raw))
        pairs = build_interaction_pairs(ratings, users, movies, threshold=4)

        if len(pairs) < 3:
            pytest.skip("Not enough positive pairs for split test with mock data")

        train, val, test = temporal_split(pairs, val_frac=0.1, test_frac=0.1)
        total = len(train) + len(val) + len(test)
        assert total == len(pairs)
        # Temporal ordering: last train ts <= first val ts (approx.)
        if len(val) > 0 and len(train) > 0:
            assert train["timestamp"].max() <= val["timestamp"].max()

    def test_full_pipeline_saves_artefacts(self, tmp_path):
        from src.preprocessing import run
        raw = make_mock_raw_dir(tmp_path)
        out = tmp_path / "processed"
        run(raw, out)

        for fname in ["train.parquet", "val.parquet", "test.parquet",
                      "users.parquet", "movies.parquet",
                      "vocab.json", "encoders.pkl"]:
            assert (out / fname).exists(), f"Missing {fname}"

        with open(out / "vocab.json") as f:
            vocab = json.load(f)
        for key in ["num_users", "num_movies", "num_genders",
                    "num_ages", "num_occupations", "num_year_buckets", "num_genres"]:
            assert key in vocab, f"Missing vocab key: {key}"
            assert vocab[key] > 0


# ── dataset tests ─────────────────────────────────────────────────────────────

class TestMovieLensDataset:

    @pytest.fixture()
    def processed_dir(self, tmp_path):
        from src.preprocessing import run
        raw = make_mock_raw_dir(tmp_path)
        out = tmp_path / "processed"
        run(raw, out)
        return out

    def test_dataset_len(self, processed_dir):
        from src.dataset import MovieLensDataset
        df = pd.read_parquet(processed_dir / "train.parquet")
        ds = MovieLensDataset(df)
        assert len(ds) == len(df)

    def test_sample_keys(self, processed_dir):
        from src.dataset import MovieLensDataset
        df = pd.read_parquet(processed_dir / "train.parquet")
        if len(df) == 0:
            pytest.skip("No training samples in mock data")
        ds = MovieLensDataset(df)
        sample = ds[0]

        assert set(sample.keys()) == {"user", "item"}
        assert set(sample["user"].keys()) == {
            "user_idx", "gender_idx", "age_idx", "occupation_idx"
        }
        assert set(sample["item"].keys()) == {"movie_idx", "year_bucket", "genres"}

    def test_sample_types_and_shapes(self, processed_dir):
        from src.dataset import MovieLensDataset
        df = pd.read_parquet(processed_dir / "train.parquet")
        if len(df) == 0:
            pytest.skip("No training samples in mock data")
        ds = MovieLensDataset(df)
        sample = ds[0]

        for k, v in sample["user"].items():
            assert isinstance(v, torch.Tensor), f"user[{k}] not a tensor"
            assert v.shape == torch.Size([]), f"user[{k}] should be scalar"

        assert sample["item"]["genres"].dtype == torch.float32
        assert sample["item"]["genres"].ndim == 1


# ── DataModule tests ──────────────────────────────────────────────────────────

class TestMovieLensDataModule:

    @pytest.fixture()
    def processed_dir(self, tmp_path):
        from src.preprocessing import run
        raw = make_mock_raw_dir(tmp_path)
        out = tmp_path / "processed"
        run(raw, out)
        return out

    def test_setup_populates_vocab(self, processed_dir):
        from src.dataset import MovieLensDataModule
        dm = MovieLensDataModule(processed_dir=processed_dir, batch_size=4, num_workers=0)
        dm.setup("fit")
        assert dm.vocab
        assert "num_users" in dm.vocab

    def test_train_dataloader_batch_shape(self, processed_dir):
        from src.dataset import MovieLensDataModule
        dm = MovieLensDataModule(processed_dir=processed_dir, batch_size=4, num_workers=0)
        dm.setup("fit")

        if dm.train_dataset is None or len(dm.train_dataset) == 0:
            pytest.skip("No training data")

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        B = min(4, len(dm.train_dataset))
        assert batch["user"]["user_idx"].shape[0] <= B
        assert batch["item"]["genres"].ndim == 2          # (B, num_genres)
        assert batch["item"]["genres"].dtype == torch.float32

    def test_val_dataloader_no_shuffle(self, processed_dir):
        from src.dataset import MovieLensDataModule
        dm = MovieLensDataModule(processed_dir=processed_dir, batch_size=4, num_workers=0)
        dm.setup("fit")

        if dm.val_dataset is None or len(dm.val_dataset) == 0:
            pytest.skip("No val data")

        loader = dm.val_dataloader()
        # Two passes should return same order
        b1 = next(iter(loader))["item"]["movie_idx"]
        b2 = next(iter(loader))["item"]["movie_idx"]
        assert torch.equal(b1, b2)

    def test_genre_cols_populated(self, processed_dir):
        from src.dataset import MovieLensDataModule
        dm = MovieLensDataModule(processed_dir=processed_dir, batch_size=4, num_workers=0)
        dm.setup("fit")
        assert len(dm.genre_cols) > 0
        assert all(c.startswith("genre_") for c in dm.genre_cols)