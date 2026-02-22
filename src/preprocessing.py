"""
src/preprocessing.py
--------------------
Loads raw MovieLens 1M files, engineers features, and saves processed
artefacts (encoded DataFrames + encoder mappings) to data/processed/.

Usage:
    python -m src.preprocessing --raw_dir data/raw --out_dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MovieLens 1M uses "::" as a separator and has no header row.
RATINGS_COLS = ["user_id", "movie_id", "rating", "timestamp"]
USERS_COLS = ["user_id", "gender", "age", "occupation", "zip_code"]
MOVIES_COLS = ["movie_id", "title", "genres"]

# We binarise ratings: positive = rating >= threshold
POSITIVE_RATING_THRESHOLD = 4

# Age brackets as defined in the MovieLens README
AGE_BUCKET_MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

# All 18 genres present in MovieLens 1M
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ratings(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=RATINGS_COLS,
        encoding="latin-1",
    )
    log.info("Ratings loaded: %d rows", len(df))
    return df


def load_users(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        names=USERS_COLS,
        encoding="latin-1",
    )
    log.info("Users loaded: %d rows", len(df))
    return df


def load_movies(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        names=MOVIES_COLS,
        encoding="latin-1",
    )
    log.info("Movies loaded: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_user_features(users: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode all user features.
    Returns the enriched DataFrame and a dict of {feature: encoder}.
    """
    encoders: Dict[str, LabelEncoder] = {}
    df = users.copy()

    # user_id — shift to 0-indexed contiguous int
    le = LabelEncoder()
    df["user_idx"] = le.fit_transform(df["user_id"])
    encoders["user_id"] = le

    # gender: M/F → 0/1
    df["gender_idx"] = (df["gender"] == "M").astype(int)

    # age: already bucketed in the dataset (1,18,25,35,45,50,56) → 0-6
    df["age_idx"] = df["age"].map(AGE_BUCKET_MAP).astype(int)

    # occupation: 0-20, already integers — just label-encode for safety
    le_occ = LabelEncoder()
    df["occupation_idx"] = le_occ.fit_transform(df["occupation"])
    encoders["occupation"] = le_occ

    # drop columns we don't need downstream
    df = df.drop(columns=["zip_code", "gender", "age", "occupation"])

    log.info(
        "User features engineered. Unique users: %d", df["user_id"].nunique()
    )
    return df, encoders


def engineer_movie_features(movies: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode all movie features.
    Returns the enriched DataFrame and a dict of {feature: encoder}.
    """
    encoders: Dict[str, LabelEncoder] = {}
    df = movies.copy()

    # movie_id → 0-indexed contiguous int
    le = LabelEncoder()
    df["movie_idx"] = le.fit_transform(df["movie_id"])
    encoders["movie_id"] = le

    # Extract year from title e.g. "Toy Story (1995)" → 1995
    df["year"] = (
        df["title"].str.extract(r"\((\d{4})\)").astype(float).squeeze()
    )
    df["year"] = df["year"].fillna(df["year"].median())

    # Bucket year into decades: <1950, 1950s, 1960s, ..., 2000s
    bins = [0, 1949, 1959, 1969, 1979, 1989, 1999, 3000]
    labels = list(range(7))
    df["year_bucket"] = pd.cut(df["year"], bins=bins, labels=labels).astype(int)

    # Multi-hot genre encoding
    for genre in ALL_GENRES:
        safe_name = genre.replace("'", "").replace("-", "_").lower()
        df[f"genre_{safe_name}"] = df["genres"].str.contains(
            genre, regex=False
        ).astype(np.float32)

    df = df.drop(columns=["title", "genres", "year"])

    log.info(
        "Movie features engineered. Unique movies: %d", df["movie_id"].nunique()
    )
    return df, encoders


# ---------------------------------------------------------------------------
# Interaction pair construction
# ---------------------------------------------------------------------------

def build_interaction_pairs(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    threshold: int = POSITIVE_RATING_THRESHOLD,
) -> pd.DataFrame:
    """
    Filter to positive interactions and join user/movie feature indices.
    Returns a DataFrame with one row per (user, movie) positive pair,
    plus all encoded feature columns needed by the dataset.
    """
    # Keep only positives
    pos = ratings[ratings["rating"] >= threshold].copy()
    log.info(
        "Positive pairs (rating >= %d): %d / %d (%.1f%%)",
        threshold,
        len(pos),
        len(ratings),
        100.0 * len(pos) / len(ratings),
    )

    # Merge user and movie feature columns
    pos = pos.merge(users, on="user_id", how="inner")
    pos = pos.merge(movies, on="movie_id", how="inner")
    pos = pos.sort_values("timestamp").reset_index(drop=True)

    return pos


# ---------------------------------------------------------------------------
# Train / val / test split (temporal)
# ---------------------------------------------------------------------------

def temporal_split(
    pairs: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: train → val → test.
    Ensures no data leakage across the time boundary.
    """
    n = len(pairs)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))

    train = pairs.iloc[:train_end].reset_index(drop=True)
    val = pairs.iloc[train_end:val_end].reset_index(drop=True)
    test = pairs.iloc[val_end:].reset_index(drop=True)

    log.info(
        "Split sizes — train: %d | val: %d | test: %d", len(train), len(val), len(test)
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Cardinality / vocabulary sizes (needed to init embedding layers)
# ---------------------------------------------------------------------------

def compute_vocab_sizes(users: pd.DataFrame, movies: pd.DataFrame) -> Dict[str, int]:
    return {
        "num_users": int(users["user_idx"].max() + 1),
        "num_movies": int(movies["movie_idx"].max() + 1),
        "num_genders": 2,
        "num_ages": len(AGE_BUCKET_MAP),
        "num_occupations": int(users["occupation_idx"].max() + 1),
        "num_year_buckets": 7,
        "num_genres": len(ALL_GENRES),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    ratings = load_ratings(raw_dir)
    users_raw = load_users(raw_dir)
    movies_raw = load_movies(raw_dir)

    # 2. Feature engineering
    users, user_encoders = engineer_user_features(users_raw)
    movies, movie_encoders = engineer_movie_features(movies_raw)

    # 3. Build interaction pairs
    pairs = build_interaction_pairs(ratings, users, movies)

    # 4. Temporal split
    train, val, test = temporal_split(pairs)

    # 5. Vocabulary sizes
    vocab = compute_vocab_sizes(users, movies)
    log.info("Vocab sizes: %s", vocab)

    # 6. Save
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    users.to_parquet(out_dir / "users.parquet", index=False)
    movies.to_parquet(out_dir / "movies.parquet", index=False)

    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    encoders = {**user_encoders, **movie_encoders}
    with open(out_dir / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    log.info("✅ Preprocessing complete. Artefacts saved to %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MovieLens 1M")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    run(args.raw_dir, args.out_dir)