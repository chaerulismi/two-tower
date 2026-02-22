"""
scripts/evaluate.py
-------------------
Full retrieval evaluation for the two-tower model.

Pipeline
--------
1. Load the trained model from a Lightning checkpoint.
2. Encode ALL items in the catalogue → build an in-memory FAISS index.
3. For every user in the test split:
   a. Encode the user's features.
   b. Retrieve top-K items from the FAISS index.
   c. Compare against the user's ground-truth positives in the test set.
4. Report Recall@K, NDCG@K, and MRR@K (K = 10, 50 by default).

Usage
-----
    python scripts/evaluate.py \
        --checkpoint checkpoints/two_tower-epoch=04-val/loss=3.21.ckpt \
        --processed_dir data/processed \
        --k 10 50

Outputs a table like:
    K=10  recall=0.1234  ndcg=0.0987  mrr=0.0876
    K=50  recall=0.2345  ndcg=0.1234  mrr=0.0876
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as `python scripts/evaluate.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


def run_evaluation(
    checkpoint_path: Path,
    processed_dir: Path,
    ks: list[int],
    batch_size: int = 2048,
    device: str = "cpu",
) -> dict[int, dict[str, float]]:
    from src.dataset import MovieLensDataModule
    from src.metrics import build_faiss_index, retrieve_top_k, compute_all_metrics
    from src.train import TwoTowerModule

    # ── 1. Load model ─────────────────────────────────────────────────
    log.info("Loading checkpoint: %s", checkpoint_path)
    dm = MovieLensDataModule(
        processed_dir=processed_dir,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup(None)  # loads train + val + test

    module = TwoTowerModule.load_from_checkpoint(
        checkpoint_path,
        vocab=dm.vocab,
        map_location=device,
    )
    module.eval().to(device)

    # ── 2. Encode full item catalogue ─────────────────────────────────
    log.info("Encoding item catalogue ...")
    item_embeddings = []
    item_ids_list = []

    with torch.no_grad():
        for batch in dm.item_catalogue_dataloader(batch_size=batch_size):
            item = {k: v.to(device) for k, v in batch["item"].items()}
            emb = module.model.encode_item(item)
            item_embeddings.append(emb.cpu().numpy())
            item_ids_list.append(batch["item"]["movie_idx"].numpy())

    item_embeddings = np.concatenate(item_embeddings, axis=0).astype(np.float32)
    item_ids_arr = np.concatenate(item_ids_list, axis=0).astype(np.int64)

    log.info("Catalogue: %d items, dim=%d", len(item_embeddings), item_embeddings.shape[1])

    index, id_map = build_faiss_index(item_embeddings, item_ids_arr)

    # ── 3. Build per-user ground truth from test set ──────────────────
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    user_gt: dict[int, list[int]] = defaultdict(list)
    for _, row in test_df.iterrows():
        user_gt[int(row["user_idx"])].append(int(row["movie_idx"]))

    unique_users = sorted(user_gt.keys())
    log.info("Evaluating on %d users from test set", len(unique_users))

    # ── 4. Encode users & retrieve ────────────────────────────────────
    # Load users table to get feature columns
    users_df = pd.read_parquet(processed_dir / "users.parquet")
    users_df = users_df.set_index("user_idx")

    # Build query tensors for all unique users
    user_emb_rows = []

    for chunk_start in range(0, len(unique_users), batch_size):
        chunk = unique_users[chunk_start : chunk_start + batch_size]
        rows = users_df.loc[chunk]
        user_batch = {
            "user_idx":       torch.tensor(rows.index.values, dtype=torch.long).to(device),
            "gender_idx":     torch.tensor(rows["gender_idx"].values, dtype=torch.long).to(device),
            "age_idx":        torch.tensor(rows["age_idx"].values, dtype=torch.long).to(device),
            "occupation_idx": torch.tensor(rows["occupation_idx"].values, dtype=torch.long).to(device),
        }
        with torch.no_grad():
            emb = module.model.encode_user(user_batch)
        user_emb_rows.append(emb.cpu().numpy())

    user_embeddings = np.concatenate(user_emb_rows, axis=0).astype(np.float32)

    # ── 5. Compute metrics at each K ──────────────────────────────────
    max_k = max(ks)
    retrieved = retrieve_top_k(index, id_map, user_embeddings, k=max_k)
    ground_truth = [user_gt[u] for u in unique_users]

    results: dict[int, dict[str, float]] = {}
    for k in ks:
        metrics = compute_all_metrics(retrieved, ground_truth, k=k)
        results[k] = metrics
        log.info(
            "K=%-4d  recall=%.4f  ndcg=%.4f  mrr=%.4f",
            k,
            metrics[f"recall@{k}"],
            metrics[f"ndcg@{k}"],
            metrics[f"mrr@{k}"],
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
    )
    parser = argparse.ArgumentParser(description="Evaluate two-tower retrieval model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[10, 50],
        help="Values of K for Recall@K / NDCG@K / MRR@K",
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save results as JSON",
    )
    args = parser.parse_args()

    results = run_evaluation(
        checkpoint_path=args.checkpoint,
        processed_dir=args.processed_dir,
        ks=args.k,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("\n── Results ──────────────────────────────")
    for k, metrics in results.items():
        print(
            f"K={k:<4d}  "
            f"Recall={metrics[f'recall@{k}']:.4f}  "
            f"NDCG={metrics[f'ndcg@{k}']:.4f}  "
            f"MRR={metrics[f'mrr@{k}']:.4f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {args.output}")
