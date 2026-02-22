"""
scripts/build_index.py
----------------------
Build a FAISS ANN index over all item (movie) embeddings from a trained model.

The script:
1. Loads a trained TwoTowerModule checkpoint.
2. Runs inference through the ItemTower over the full movie catalogue.
3. Saves the resulting index + item-id mapping so retrieval can run
   without re-encoding items at query time.

Usage:
    python scripts/build_index.py \
        --checkpoint checkpoints/two_tower-epoch=09-val_loss=1.2345.ckpt \
        --processed_dir data/processed \
        --output_dir data/index

The output directory will contain:
    faiss.index   – the FAISS FlatL2 index
    item_ids.npy  – int64 array mapping index row → movie_idx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/build_index.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

log = logging.getLogger(__name__)


def build_index(
    checkpoint_path: Path,
    processed_dir: Path,
    output_dir: Path,
    batch_size: int = 2048,
    device: str = "cpu",
) -> None:
    import faiss  # noqa: F401 – will raise early if not installed

    from src.dataset import MovieLensDataModule
    from src.metrics import build_faiss_index
    from src.train import TwoTowerModule

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    log.info("Loading checkpoint: %s", checkpoint_path)
    dm = MovieLensDataModule(
        processed_dir=processed_dir,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False,
    )
    dm.setup("fit")  # need vocab

    module = TwoTowerModule.load_from_checkpoint(
        checkpoint_path,
        vocab=dm.vocab,
        map_location=device,
    )
    module.eval()
    module.to(device)

    # ── Encode all items ──────────────────────────────────────────────
    log.info("Encoding item catalogue ...")
    loader = dm.item_catalogue_dataloader(batch_size=batch_size)

    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            item = {k: v.to(device) for k, v in batch["item"].items()}
            emb = module.model.encode_item(item)  # (B, D)
            all_embeddings.append(emb.cpu().numpy())
            all_ids.append(batch["item"]["movie_idx"].numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    item_ids = np.concatenate(all_ids, axis=0).astype(np.int64)

    log.info("Encoded %d items, embedding dim=%d", len(embeddings), embeddings.shape[1])

    # ── Build & save FAISS index ──────────────────────────────────────
    index, id_map = build_faiss_index(embeddings, item_ids)

    index_path = output_dir / "faiss.index"
    ids_path = output_dir / "item_ids.npy"

    faiss.write_index(index, str(index_path))
    np.save(ids_path, id_map)

    log.info("FAISS index saved to %s", index_path)
    log.info("Item IDs saved to    %s", ids_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
    )
    parser = argparse.ArgumentParser(description="Build FAISS index from trained model")
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
        help="Directory with processed parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/index"),
        help="Directory to write the FAISS index and id map",
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    build_index(
        checkpoint_path=args.checkpoint,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
