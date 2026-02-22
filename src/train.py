"""
src/train.py
------------
PyTorch Lightning training loop for the two-tower retrieval model.

Entry point:
    python -m src.train                        # uses configs/default.yaml
    python -m src.train --config my_cfg.yaml   # custom config

The module also exposes TwoTowerModule so it can be imported in notebooks
or evaluation scripts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.dataset import MovieLensDataModule
from src.loss import in_batch_softmax_loss
from src.towers import TwoTowerModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class TwoTowerModule(L.LightningModule):
    """
    Wraps TwoTowerModel + in-batch loss for Lightning-managed training.

    Hyperparameters (passed as a flat dict or keyword args):
        emb_dim       int   Embedding dim per feature (default 32)
        hidden_dim    int   MLP hidden size (default 128)
        output_dim    int   Final embedding / similarity dim (default 64)
        dropout       float Dropout probability (default 0.1)
        temperature   float InfoNCE temperature (default 0.07)
        lr            float AdamW learning rate (default 1e-3)
        weight_decay  float AdamW weight decay (default 1e-4)
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        emb_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 0.07,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["vocab"])
        self.vocab = vocab

        self.model = TwoTowerModel(
            vocab=vocab,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Forward / encoding helpers
    # ------------------------------------------------------------------

    def forward(self, user: Dict[str, torch.Tensor], item: Dict[str, torch.Tensor]):
        return self.model(user, item)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        q, k = self.model(batch["user"], batch["item"])
        loss = in_batch_softmax_loss(q, k, temperature=self.temperature)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        q, k = self.model(batch["user"], batch["item"])
        loss = in_batch_softmax_loss(q, k, temperature=self.temperature)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any]) -> TwoTowerModule:
    """
    Run full training from a config dict.

    Returns the trained LightningModule.
    """
    L.seed_everything(cfg.get("seed", 42), workers=True)

    # ── DataModule ────────────────────────────────────────────────────
    data_cfg = cfg.get("data", {})
    dm = MovieLensDataModule(
        processed_dir=data_cfg.get("processed_dir", "data/processed"),
        batch_size=data_cfg.get("batch_size", 1024),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
    dm.setup("fit")

    # ── Model ─────────────────────────────────────────────────────────
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    module = TwoTowerModule(
        vocab=dm.vocab,
        emb_dim=model_cfg.get("emb_dim", 32),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        output_dim=model_cfg.get("output_dim", 64),
        dropout=model_cfg.get("dropout", 0.1),
        temperature=train_cfg.get("temperature", 0.07),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="two_tower-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=train_cfg.get("patience", 5),
            mode="min",
        ),
    ]

    # ── Logger ────────────────────────────────────────────────────────
    logger = CSVLogger("logs", name="two_tower")

    # ── Trainer ───────────────────────────────────────────────────────
    trainer_cfg = train_cfg.get("trainer", {})
    trainer = L.Trainer(
        max_epochs=train_cfg.get("max_epochs", 10),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        callbacks=callbacks,
        logger=logger,
        deterministic=False,
    )

    trainer.fit(module, datamodule=dm)
    log.info("Training complete. Best checkpoint: %s", callbacks[0].best_model_path)
    return module


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    """Load YAML config if provided, otherwise use defaults."""
    if path is None:
        return {}
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise ImportError("PyYAML is required for config loading: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
    )
    parser = argparse.ArgumentParser(description="Train two-tower retrieval model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    cfg = _load_config(args.config)
    train(cfg)
