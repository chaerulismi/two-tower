"""
tests/test_towers.py
--------------------
Unit tests for towers.py, loss.py, and metrics.py.
Run with: pytest tests/test_towers.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB = {
    "num_users": 10,
    "num_movies": 8,
    "num_genders": 2,
    "num_ages": 7,
    "num_occupations": 5,
    "num_year_buckets": 7,
    "num_genres": 18,
}

BATCH_SIZE = 4
OUTPUT_DIM = 16
EMB_DIM = 8
HIDDEN_DIM = 32


def make_user_batch(B: int = BATCH_SIZE) -> dict:
    return {
        "user_idx":       torch.randint(0, VOCAB["num_users"], (B,)),
        "gender_idx":     torch.randint(0, VOCAB["num_genders"], (B,)),
        "age_idx":        torch.randint(0, VOCAB["num_ages"], (B,)),
        "occupation_idx": torch.randint(0, VOCAB["num_occupations"], (B,)),
    }


def make_item_batch(B: int = BATCH_SIZE) -> dict:
    return {
        "movie_idx":   torch.randint(0, VOCAB["num_movies"], (B,)),
        "year_bucket": torch.randint(0, VOCAB["num_year_buckets"], (B,)),
        "genres":      torch.rand(B, VOCAB["num_genres"]),
    }


# ---------------------------------------------------------------------------
# UserTower tests
# ---------------------------------------------------------------------------

class TestUserTower:
    @pytest.fixture()
    def tower(self):
        from src.towers import UserTower
        return UserTower(
            num_users=VOCAB["num_users"],
            num_genders=VOCAB["num_genders"],
            num_ages=VOCAB["num_ages"],
            num_occupations=VOCAB["num_occupations"],
            emb_dim=EMB_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
        )

    def test_output_shape(self, tower):
        out = tower(make_user_batch())
        assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_output_l2_normalised(self, tower):
        out = tower(make_user_batch())
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_output_dtype(self, tower):
        out = tower(make_user_batch())
        assert out.dtype == torch.float32

    def test_single_sample(self, tower):
        """Tower must work with batch size of 1 (BatchNorm edge case)."""
        tower.eval()
        with torch.no_grad():
            out = tower(make_user_batch(B=1))
        assert out.shape == (1, OUTPUT_DIM)

    def test_gradients_flow(self, tower):
        out = tower(make_user_batch())
        loss = out.sum()
        loss.backward()
        for name, param in tower.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# ItemTower tests
# ---------------------------------------------------------------------------

class TestItemTower:
    @pytest.fixture()
    def tower(self):
        from src.towers import ItemTower
        return ItemTower(
            num_movies=VOCAB["num_movies"],
            num_year_buckets=VOCAB["num_year_buckets"],
            num_genres=VOCAB["num_genres"],
            emb_dim=EMB_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
        )

    def test_output_shape(self, tower):
        out = tower(make_item_batch())
        assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_output_l2_normalised(self, tower):
        out = tower(make_item_batch())
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_output_dtype(self, tower):
        out = tower(make_item_batch())
        assert out.dtype == torch.float32

    def test_zero_genre_vector(self, tower):
        """Item tower should handle all-zero genre vectors gracefully."""
        batch = make_item_batch()
        batch["genres"] = torch.zeros(BATCH_SIZE, VOCAB["num_genres"])
        out = tower(batch)
        assert out.shape == (BATCH_SIZE, OUTPUT_DIM)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_gradients_flow(self, tower):
        out = tower(make_item_batch())
        loss = out.sum()
        loss.backward()
        for name, param in tower.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# TwoTowerModel tests
# ---------------------------------------------------------------------------

class TestTwoTowerModel:
    @pytest.fixture()
    def model(self):
        from src.towers import TwoTowerModel
        return TwoTowerModel(
            vocab=VOCAB,
            emb_dim=EMB_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
        )

    def test_forward_shapes(self, model):
        q, k = model(make_user_batch(), make_item_batch())
        assert q.shape == (BATCH_SIZE, OUTPUT_DIM)
        assert k.shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_encode_user_shape(self, model):
        q = model.encode_user(make_user_batch())
        assert q.shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_encode_item_shape(self, model):
        k = model.encode_item(make_item_batch())
        assert k.shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_towers_independent(self, model):
        """User and item towers must not share parameters."""
        user_params = set(id(p) for p in model.user_tower.parameters())
        item_params = set(id(p) for p in model.item_tower.parameters())
        assert len(user_params & item_params) == 0, "Towers share parameters"

    def test_eval_mode_no_dropout(self, model):
        """In eval mode two identical inputs should give identical outputs."""
        model.eval()
        b = make_item_batch()
        with torch.no_grad():
            k1 = model.encode_item(b)
            k2 = model.encode_item(b)
        assert torch.allclose(k1, k2)


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestInBatchSoftmaxLoss:
    def _random_embs(self, B: int = 8, D: int = 16) -> tuple:
        import torch.nn.functional as F
        q = torch.randn(B, D)
        k = torch.randn(B, D)
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1)

    def test_scalar_output(self):
        from src.loss import in_batch_softmax_loss
        q, k = self._random_embs()
        loss = in_batch_softmax_loss(q, k)
        assert loss.ndim == 0

    def test_non_negative(self):
        from src.loss import in_batch_softmax_loss
        q, k = self._random_embs()
        loss = in_batch_softmax_loss(q, k)
        assert loss.item() >= 0.0

    def test_perfect_alignment_lower_loss(self):
        """Aligned pairs (q==k) should yield lower loss than random embeddings."""
        import torch.nn.functional as F
        from src.loss import in_batch_softmax_loss
        B, D = 16, 32
        q = F.normalize(torch.randn(B, D), dim=-1)
        loss_aligned = in_batch_softmax_loss(q, q.clone())
        loss_random = in_batch_softmax_loss(q, F.normalize(torch.randn(B, D), dim=-1))
        assert loss_aligned.item() < loss_random.item()

    def test_shape_mismatch_raises(self):
        from src.loss import in_batch_softmax_loss
        q = torch.randn(4, 16)
        k = torch.randn(5, 16)
        with pytest.raises(ValueError):
            in_batch_softmax_loss(q, k)

    def test_gradient_flows(self):
        from src.loss import in_batch_softmax_loss
        q, k = self._random_embs()
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        loss = in_batch_softmax_loss(q, k)
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_recall_perfect(self):
        from src.metrics import recall_at_k
        retrieved = [[0, 1, 2], [3, 4, 5]]
        gt        = [[0], [3]]
        assert recall_at_k(retrieved, gt, k=3) == pytest.approx(1.0)

    def test_recall_zero(self):
        from src.metrics import recall_at_k
        retrieved = [[0, 1, 2], [3, 4, 5]]
        gt        = [[9], [8]]
        assert recall_at_k(retrieved, gt, k=3) == pytest.approx(0.0)

    def test_recall_partial(self):
        from src.metrics import recall_at_k
        retrieved = [[0, 1], [3, 4]]
        gt        = [[0, 1], [3, 9]]   # user 0: 2/2=1.0, user 1: 1/2=0.5 → avg=0.75
        score = recall_at_k(retrieved, gt, k=2)
        assert score == pytest.approx(0.75)

    def test_recall_empty_gt_skipped(self):
        from src.metrics import recall_at_k
        retrieved = [[0, 1]]
        gt        = [[]]
        assert recall_at_k(retrieved, gt, k=2) == pytest.approx(0.0)

    def test_ndcg_perfect(self):
        from src.metrics import ndcg_at_k
        retrieved = [[0, 1, 2]]
        gt        = [[0]]
        assert ndcg_at_k(retrieved, gt, k=3) == pytest.approx(1.0)

    def test_ndcg_rank_matters(self):
        from src.metrics import ndcg_at_k
        score_rank1 = ndcg_at_k([[0, 1]], [[0]], k=2)
        score_rank2 = ndcg_at_k([[1, 0]], [[0]], k=2)
        assert score_rank1 > score_rank2

    def test_mrr_first_hit(self):
        from src.metrics import mrr_at_k
        retrieved = [[5, 0, 1]]
        gt        = [[0]]
        # First hit at position 2 (1-indexed) -> MRR = 0.5
        assert mrr_at_k(retrieved, gt, k=3) == pytest.approx(0.5)

    def test_mrr_no_hit(self):
        from src.metrics import mrr_at_k
        retrieved = [[1, 2, 3]]
        gt        = [[9]]
        assert mrr_at_k(retrieved, gt, k=3) == pytest.approx(0.0)

    def test_compute_all_metrics_keys(self):
        from src.metrics import compute_all_metrics
        result = compute_all_metrics([[0, 1, 2]], [[0]], k=3)
        assert set(result.keys()) == {"recall@3", "ndcg@3", "mrr@3"}
        for v in result.values():
            assert 0.0 <= v <= 1.0
