"""
Unit Tests for Playlist Ranker

Run with: pytest tests/
"""

import pytest
import torch
import numpy as np
from models.ranker import RankingModel, ConstraintLosses
from evaluation.metrics import RankingMetrics, PlaylistHealthMetrics


class TestRankingModel:
    """Tests for ranking model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        model = RankingModel(input_dim=32, hidden_dims=[128, 64, 32])
        assert model is not None
        
        # Check parameter count
        params = sum(p.numel() for p in model.parameters())
        assert params > 0
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = RankingModel(input_dim=32)
        
        batch_size = 16
        features = torch.randn(batch_size, 32)
        
        scores = model(features)
        
        assert scores.shape == (batch_size, 1)
    
    def test_model_gradient_flow(self):
        """Test gradients flow through model."""
        model = RankingModel(input_dim=32)

        features = torch.randn(8, 32)
        scores = model(features)
        
        loss = scores.sum()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None


def _group(n=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        'features': torch.randn(n, 32, generator=g),
        'labels': torch.tensor([1] * 4 + [0] * (n - 4)),
        'artist_idx': torch.randint(0, 6, (n,), generator=g),
        'in_recent_prefix': torch.rand(n, generator=g) < 0.4,
        'popularity': torch.rand(6, generator=g),
    }


class TestNormalization:
    """Feature standardisation stored in the model."""

    def test_normalization_is_saved_in_state_dict(self):
        model = RankingModel(input_dim=4, hidden_dims=[8])
        model.set_normalization(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([2.0, 0.0, 1.0, 1.0]))
        state = model.state_dict()
        assert torch.equal(state['feature_mean'], torch.tensor([1.0, 2.0, 3.0, 4.0]))
        # zero std is replaced by 1 so the feature passes through unscaled
        assert torch.equal(state['feature_std'], torch.tensor([2.0, 1.0, 1.0, 1.0]))

        clone = RankingModel(input_dim=4, hidden_dims=[8])
        clone.load_state_dict(state)
        clone.eval(); model.eval()
        x = torch.randn(3, 4)
        assert torch.allclose(clone(x), model(x))


class TestConstraintLosses:
    """Tests for ranking and constraint losses."""

    def test_pairwise_margin_loss(self):
        pos_scores = torch.tensor([0.8, 0.7, 0.9])
        neg_scores = torch.tensor([0.3, 0.4, 0.2])
        loss = ConstraintLosses.pairwise_margin_loss(pos_scores, neg_scores, margin=0.5)
        # margins: 0.5-0.5=0, 0.5-0.3=0.2, 0.5-0.7<0 -> mean(0, 0.2, 0) = 0.0667
        assert abs(loss.item() - 0.2 / 3) < 1e-6

    def test_fatigue_penalty_is_mass_on_recent_artists(self):
        p = torch.tensor([0.5, 0.3, 0.2])
        recent = torch.tensor([True, False, True])
        assert abs(ConstraintLosses.fatigue_penalty(p, recent).item() - 0.7) < 1e-6

    def test_diversity_penalty_is_negative_entropy(self):
        p = torch.full((4,), 0.25)
        # 4 distinct artists, uniform -> H = ln 4
        distinct = ConstraintLosses.diversity_penalty(p, torch.tensor([0, 1, 2, 3]))
        assert abs(distinct.item() + np.log(4)) < 1e-6
        # one artist -> H = 0
        single = ConstraintLosses.diversity_penalty(p, torch.tensor([7, 7, 7, 7]))
        assert abs(single.item()) < 1e-6

    def test_exposure_penalty_zero_when_matching_prior(self):
        p = torch.tensor([0.25, 0.25, 0.5])
        artist_idx = torch.tensor([0, 0, 1])
        popularity = torch.tensor([1.0, 1.0])  # prior over the group's artists: 0.5 / 0.5
        assert abs(ConstraintLosses.exposure_fairness_penalty(p, artist_idx, popularity).item()) < 1e-6

    @pytest.mark.parametrize("term", ["fatigue", "diversity", "exposure"])
    def test_each_constraint_has_gradient(self, term):
        """Every constraint term must be differentiable w.r.t. the scores."""
        g = _group()
        scores = torch.randn(20, requires_grad=True)
        p = ConstraintLosses.soft_attention(scores)
        if term == "fatigue":
            out = ConstraintLosses.fatigue_penalty(p, g['in_recent_prefix'])
        elif term == "diversity":
            out = ConstraintLosses.diversity_penalty(p, g['artist_idx'])
        else:
            out = ConstraintLosses.exposure_fairness_penalty(p, g['artist_idx'], g['popularity'])
        assert out.requires_grad
        out.backward()
        assert scores.grad is not None and scores.grad.abs().sum() > 0

    def test_constraint_weights_change_model_gradients(self):
        """With lambdas > 0 the parameter gradients must differ from the rank-only loss."""
        g = _group()
        model = RankingModel(input_dim=32)
        model.eval()

        def grads(lf, ld, le):
            model.zero_grad()
            scores = model(g['features']).squeeze(-1)
            loss, _ = ConstraintLosses.compute_group_loss(
                scores, g['labels'], g['artist_idx'], g['in_recent_prefix'], g['popularity'],
                lambda_fatigue=lf, lambda_diversity=ld, lambda_exposure=le,
            )
            loss.backward()
            return torch.cat([p.grad.flatten() for p in model.parameters()])

        base = grads(0.0, 0.0, 0.0)
        for lambdas in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
            assert (grads(*lambdas) - base).abs().max() > 1e-6, lambdas

    def test_fatigue_weight_lowers_mass_on_recent_artists_after_training(self):
        """Optimising with a fatigue weight must reduce the soft mass on recent-prefix artists."""
        g = _group()
        # Every positive repeats a recent artist, so relevance and fatigue pull in opposite
        # directions: the rank loss alone puts most of the mass on these tracks.
        g['in_recent_prefix'] = torch.tensor([True] * 4 + [False] * 16)

        def train(lf):
            torch.manual_seed(0)
            model = RankingModel(input_dim=32, dropout=0.0)
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(100):
                scores = model(g['features']).squeeze(-1)
                loss, _ = ConstraintLosses.compute_group_loss(
                    scores, g['labels'], g['artist_idx'], g['in_recent_prefix'], g['popularity'],
                    lambda_fatigue=lf, lambda_diversity=0.0, lambda_exposure=0.0,
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                p = ConstraintLosses.soft_attention(model(g['features']).squeeze(-1))
                return ConstraintLosses.fatigue_penalty(p, g['in_recent_prefix']).item()

        without, with_fatigue = train(0.0), train(5.0)
        assert without > 0.5           # rank loss alone: most mass on the (recent-artist) positives
        assert with_fatigue < 0.5 * without


class TestRankingMetrics:
    """Tests for ranking metrics."""
    
    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        relevance = [1, 1, 1, 0, 0]
        
        ndcg = RankingMetrics.ndcg_at_k(relevance, k=5)
        
        assert ndcg == 1.0
    
    def test_ndcg_zero_ranking(self):
        """Test NDCG with no relevant items."""
        relevance = [0, 0, 0, 0, 0]
        
        ndcg = RankingMetrics.ndcg_at_k(relevance, k=5)
        
        assert ndcg == 0.0
    
    def test_mrr(self):
        """Test MRR calculation."""
        relevance = [0, 0, 1, 0, 0]  # First relevant at position 3
        
        mrr = RankingMetrics.mrr(relevance)
        
        assert mrr == 1.0 / 3
    
    def test_recall_at_k(self):
        """Test Recall@K."""
        relevance = [1, 0, 1, 0, 1]  # 3 relevant total
        
        recall = RankingMetrics.recall_at_k(relevance, k=3, total_relevant=3)
        
        assert recall == 2.0 / 3


class TestPlaylistHealthMetrics:
    """Tests for playlist health metrics."""
    
    def test_artist_coverage(self):
        """Test artist coverage calculation."""
        artist_ids = ['a1', 'a2', 'a1', 'a3', 'a2']
        
        coverage = PlaylistHealthMetrics.artist_coverage(artist_ids)
        
        assert coverage == 3 / 5  # 3 unique out of 5
    
    def test_artist_entropy(self):
        """Test artist entropy calculation."""
        artist_ids = ['a1', 'a1', 'a1', 'a1']  # No diversity
        
        entropy = PlaylistHealthMetrics.artist_entropy(artist_ids)
        
        assert entropy == 0.0
        
        # High diversity
        artist_ids = ['a1', 'a2', 'a3', 'a4']
        entropy = PlaylistHealthMetrics.artist_entropy(artist_ids)
        assert entropy > 0
    
    def test_repeat_rate(self):
        """Test repeat rate calculation."""
        artist_ids = ['a1', 'a1', 'a2', 'a2', 'a3']
        
        rate = PlaylistHealthMetrics.repeat_rate(artist_ids)
        
        assert rate == 2 / 4  # 2 repeats out of 4 transitions
    
    def test_gini_coefficient(self):
        """Test Gini coefficient."""
        # Perfect equality
        frequencies = [1, 1, 1, 1]
        gini = PlaylistHealthMetrics.gini_coefficient(frequencies)
        assert gini < 0.1
        
        # Perfect inequality
        frequencies = [10, 0, 0, 0]
        gini = PlaylistHealthMetrics.gini_coefficient(frequencies)
        assert gini > 0.5


def test_end_to_end():
    """End-to-end integration test."""
    # Create model
    model = RankingModel(input_dim=32)
    
    # Create sample data
    pos_features = torch.randn(8, 32)
    neg_features = torch.randn(8, 32)
    
    # Forward pass
    pos_scores = model(pos_features)
    neg_scores = model(neg_features)
    
    # Compute loss for one group: 8 positives, 8 negatives
    scores = torch.cat([pos_scores, neg_scores]).squeeze(-1)
    labels = torch.tensor([1] * 8 + [0] * 8)
    loss, loss_dict = ConstraintLosses.compute_group_loss(scores, labels)
    
    # Check results
    assert loss.item() >= 0
    assert 'total' in loss_dict
    assert 'rank' in loss_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_metrics_return_python_floats():
    """Metrics are stored in checkpoints, which torch.load(weights_only=True) must read back."""
    values = [
        RankingMetrics.ndcg_at_k([1, 0, 1], 3),
        PlaylistHealthMetrics.artist_entropy(['a', 'b']),
        PlaylistHealthMetrics.gini_coefficient([1, 2, 3]),
        PlaylistHealthMetrics.popularity_bias(['a'], {'a': 0.5}),
    ]
    assert all(type(v) is float for v in values)
