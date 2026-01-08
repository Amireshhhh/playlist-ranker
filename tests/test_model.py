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


class TestConstraintLosses:
    """Tests for constraint losses."""
    
    def test_pairwise_margin_loss(self):
        """Test pairwise margin loss."""
        pos_scores = torch.tensor([0.8, 0.7, 0.9])
        neg_scores = torch.tensor([0.3, 0.4, 0.2])
        
        loss = ConstraintLosses.pairwise_margin_loss(pos_scores, neg_scores, margin=0.5)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_fatigue_penalty(self):
        """Test artist fatigue penalty."""
        artist_ids = torch.tensor([1, 2, 3, 1, 4, 2, 5])
        prefix_artist_ids = torch.tensor([1, 2])
        top_k_indices = torch.tensor([0, 1, 2])
        
        penalty = ConstraintLosses.fatigue_penalty(
            artist_ids, prefix_artist_ids, top_k_indices
        )
        
        assert isinstance(penalty, torch.Tensor)
        assert penalty.item() >= 0
    
    def test_diversity_penalty(self):
        """Test diversity penalty."""
        artist_ids = torch.tensor([1, 2, 3, 1, 4, 2, 5])
        top_k_indices = torch.tensor([0, 1, 2, 3, 4])
        
        penalty = ConstraintLosses.diversity_penalty(artist_ids, top_k_indices)
        
        assert isinstance(penalty, torch.Tensor)


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
    
    # Compute loss
    loss, loss_dict = ConstraintLosses.compute_total_loss(
        pos_scores.squeeze(),
        neg_scores.squeeze()
    )
    
    # Check results
    assert loss.item() >= 0
    assert 'total' in loss_dict
    assert 'rank' in loss_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
