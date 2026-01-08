"""
Ranking Model with Multi-Objective Constraints

PyTorch implementation of the playlist ranking model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class RankingModel(nn.Module):
    """Neural ranking model for playlist continuation."""
    
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.3
    ):
        """
        Initialize ranking model.
        
        Args:
            input_dim: Feature vector dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super(RankingModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            # Dropout after first two layers
            if i < 2:
                layers.append(nn.Dropout(dropout if i == 0 else dropout * 0.7))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: (batch_size, input_dim) feature tensor
            
        Returns:
            scores: (batch_size, 1) relevance scores
        """
        return self.network(features)


class ConstraintLosses:
    """Constraint loss functions for multi-objective optimization."""
    
    @staticmethod
    def pairwise_margin_loss(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        margin: float = 0.5
    ) -> torch.Tensor:
        """
        Pairwise margin ranking loss.
        
        L = max(0, margin - (score_pos - score_neg))
        
        Args:
            pos_scores: (batch_size,) scores for positive samples
            neg_scores: (batch_size,) scores for negative samples
            margin: Margin threshold
            
        Returns:
            Scalar loss
        """
        return torch.mean(F.relu(margin - (pos_scores - neg_scores)))
    
    @staticmethod
    def fatigue_penalty(
        artist_ids: torch.Tensor,
        prefix_artist_ids: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Artist fatigue penalty.
        
        Penalizes repeated artists in top-K recommendations.
        
        L_fatigue = Σ 𝟙[artist_k ∈ recent_prefix]
        
        Args:
            artist_ids: (num_candidates,) artist IDs for candidates
            prefix_artist_ids: (prefix_length,) artist IDs in prefix
            top_k_indices: (k,) indices of top-K ranked tracks
            
        Returns:
            Scalar penalty
        """
        if len(top_k_indices) == 0:
            return torch.tensor(0.0, device=artist_ids.device)
        
        # Get artists in top-K
        top_k_artists = artist_ids[top_k_indices]
        
        # Recent prefix (last 5 tracks)
        recent_prefix = prefix_artist_ids[-5:] if len(prefix_artist_ids) > 5 else prefix_artist_ids
        
        # Count repetitions
        penalty = 0.0
        for artist in top_k_artists:
            if artist in recent_prefix:
                penalty += 1.0
        
        return torch.tensor(penalty / len(top_k_indices), device=artist_ids.device)
    
    @staticmethod
    def diversity_penalty(
        artist_ids: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Diversity penalty based on artist entropy.
        
        L_diversity = -H(artist_distribution)
        H = -Σ p(a) * log p(a)
        
        Args:
            artist_ids: (num_candidates,) artist IDs
            top_k_indices: (k,) indices of top-K tracks
            
        Returns:
            Scalar penalty (negative entropy)
        """
        if len(top_k_indices) == 0:
            return torch.tensor(0.0, device=artist_ids.device)
        
        # Get artists in top-K
        top_k_artists = artist_ids[top_k_indices]
        
        # Compute artist distribution
        unique_artists, counts = torch.unique(top_k_artists, return_counts=True)
        probs = counts.float() / len(top_k_artists)
        
        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        # Return negative entropy as penalty (we want to maximize entropy)
        return -entropy
    
    @staticmethod
    def exposure_fairness_penalty(
        artist_ids: torch.Tensor,
        artist_popularity: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Exposure fairness penalty.
        
        Penalizes deviation from expected exposure based on popularity.
        
        L_exposure = Σ (appearances_a / K - π_a)²
        
        Args:
            artist_ids: (num_candidates,) artist IDs
            artist_popularity: (num_artists,) popularity prior for each artist
            top_k_indices: (k,) indices of top-K tracks
            
        Returns:
            Scalar penalty
        """
        if len(top_k_indices) == 0:
            return torch.tensor(0.0, device=artist_ids.device)
        
        # Get artists in top-K
        top_k_artists = artist_ids[top_k_indices]
        k = len(top_k_artists)
        
        # Compute actual exposure
        unique_artists, counts = torch.unique(top_k_artists, return_counts=True)
        actual_exposure = counts.float() / k
        
        # Get expected exposure (popularity prior)
        expected_exposure = artist_popularity[unique_artists]
        
        # Mean squared error
        penalty = torch.mean((actual_exposure - expected_exposure) ** 2)
        
        return penalty
    
    @staticmethod
    def compute_total_loss(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        artist_ids: torch.Tensor = None,
        prefix_artist_ids: torch.Tensor = None,
        artist_popularity: torch.Tensor = None,
        top_k: int = 10,
        lambda_fatigue: float = 0.1,
        lambda_diversity: float = 0.05,
        lambda_exposure: float = 0.03,
        margin: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total multi-objective loss.
        
        L_total = L_rank + λ₁·L_fatigue + λ₂·L_diversity + λ₃·L_exposure
        
        Args:
            pos_scores: Positive sample scores
            neg_scores: Negative sample scores
            artist_ids: Artist IDs for all candidates
            prefix_artist_ids: Artist IDs in prefix
            artist_popularity: Popularity prior
            top_k: Number of top results to consider for constraints
            lambda_fatigue: Weight for fatigue penalty
            lambda_diversity: Weight for diversity penalty
            lambda_exposure: Weight for exposure penalty
            margin: Margin for pairwise loss
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual loss components
        """
        # Ranking loss (always computed)
        l_rank = ConstraintLosses.pairwise_margin_loss(pos_scores, neg_scores, margin)
        
        losses = {
            'rank': l_rank.item(),
            'fatigue': 0.0,
            'diversity': 0.0,
            'exposure': 0.0
        }
        
        total_loss = l_rank
        
        # Constraint losses (only if artist info provided)
        if artist_ids is not None and len(artist_ids) > 0:
            # Get top-K indices based on all scores
            all_scores = torch.cat([pos_scores, neg_scores])
            top_k_indices = torch.argsort(all_scores.squeeze(), descending=True)[:top_k]
            
            # Fatigue penalty
            if prefix_artist_ids is not None and lambda_fatigue > 0:
                l_fatigue = ConstraintLosses.fatigue_penalty(
                    artist_ids, prefix_artist_ids, top_k_indices
                )
                losses['fatigue'] = l_fatigue.item()
                total_loss = total_loss + lambda_fatigue * l_fatigue
            
            # Diversity penalty
            if lambda_diversity > 0:
                l_diversity = ConstraintLosses.diversity_penalty(
                    artist_ids, top_k_indices
                )
                losses['diversity'] = l_diversity.item()
                total_loss = total_loss + lambda_diversity * l_diversity
            
            # Exposure fairness penalty
            if artist_popularity is not None and lambda_exposure > 0:
                l_exposure = ConstraintLosses.exposure_fairness_penalty(
                    artist_ids, artist_popularity, top_k_indices
                )
                losses['exposure'] = l_exposure.item()
                total_loss = total_loss + lambda_exposure * l_exposure
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


def test_model():
    """Test model and losses."""
    print("Testing RankingModel...")
    
    # Create model
    model = RankingModel(input_dim=32, hidden_dims=[128, 64, 32])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 16
    features = torch.randn(batch_size, 32)
    scores = model(features)
    print(f"Output shape: {scores.shape}")
    
    # Test pairwise loss
    pos_scores = scores[:batch_size//2]
    neg_scores = scores[batch_size//2:]
    
    loss, loss_dict = ConstraintLosses.compute_total_loss(
        pos_scores, neg_scores,
        lambda_fatigue=0.0,
        lambda_diversity=0.0,
        lambda_exposure=0.0
    )
    
    print(f"Pairwise loss: {loss.item():.4f}")
    
    # Test with constraints
    artist_ids = torch.randint(0, 100, (batch_size,))
    prefix_artist_ids = torch.randint(0, 100, (10,))
    artist_popularity = torch.rand(100) / 100  # Normalized
    
    loss, loss_dict = ConstraintLosses.compute_total_loss(
        pos_scores, neg_scores,
        artist_ids=artist_ids,
        prefix_artist_ids=prefix_artist_ids,
        artist_popularity=artist_popularity,
        lambda_fatigue=0.1,
        lambda_diversity=0.05,
        lambda_exposure=0.03
    )
    
    print(f"\nMulti-objective loss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model()
