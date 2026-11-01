"""
Ranking Model with Multi-Objective Constraints

PyTorch implementation of the playlist ranking model and its training losses.

Training works on *groups*: one group is the candidate list for one playlist
prefix (its true continuation tracks plus sampled negatives). Within a group:

  * the ranking loss is a pairwise margin loss over every (positive, negative) pair;
  * the constraint losses are computed on p = softmax(z / temperature), where z
    are the group's standardised scores: the model's soft ranking distribution
    over the group's candidates. p is a smooth
    surrogate for "which tracks end up at the top": raising a candidate's score
    raises its share of p. Because p is a differentiable function of the scores,
    every constraint term sends gradient back into the model.

Hard top-K selection (argsort, unique, counting) has zero gradient almost
everywhere, so penalties built on it cannot change the model's weights.
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingModel(nn.Module):
    """Neural ranking model for playlist continuation."""

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.3
    ):
        """
        Initialize ranking model.

        Args:
            input_dim: Feature vector dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)

        # Feature standardisation, stored with the weights so a checkpoint carries
        # the exact normalisation it was trained with. Identity until set.
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Dropout after the first two hidden layers
            if i < 2:
                layers.append(nn.Dropout(dropout if i == 0 else dropout * 0.7))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @torch.no_grad()
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Set per-feature mean/std; features with zero variance are left unscaled."""
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(torch.where(std > 0, std, torch.ones_like(std)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: (batch_size, input_dim) raw feature tensor

        Returns:
            scores: (batch_size, 1) relevance scores
        """
        return self.network((features - self.feature_mean) / self.feature_std)


class ConstraintLosses:
    """Ranking and constraint losses. All constraint terms are differentiable."""

    @staticmethod
    def pairwise_margin_loss(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        margin: float = 0.5
    ) -> torch.Tensor:
        """
        Pairwise margin ranking loss, averaged over all given pairs.

        L = mean(max(0, margin - (score_pos - score_neg)))

        `pos_scores` and `neg_scores` must broadcast against each other: equal
        shapes for matched pairs, or (P, 1) and (1, N) for all P x N pairs.
        """
        return torch.mean(F.relu(margin - (pos_scores - neg_scores)))

    @staticmethod
    def soft_attention(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Soft ranking distribution of a group: p = softmax(z / temperature), where
        z = (scores - mean) / std are the group's standardised scores.

        Standardising makes p depend only on the *relative* order and spacing of the
        scores, not on their scale. Without it the margin loss can grow the scores
        without bound, softmax saturates (p -> one-hot) and its gradient p(1 - p)
        vanishes, which silently switches the constraint terms off.
        """
        if scores.numel() > 1:
            z = (scores - scores.mean()) / (scores.std() + 1e-6)
        else:
            z = scores - scores.detach()
        return torch.softmax(z / temperature, dim=0)

    @staticmethod
    def fatigue_penalty(p: torch.Tensor, in_recent_prefix: torch.Tensor) -> torch.Tensor:
        """
        Artist fatigue: probability mass placed on candidates whose artist already
        appears in the last 5 tracks of the prefix.

        L_fatigue = sum_i p_i * 1[artist_i in recent prefix]      (in [0, 1])
        """
        return torch.sum(p * in_recent_prefix.to(p.dtype))

    @staticmethod
    def artist_distribution(p: torch.Tensor, artist_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft artist distribution q_a = sum_{i: artist_i = a} p_i over the group's artists.

        Returns:
            unique_artists: (A,) artist indices present in the group
            q: (A,) probability mass per artist (sums to 1)
        """
        unique_artists, inverse = torch.unique(artist_idx, return_inverse=True)
        q = torch.zeros(len(unique_artists), dtype=p.dtype, device=p.device).index_add(0, inverse, p)
        return unique_artists, q

    @staticmethod
    def diversity_penalty(p: torch.Tensor, artist_idx: torch.Tensor) -> torch.Tensor:
        """
        Diversity: negative entropy of the soft artist distribution.

        L_diversity = -H(q) = sum_a q_a * log q_a      (minimising it maximises entropy)
        """
        _, q = ConstraintLosses.artist_distribution(p, artist_idx)
        return torch.sum(q * torch.log(q + 1e-10))

    @staticmethod
    def exposure_fairness_penalty(
        p: torch.Tensor,
        artist_idx: torch.Tensor,
        artist_popularity: torch.Tensor
    ) -> torch.Tensor:
        """
        Exposure fairness: squared distance between the soft artist distribution
        and the popularity prior restricted to the group's artists.

        L_exposure = sum_a (q_a - pi_a)^2,  pi_a = popularity_a / sum_{b in group} popularity_b

        Args:
            p: (n,) soft ranking distribution
            artist_idx: (n,) artist index of each candidate
            artist_popularity: (num_artists,) popularity prior, indexed by artist index
        """
        unique_artists, q = ConstraintLosses.artist_distribution(p, artist_idx)
        prior = artist_popularity[unique_artists].to(p.dtype)
        total = prior.sum()
        if total <= 0:
            return torch.zeros((), dtype=p.dtype, device=p.device)
        return torch.sum((q - prior / total) ** 2)

    @staticmethod
    def compute_group_loss(
        scores: torch.Tensor,
        labels: torch.Tensor,
        artist_idx: Optional[torch.Tensor] = None,
        in_recent_prefix: Optional[torch.Tensor] = None,
        artist_popularity: Optional[torch.Tensor] = None,
        lambda_fatigue: float = 0.1,
        lambda_diversity: float = 0.05,
        lambda_exposure: float = 0.03,
        margin: float = 0.5,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Multi-objective loss for one group.

        L_total = L_rank + l1 * L_fatigue + l2 * L_diversity + l3 * L_exposure

        Args:
            scores: (n,) model scores for the group's candidates
            labels: (n,) 1 for continuation tracks, 0 for negatives
            artist_idx: (n,) artist index per candidate (needed for diversity/exposure)
            in_recent_prefix: (n,) bool, artist appears in the last 5 prefix tracks
            artist_popularity: (num_artists,) popularity prior (needed for exposure)
            lambda_*: constraint weights (0 disables a term)
            margin: margin for the pairwise loss
            temperature: softmax temperature for the soft ranking distribution

        Returns:
            total_loss: scalar tensor
            loss_dict: float value of each component
        """
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) and len(neg):
            l_rank = ConstraintLosses.pairwise_margin_loss(pos.unsqueeze(1), neg.unsqueeze(0), margin)
        else:
            l_rank = scores.sum() * 0.0  # no pairs in this group; keeps the graph connected

        losses = {'rank': l_rank.item(), 'fatigue': 0.0, 'diversity': 0.0, 'exposure': 0.0}
        total_loss = l_rank

        needs_p = (
            (lambda_fatigue > 0 and in_recent_prefix is not None)
            or (lambda_diversity > 0 and artist_idx is not None)
            or (lambda_exposure > 0 and artist_idx is not None and artist_popularity is not None)
        )
        if needs_p:
            p = ConstraintLosses.soft_attention(scores, temperature)

            if lambda_fatigue > 0 and in_recent_prefix is not None:
                l_fatigue = ConstraintLosses.fatigue_penalty(p, in_recent_prefix)
                losses['fatigue'] = l_fatigue.item()
                total_loss = total_loss + lambda_fatigue * l_fatigue

            if lambda_diversity > 0 and artist_idx is not None:
                l_diversity = ConstraintLosses.diversity_penalty(p, artist_idx)
                losses['diversity'] = l_diversity.item()
                total_loss = total_loss + lambda_diversity * l_diversity

            if lambda_exposure > 0 and artist_idx is not None and artist_popularity is not None:
                l_exposure = ConstraintLosses.exposure_fairness_penalty(p, artist_idx, artist_popularity)
                losses['exposure'] = l_exposure.item()
                total_loss = total_loss + lambda_exposure * l_exposure

        losses['total'] = total_loss.item()
        return total_loss, losses


def test_model():
    """Smoke test of the model and losses."""
    print("Testing RankingModel...")

    model = RankingModel(input_dim=32, hidden_dims=[128, 64, 32])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    n = 20
    scores = model(torch.randn(n, 32)).squeeze(-1)
    labels = torch.tensor([1] * 4 + [0] * (n - 4))
    artist_idx = torch.randint(0, 10, (n,))
    in_recent = torch.rand(n) < 0.3
    popularity = torch.rand(10)

    loss, loss_dict = ConstraintLosses.compute_group_loss(
        scores, labels, artist_idx, in_recent, popularity
    )
    loss.backward()
    print(f"Multi-objective loss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    print("\n✓ Smoke test passed")


if __name__ == "__main__":
    test_model()
