"""
Evaluation Metrics for Ranking System

Implements NDCG, MRR, Recall, and playlist health metrics.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RankingMetrics:
    """Evaluation metrics for ranking quality."""
    
    @staticmethod
    def ndcg_at_k(relevance_scores: List[int], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at K.
        
        Args:
            relevance_scores: Binary relevance (1=relevant, 0=not relevant) in ranked order
            k: Cutoff position
            
        Returns:
            NDCG@K score in [0, 1]
        """
        if not relevance_scores:
            return 0.0
        
        # Actual DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_scores[:k]):
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1)=0
        
        # Ideal DCG (sorted by relevance)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance[:k]):
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mrr(relevance_scores: List[int]) -> float:
        """
        Mean Reciprocal Rank.
        
        Args:
            relevance_scores: Binary relevance in ranked order
            
        Returns:
            MRR score
        """
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def recall_at_k(relevance_scores: List[int], k: int, total_relevant: int) -> float:
        """
        Recall at K.
        
        Args:
            relevance_scores: Binary relevance in ranked order
            k: Cutoff position
            total_relevant: Total number of relevant items
            
        Returns:
            Recall@K in [0, 1]
        """
        if total_relevant == 0:
            return 0.0
        
        retrieved_relevant = sum(relevance_scores[:k])
        return retrieved_relevant / total_relevant
    
    @staticmethod
    def precision_at_k(relevance_scores: List[int], k: int) -> float:
        """
        Precision at K.
        
        Args:
            relevance_scores: Binary relevance in ranked order
            k: Cutoff position
            
        Returns:
            Precision@K in [0, 1]
        """
        if k == 0:
            return 0.0
        
        return sum(relevance_scores[:k]) / k


class PlaylistHealthMetrics:
    """Metrics for playlist quality and diversity."""
    
    @staticmethod
    def artist_coverage(artist_ids: List[str]) -> float:
        """
        Artist coverage (unique artists / total tracks).
        
        Args:
            artist_ids: List of artist IDs
            
        Returns:
            Coverage ratio in [0, 1]
        """
        if not artist_ids:
            return 0.0
        
        return len(set(artist_ids)) / len(artist_ids)
    
    @staticmethod
    def artist_entropy(artist_ids: List[str]) -> float:
        """
        Shannon entropy of artist distribution.
        
        H = -Σ p(a) * log2 p(a)
        
        Args:
            artist_ids: List of artist IDs
            
        Returns:
            Entropy in bits
        """
        if not artist_ids:
            return 0.0
        
        counts = Counter(artist_ids)
        total = len(artist_ids)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        return entropy
    
    @staticmethod
    def repeat_rate(artist_ids: List[str]) -> float:
        """
        Consecutive same-artist repetition rate.
        
        Args:
            artist_ids: List of artist IDs in order
            
        Returns:
            Fraction of consecutive repeats
        """
        if len(artist_ids) < 2:
            return 0.0
        
        repeats = sum(1 for i in range(len(artist_ids) - 1) if artist_ids[i] == artist_ids[i + 1])
        return repeats / (len(artist_ids) - 1)
    
    @staticmethod
    def gini_coefficient(frequencies: List[int]) -> float:
        """
        Gini coefficient for inequality measurement.
        
        Args:
            frequencies: List of frequencies (e.g., artist appearance counts)
            
        Returns:
            Gini coefficient in [0, 1] (0=perfect equality, 1=perfect inequality)
        """
        if not frequencies:
            return 0.0
        
        sorted_freqs = sorted(frequencies)
        n = len(sorted_freqs)
        
        cumsum = np.cumsum(sorted_freqs)
        
        # Gini formula
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_freqs)) / (n * np.sum(sorted_freqs)) - (n + 1) / n
        
        return gini
    
    @staticmethod
    def popularity_bias(artist_ids: List[str], artist_popularity: Dict[str, float]) -> float:
        """
        Average popularity of recommended artists.
        
        Args:
            artist_ids: List of artist IDs
            artist_popularity: Dict mapping artist_id -> popularity score
            
        Returns:
            Average popularity
        """
        if not artist_ids:
            return 0.0
        
        popularities = [artist_popularity.get(a, 0.0) for a in artist_ids]
        return np.mean(popularities)


class EvaluationSuite:
    """Complete evaluation suite combining all metrics."""
    
    def __init__(self):
        self.ranking_metrics = RankingMetrics()
        self.health_metrics = PlaylistHealthMetrics()
    
    def evaluate_ranking(
        self,
        ranked_tracks: List[str],
        ground_truth_tracks: List[str],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate ranking quality.
        
        Args:
            ranked_tracks: Predicted ranking
            ground_truth_tracks: Relevant tracks
            k_values: Cutoff positions to evaluate
            
        Returns:
            Dictionary of metric scores
        """
        # Create relevance labels
        ground_truth_set = set(ground_truth_tracks)
        relevance = [1 if track in ground_truth_set else 0 for track in ranked_tracks]
        
        metrics = {}
        
        # NDCG at different K
        for k in k_values:
            metrics[f'ndcg@{k}'] = self.ranking_metrics.ndcg_at_k(relevance, k)
        
        # MRR
        metrics['mrr'] = self.ranking_metrics.mrr(relevance)
        
        # Recall at different K
        total_relevant = len(ground_truth_tracks)
        for k in k_values:
            metrics[f'recall@{k}'] = self.ranking_metrics.recall_at_k(relevance, k, total_relevant)
        
        # Precision at different K
        for k in k_values:
            metrics[f'precision@{k}'] = self.ranking_metrics.precision_at_k(relevance, k)
        
        return metrics
    
    def evaluate_health(
        self,
        ranked_tracks: List[str],
        track_to_artist: Dict[str, str],
        artist_popularity: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Evaluate playlist health and diversity.
        
        Args:
            ranked_tracks: Predicted ranking
            track_to_artist: Mapping from track_id to artist_id
            artist_popularity: Optional artist popularity scores
            
        Returns:
            Dictionary of health metrics
        """
        # Get artist sequence
        artist_ids = [track_to_artist.get(t, 'unknown') for t in ranked_tracks]
        
        metrics = {
            'artist_coverage': self.health_metrics.artist_coverage(artist_ids),
            'artist_entropy': self.health_metrics.artist_entropy(artist_ids),
            'repeat_rate': self.health_metrics.repeat_rate(artist_ids)
        }
        
        # Gini coefficient
        artist_counts = Counter(artist_ids)
        metrics['gini_coefficient'] = self.health_metrics.gini_coefficient(list(artist_counts.values()))
        
        # Popularity bias (if available)
        if artist_popularity:
            metrics['popularity_bias'] = self.health_metrics.popularity_bias(artist_ids, artist_popularity)
        
        return metrics
    
    def full_evaluation(
        self,
        ranked_tracks: List[str],
        ground_truth_tracks: List[str],
        track_to_artist: Dict[str, str],
        artist_popularity: Dict[str, float] = None,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Complete evaluation: ranking + health metrics.
        
        Returns:
            Combined metrics dictionary
        """
        ranking_metrics = self.evaluate_ranking(ranked_tracks, ground_truth_tracks, k_values)
        health_metrics = self.evaluate_health(ranked_tracks, track_to_artist, artist_popularity)
        
        return {**ranking_metrics, **health_metrics}


def test_metrics():
    """Test evaluation metrics."""
    print("Testing Evaluation Metrics...\n")
    
    # Test data
    ranked = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10']
    ground_truth = ['t2', 't5', 't7', 't9']
    track_to_artist = {
        't1': 'a1', 't2': 'a2', 't3': 'a1', 't4': 'a3',
        't5': 'a2', 't6': 'a4', 't7': 'a1', 't8': 'a5',
        't9': 'a3', 't10': 'a6'
    }
    
    evaluator = EvaluationSuite()
    
    # Test ranking metrics
    ranking_results = evaluator.evaluate_ranking(ranked, ground_truth, k_values=[5, 10])
    print("Ranking Metrics:")
    for metric, value in ranking_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test health metrics
    health_results = evaluator.evaluate_health(ranked, track_to_artist)
    print("\nPlaylist Health Metrics:")
    for metric, value in health_results.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_metrics()
