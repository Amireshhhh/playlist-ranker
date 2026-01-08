"""
Feature Engineering for Ranking Model

Extracts features from prefix-continuation pairs for training and inference.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Build features for (prefix, candidate_track) pairs."""
    
    def __init__(self, data_dir: str):
        """
        Initialize feature builder.
        
        Args:
            data_dir: Directory containing processed splits and mappings
        """
        self.data_dir = Path(data_dir)
        
        # Load mappings
        logger.info("Loading track-artist mappings...")
        with open(self.data_dir / "track_to_artist.json", 'r') as f:
            self.track_to_artist = json.load(f)
        
        with open(self.data_dir / "artist_to_tracks.json", 'r') as f:
            self.artist_to_tracks = json.load(f)
        
        # Build co-occurrence statistics
        logger.info("Building co-occurrence statistics...")
        self.track_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.track_frequencies = Counter()
        self.artist_frequencies = Counter()
        
        self._build_statistics()
    
    def _build_statistics(self):
        """Build global co-occurrence and frequency statistics."""
        # Load training pairs to build statistics
        train_file = self.data_dir / "train_pairs.json"
        
        if not train_file.exists():
            logger.warning(f"Training file not found: {train_file}")
            return
        
        with open(train_file, 'r') as f:
            train_pairs = json.load(f)
        
        logger.info(f"Processing {len(train_pairs)} training pairs...")
        
        for pair in train_pairs:
            prefix_tracks = [t['track_uri'] for t in pair['prefix']]
            continuation_tracks = [t['track_uri'] for t in pair['continuation']]
            
            # Track frequencies
            for track in prefix_tracks + continuation_tracks:
                self.track_frequencies[track] += 1
                artist = self.track_to_artist.get(track)
                if artist:
                    self.artist_frequencies[artist] += 1
            
            # Co-occurrence: tracks in prefix with tracks in continuation
            for prefix_track in prefix_tracks:
                for cont_track in continuation_tracks:
                    self.track_cooccurrence[prefix_track][cont_track] += 1
        
        logger.info(f"Built statistics:")
        logger.info(f"  Unique tracks: {len(self.track_frequencies)}")
        logger.info(f"  Unique artists: {len(self.artist_frequencies)}")
        logger.info(f"  Co-occurrence pairs: {sum(len(v) for v in self.track_cooccurrence.values())}")
    
    def extract_features(
        self,
        prefix: List[Dict],
        candidate_track: str
    ) -> np.ndarray:
        """
        Extract features for (prefix, candidate_track) pair.
        
        Args:
            prefix: List of track dicts with 'track_uri' and 'artist_uri'
            candidate_track: Candidate track URI
            
        Returns:
            Feature vector (numpy array)
        """
        features = []
        
        prefix_track_uris = [t['track_uri'] for t in prefix]
        prefix_artist_uris = [t['artist_uri'] for t in prefix]
        candidate_artist = self.track_to_artist.get(candidate_track)
        
        # === Co-occurrence Features ===
        # Max co-occurrence with prefix tracks
        max_cooccur = 0
        total_cooccur = 0
        for prefix_track in prefix_track_uris:
            cooccur = self.track_cooccurrence.get(prefix_track, {}).get(candidate_track, 0)
            max_cooccur = max(max_cooccur, cooccur)
            total_cooccur += cooccur
        
        features.append(max_cooccur)
        features.append(total_cooccur)
        features.append(total_cooccur / len(prefix_track_uris) if prefix_track_uris else 0)
        
        # === Artist Repetition Features ===
        # Artist already in prefix?
        artist_in_prefix = int(candidate_artist in prefix_artist_uris)
        features.append(artist_in_prefix)
        
        # Artist repetition count
        artist_count = prefix_artist_uris.count(candidate_artist)
        features.append(artist_count)
        
        # Artist repetition ratio
        features.append(artist_count / len(prefix_artist_uris) if prefix_artist_uris else 0)
        
        # Recent artist repetition (last 5 tracks)
        recent_artists = prefix_artist_uris[-5:]
        recent_artist_count = recent_artists.count(candidate_artist)
        features.append(recent_artist_count)
        
        # === Position Features ===
        # Prefix length
        features.append(len(prefix_track_uris))
        
        # Normalized prefix length (assuming max ~100 tracks)
        features.append(len(prefix_track_uris) / 100.0)
        
        # === Popularity Features ===
        # Candidate track frequency (log scale)
        track_freq = self.track_frequencies.get(candidate_track, 1)
        features.append(np.log1p(track_freq))
        
        # Candidate artist frequency (log scale)
        artist_freq = self.artist_frequencies.get(candidate_artist, 1)
        features.append(np.log1p(artist_freq))
        
        # Popularity ratio (track vs artist)
        features.append(track_freq / max(artist_freq, 1))
        
        # === Prefix Diversity Features ===
        # Unique artists in prefix
        unique_artists = len(set(prefix_artist_uris))
        features.append(unique_artists)
        
        # Artist diversity ratio
        features.append(unique_artists / len(prefix_artist_uris) if prefix_artist_uris else 0)
        
        # === Artist Catalog Features ===
        # Number of tracks by candidate artist
        artist_catalog_size = len(self.artist_to_tracks.get(candidate_artist, []))
        features.append(np.log1p(artist_catalog_size))
        
        # === Recency Features ===
        # Position of last occurrence of this artist in prefix (-1 if not present)
        last_artist_pos = -1
        for i, artist in enumerate(prefix_artist_uris):
            if artist == candidate_artist:
                last_artist_pos = i
        
        features.append(last_artist_pos)
        features.append((last_artist_pos / len(prefix_artist_uris)) if last_artist_pos >= 0 else -1)
        
        # Distance from last occurrence
        distance_from_last = len(prefix_artist_uris) - last_artist_pos - 1 if last_artist_pos >= 0 else -1
        features.append(distance_from_last)
        
        # === Prefix Statistics ===
        # Max artist repetition in prefix
        artist_counts = Counter(prefix_artist_uris)
        max_artist_count = max(artist_counts.values()) if artist_counts else 0
        features.append(max_artist_count)
        
        # Average track popularity in prefix
        prefix_popularities = [self.track_frequencies.get(t, 1) for t in prefix_track_uris]
        avg_prefix_popularity = np.mean(prefix_popularities) if prefix_popularities else 0
        features.append(np.log1p(avg_prefix_popularity))
        
        # === Comparative Features ===
        # Candidate popularity vs prefix average
        popularity_diff = track_freq - avg_prefix_popularity
        features.append(np.log1p(abs(popularity_diff)) * np.sign(popularity_diff))
        
        # Artist frequency in prefix vs global
        prefix_artist_freq = artist_count
        global_artist_freq = artist_freq
        features.append(prefix_artist_freq / max(global_artist_freq, 1))
        
        # === Padding to fixed size ===
        # Ensure we have exactly 32 features
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32], dtype=np.float32)
    
    def build_feature_dataset(
        self,
        split_name: str,
        output_file: str,
        num_negatives: int = 5
    ):
        """
        Build feature dataset for a split.
        
        Args:
            split_name: 'train', 'val', or 'test'
            output_file: Output file path
            num_negatives: Number of negative samples per positive
        """
        logger.info(f"Building features for {split_name} split...")
        
        # Load pairs
        pairs_file = self.data_dir / f"{split_name}_pairs.json"
        with open(pairs_file, 'r') as f:
            pairs = json.load(f)
        
        logger.info(f"Loaded {len(pairs)} pairs")
        
        # Build feature dataset
        feature_data = []
        
        all_tracks = list(self.track_frequencies.keys())
        
        for i, pair in enumerate(pairs):
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i + 1}/{len(pairs)} pairs...")
            
            prefix = pair['prefix']
            continuation = pair['continuation']
            
            # Positive examples (tracks in continuation)
            for cont_track in continuation:
                features = self.extract_features(prefix, cont_track['track_uri'])
                
                feature_data.append({
                    'features': features.tolist(),
                    'label': 1,  # Positive
                    'track_uri': cont_track['track_uri'],
                    'prefix_length': len(prefix)
                })
            
            # Negative examples (random tracks NOT in continuation)
            continuation_track_uris = {t['track_uri'] for t in continuation}
            negatives = []
            
            while len(negatives) < num_negatives:
                neg_track = np.random.choice(all_tracks)
                if neg_track not in continuation_track_uris:
                    negatives.append(neg_track)
            
            for neg_track in negatives:
                features = self.extract_features(prefix, neg_track)
                
                feature_data.append({
                    'features': features.tolist(),
                    'label': 0,  # Negative
                    'track_uri': neg_track,
                    'prefix_length': len(prefix)
                })
        
        # Save feature dataset
        with open(output_file, 'w') as f:
            json.dump(feature_data, f)
        
        logger.info(f"Saved {len(feature_data)} feature vectors to {output_file}")
        logger.info(f"  Positives: {sum(1 for x in feature_data if x['label'] == 1)}")
        logger.info(f"  Negatives: {sum(1 for x in feature_data if x['label'] == 0)}")


def main():
    """Main entry point for feature building."""
    parser = argparse.ArgumentParser(description="Build features for ranking")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Processed data directory")
    parser.add_argument("--num-negatives", type=int, default=5, help="Negatives per positive")
    
    args = parser.parse_args()
    
    builder = FeatureBuilder(data_dir=args.data_dir)
    
    # Build features for all splits
    for split in ['train', 'val', 'test']:
        output_file = Path(args.data_dir) / f"{split}_features.json"
        builder.build_feature_dataset(
            split_name=split,
            output_file=str(output_file),
            num_negatives=args.num_negatives
        )
    
    logger.info("\n" + "="*60)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Output: {args.data_dir}")
    logger.info("\nNext step:")
    logger.info("  Train model: python training/train.py")
    logger.info("="*60)


if __name__ == "__main__":
    main()
