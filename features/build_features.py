"""
Feature Engineering for Ranking Model

Extracts features from prefix-continuation pairs for training and inference.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Build features for (prefix, candidate_track) pairs."""
    
    def __init__(self, data_dir: str, seed: int = 42):
        """
        Initialize feature builder.
        
        Args:
            data_dir: Directory containing processed splits and mappings
            seed: Seed for negative sampling
        """
        self.data_dir = Path(data_dir)
        self.rng = np.random.default_rng(seed)
        # Training pairs grouped by playlist, kept so train features can be computed
        # with that playlist's own contribution removed (see build_feature_dataset).
        self.train_pairs_by_playlist: Dict[int, List[Dict]] = defaultdict(list)
        
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
            self.train_pairs_by_playlist[pair['playlist_id']].append(pair)
        self._apply_pairs(train_pairs, +1)
        
        logger.info("Built statistics:")
        logger.info(f"  Unique tracks: {len(self.track_frequencies)}")
        logger.info(f"  Unique artists: {len(self.artist_frequencies)}")
        logger.info(f"  Co-occurrence pairs: {sum(len(v) for v in self.track_cooccurrence.values())}")
    
    def _apply_pairs(self, pairs: List[Dict], sign: int):
        """Add (sign=+1) or remove (sign=-1) the statistics contributed by `pairs`."""
        for pair in pairs:
            prefix_tracks = [t['track_uri'] for t in pair['prefix']]
            continuation_tracks = [t['track_uri'] for t in pair['continuation']]

            # Track and artist frequencies
            for track in prefix_tracks + continuation_tracks:
                self.track_frequencies[track] += sign
                artist = self.track_to_artist.get(track)
                if artist:
                    self.artist_frequencies[artist] += sign

            # Co-occurrence: tracks in prefix with tracks in continuation
            for prefix_track in prefix_tracks:
                row = self.track_cooccurrence[prefix_track]
                for cont_track in continuation_tracks:
                    row[cont_track] += sign

    @staticmethod
    def _freq(counter: Counter, key) -> int:
        """Frequency with a floor of 1, so unseen and left-out items are treated alike."""
        return max(counter.get(key, 0), 1)

    def artist_popularity(self) -> Dict[str, float]:
        """Share of training track occurrences belonging to each artist (sums to 1)."""
        total = sum(v for v in self.artist_frequencies.values() if v > 0)
        if total == 0:
            return {}
        return {a: v / total for a, v in self.artist_frequencies.items() if v > 0}

    def candidates_from_cooccurrence(self, prefix_tracks: List[str], top_n: int = 100) -> List[str]:
        """
        Candidate generation: tracks that co-occurred with the prefix tracks in the
        training data, ranked by summed co-occurrence count. If fewer than top_n are
        found, the list is filled with the most frequent training tracks.
        Prefix tracks are never returned.
        """
        prefix_set = set(prefix_tracks)
        candidate_scores: Counter = Counter()
        for prefix_track in prefix_tracks:
            for track, count in self.track_cooccurrence.get(prefix_track, {}).items():
                if count > 0 and track not in prefix_set:
                    candidate_scores[track] += count

        candidates = [t for t, _ in candidate_scores.most_common(top_n)]
        if len(candidates) < top_n:
            chosen = set(candidates) | prefix_set
            for track, count in self.track_frequencies.most_common():
                if len(candidates) >= top_n:
                    break
                if count > 0 and track not in chosen:
                    candidates.append(track)
                    chosen.add(track)
        return candidates

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
        track_freq = self._freq(self.track_frequencies, candidate_track)
        features.append(np.log1p(track_freq))
        
        # Candidate artist frequency (log scale)
        artist_freq = self._freq(self.artist_frequencies, candidate_artist)
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
        prefix_popularities = [self._freq(self.track_frequencies, t) for t in prefix_track_uris]
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

        Every (prefix, continuation) pair becomes one *group*: its continuation
        tracks are positives (label 1), and `num_negatives` random tracks per
        positive are negatives (label 0). Records carry the group id and artist,
        which the constraint losses need during training.

        For the train split, the statistics used for a playlist's pairs are computed
        with all of that playlist's pairs removed (leave-one-playlist-out). Without
        this, each positive would be counted in its own co-occurrence feature, which
        leaks the label: the model then learns "co-occurrence > 0 means positive",
        a signal that does not exist for unseen playlists.

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

        all_tracks = [t for t, c in self.track_frequencies.items() if c > 0]
        if not all_tracks:
            raise RuntimeError("No training statistics: build train_pairs.json first")

        # Group pairs by playlist so train pairs can be featurized leave-one-playlist-out.
        by_playlist: Dict[int, List[Dict]] = defaultdict(list)
        for pair in pairs:
            by_playlist[pair['playlist_id']].append(pair)

        feature_data = []
        group_id = 0
        done = 0

        for playlist_id, playlist_pairs in by_playlist.items():
            held_out = self.train_pairs_by_playlist.get(playlist_id, []) if split_name == 'train' else []
            self._apply_pairs(held_out, -1)
            try:
                for pair in playlist_pairs:
                    feature_data.extend(self._featurize_pair(pair, group_id, all_tracks, num_negatives))
                    group_id += 1
                    done += 1
                    if done % 1000 == 0:
                        logger.info(f"  Processed {done}/{len(pairs)} pairs...")
            finally:
                self._apply_pairs(held_out, +1)

        # Save feature dataset
        with open(output_file, 'w') as f:
            json.dump(feature_data, f)
        
        logger.info(f"Saved {len(feature_data)} feature vectors to {output_file}")
        logger.info(f"  Positives: {sum(1 for x in feature_data if x['label'] == 1)}")
        logger.info(f"  Negatives: {sum(1 for x in feature_data if x['label'] == 0)}")

    def _featurize_pair(self, pair: Dict, group_id: int, all_tracks: List[str], num_negatives: int) -> List[Dict]:
        """Feature records for one pair: its positives plus sampled negatives."""
        prefix = pair['prefix']
        continuation = pair['continuation']
        recent_artists = {t['artist_uri'] for t in prefix[-5:]}

        def record(track_uri: str, label: int) -> Dict:
            artist = self.track_to_artist.get(track_uri, 'unknown')
            return {
                'features': [round(float(v), 6) for v in self.extract_features(prefix, track_uri)],
                'label': label,
                'group': group_id,
                'track_uri': track_uri,
                'artist_uri': artist,
                'artist_in_recent_prefix': artist in recent_artists,
                'prefix_length': len(prefix),
            }

        records = [record(t['track_uri'], 1) for t in continuation]

        # Negatives: tracks that are neither in the continuation nor in the prefix.
        excluded = {t['track_uri'] for t in continuation} | {t['track_uri'] for t in prefix}
        wanted = num_negatives * len(continuation)
        if len(all_tracks) - len(excluded & set(all_tracks)) < 1:
            return records
        negatives = []
        while len(negatives) < wanted:
            neg_track = all_tracks[self.rng.integers(len(all_tracks))]
            if neg_track not in excluded:
                negatives.append(neg_track)
        records.extend(record(t, 0) for t in negatives)
        return records


def main():
    """Main entry point for feature building."""
    parser = argparse.ArgumentParser(description="Build features for ranking")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Processed data directory")
    parser.add_argument("--num-negatives", type=int, default=5, help="Negatives per positive")
    
    args = parser.parse_args()
    
    builder = FeatureBuilder(data_dir=args.data_dir)

    # Artist popularity prior, used by the exposure-fairness loss during training.
    with open(Path(args.data_dir) / "artist_popularity.json", 'w') as f:
        json.dump(builder.artist_popularity(), f)
    
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
