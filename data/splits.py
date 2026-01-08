"""
Train/Validation/Test Split Generation

Generates prefix-continuation pairs for training the ranking model.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import logging

from mpd_loader import MPDLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SplitGenerator:
    """Generate train/val/test splits from MPD dataset."""
    
    def __init__(
        self,
        mpd_loader: MPDLoader,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Initialize split generator.
        
        Args:
            mpd_loader: MPD dataset loader
            train_ratio: Fraction for training (default: 0.7)
            val_ratio: Fraction for validation (default: 0.15)
            test_ratio: Fraction for test (default: 0.15)
            seed: Random seed for reproducibility
        """
        self.mpd_loader = mpd_loader
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        random.seed(seed)
    
    def generate_prefix_continuation_pairs(
        self,
        playlist: Dict,
        prefix_lengths: List[int] = [5, 10, 25]
    ) -> List[Dict]:
        """
        Generate prefix-continuation pairs from a playlist.
        
        Args:
            playlist: Playlist dict with 'tracks' key
            prefix_lengths: List of prefix lengths to generate
            
        Returns:
            List of dicts with 'prefix', 'continuation', and metadata
        """
        tracks = playlist['tracks']
        num_tracks = len(tracks)
        
        pairs = []
        
        for prefix_len in prefix_lengths:
            # Need at least prefix_len + 1 tracks (for at least 1 continuation)
            if num_tracks < prefix_len + 1:
                continue
            
            # Multiple samples per prefix length if playlist is long enough
            max_continuations = min(3, num_tracks - prefix_len)
            
            for cont_start in range(prefix_len, min(prefix_len + max_continuations, num_tracks)):
                continuation_len = min(10, num_tracks - cont_start)
                
                if continuation_len < 1:
                    continue
                
                prefix_tracks = tracks[:prefix_len]
                continuation_tracks = tracks[cont_start:cont_start + continuation_len]
                
                pairs.append({
                    'playlist_id': playlist['pid'],
                    'playlist_name': playlist['name'],
                    'prefix_length': prefix_len,
                    'continuation_length': continuation_len,
                    'prefix': [
                        {
                            'track_uri': t['track_uri'],
                            'artist_uri': t['artist_uri'],
                            'pos': t['pos']
                        }
                        for t in prefix_tracks
                    ],
                    'continuation': [
                        {
                            'track_uri': t['track_uri'],
                            'artist_uri': t['artist_uri'],
                            'pos': t['pos']
                        }
                        for t in continuation_tracks
                    ]
                })
        
        return pairs
    
    def generate_splits(self, output_dir: str, max_playlists: int = None):
        """
        Generate and save train/val/test splits.
        
        Args:
            output_dir: Directory to save splits
            max_playlists: Maximum playlists to process (for testing)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating splits...")
        
        # Collect all playlists
        playlists = []
        for i, playlist in enumerate(self.mpd_loader.iter_playlists()):
            playlists.append(playlist)
            
            if max_playlists and i + 1 >= max_playlists:
                break
            
            if (i + 1) % 10000 == 0:
                logger.info(f"Loaded {i + 1} playlists...")
        
        logger.info(f"Total playlists: {len(playlists)}")
        
        # Shuffle playlists
        random.shuffle(playlists)
        
        # Split indices
        n = len(playlists)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        splits = {
            'train': playlists[:train_end],
            'val': playlists[train_end:val_end],
            'test': playlists[val_end:]
        }
        
        logger.info(f"Train: {len(splits['train'])} playlists")
        logger.info(f"Val: {len(splits['val'])} playlists")
        logger.info(f"Test: {len(splits['test'])} playlists")
        
        # Generate prefix-continuation pairs for each split
        for split_name, split_playlists in splits.items():
            logger.info(f"\nProcessing {split_name} split...")
            
            all_pairs = []
            for i, playlist in enumerate(split_playlists):
                pairs = self.generate_prefix_continuation_pairs(playlist)
                all_pairs.extend(pairs)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"  Processed {i + 1}/{len(split_playlists)} playlists...")
            
            # Save to JSON
            output_file = output_path / f"{split_name}_pairs.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_pairs, f, indent=2)
            
            logger.info(f"Saved {len(all_pairs)} pairs to {output_file}")
        
        # Save metadata
        metadata = {
            'seed': self.seed,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'num_playlists': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test'])
            },
            'total_playlists': len(playlists)
        }
        
        metadata_file = output_path / "split_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\nSplit generation complete!")
        logger.info(f"Output directory: {output_path}")
    
    def build_track_artist_mapping(self, output_dir: str):
        """
        Build global track->artist and artist->tracks mappings.
        
        Args:
            output_dir: Directory to save mappings
        """
        logger.info("Building track-artist mappings...")
        
        track_to_artist = {}
        artist_to_tracks = defaultdict(set)
        track_names = {}
        artist_names = {}
        
        for playlist in self.mpd_loader.iter_playlists():
            for track in playlist['tracks']:
                track_uri = track['track_uri']
                artist_uri = track['artist_uri']
                
                track_to_artist[track_uri] = artist_uri
                artist_to_tracks[artist_uri].add(track_uri)
                track_names[track_uri] = track['track_name']
                artist_names[artist_uri] = track['artist_name']
        
        # Convert sets to lists for JSON serialization
        artist_to_tracks = {k: list(v) for k, v in artist_to_tracks.items()}
        
        output_path = Path(output_dir)
        
        # Save mappings
        with open(output_path / "track_to_artist.json", 'w') as f:
            json.dump(track_to_artist, f)
        
        with open(output_path / "artist_to_tracks.json", 'w') as f:
            json.dump(artist_to_tracks, f)
        
        with open(output_path / "track_names.json", 'w') as f:
            json.dump(track_names, f)
        
        with open(output_path / "artist_names.json", 'w') as f:
            json.dump(artist_names, f)
        
        logger.info(f"Saved mappings:")
        logger.info(f"  Tracks: {len(track_to_artist)}")
        logger.info(f"  Artists: {len(artist_to_tracks)}")


def main():
    """Main entry point for split generation."""
    parser = argparse.ArgumentParser(description="Generate train/val/test splits")
    parser.add_argument("--input-dir", type=str, default="./data/mpd", help="MPD data directory")
    parser.add_argument("--output-dir", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--max-playlists", type=int, default=None, help="Max playlists (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load MPD
    loader = MPDLoader(data_dir=args.input_dir)
    
    # Generate splits
    generator = SplitGenerator(mpd_loader=loader, seed=args.seed)
    generator.generate_splits(output_dir=args.output_dir, max_playlists=args.max_playlists)
    
    # Build mappings
    generator.build_track_artist_mapping(output_dir=args.output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("SPLIT GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Output: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Run feature extraction: python features/build_features.py")
    logger.info("  2. Train model: python training/train.py")
    logger.info("="*60)


if __name__ == "__main__":
    main()
