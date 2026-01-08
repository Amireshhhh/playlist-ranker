"""
Spotify Million Playlist Dataset (MPD) Loader

Downloads and parses the Spotify MPD dataset.
Dataset: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Iterator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MPDLoader:
    """Load and parse Spotify Million Playlist Dataset."""
    
    def __init__(self, data_dir: str = "./data/mpd"):
        """
        Initialize MPD loader.
        
        Args:
            data_dir: Directory containing MPD JSON files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_instructions(self):
        """Print instructions for downloading MPD dataset."""
        instructions = """
        ===================================================================
        SPOTIFY MILLION PLAYLIST DATASET DOWNLOAD INSTRUCTIONS
        ===================================================================
        
        The Spotify MPD is available at:
        https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
        
        Steps:
        1. Create account on AIcrowd (free)
        2. Accept dataset terms
        3. Download the dataset (spotify_million_playlist_dataset.zip, ~5GB)
        4. Extract to: {data_dir}
        
        Expected structure after extraction:
        {data_dir}/
        ├── data/
        │   ├── mpd.slice.0-999.json
        │   ├── mpd.slice.1000-1999.json
        │   ├── ...
        │   └── mpd.slice.999000-999999.json
        
        Alternative: Use sample data for quick testing
        A sample subset will be generated automatically if full dataset unavailable.
        
        ===================================================================
        """.format(data_dir=self.data_dir)
        print(instructions)
        
    def iter_playlists(self) -> Iterator[Dict]:
        """
        Iterate over all playlists in the dataset.
        
        Yields:
            Dict with keys: pid, name, num_tracks, tracks
            Each track has: track_uri, track_name, artist_uri, artist_name, album_uri, pos
        """
        json_files = sorted(self.data_dir.glob("**/*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir}")
            logger.info("Creating sample data for testing...")
            self._create_sample_data()
            json_files = sorted(self.data_dir.glob("**/*.json"))
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for playlist in data.get('playlists', []):
                    yield {
                        'pid': playlist['pid'],
                        'name': playlist.get('name', ''),
                        'num_tracks': playlist.get('num_tracks', len(playlist.get('tracks', []))),
                        'tracks': playlist.get('tracks', [])
                    }
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
    
    def _create_sample_data(self):
        """Create sample data for testing when full MPD is unavailable."""
        logger.info("Generating sample dataset...")
        
        # Sample artists and tracks
        sample_data = {
            "info": {
                "generated_on": "2026-01-03",
                "slice": "0-999",
                "version": "v1"
            },
            "playlists": []
        }
        
        # Sample artist pool
        artists = [
            ("artist_0", "The Beatles"),
            ("artist_1", "Led Zeppelin"),
            ("artist_2", "Pink Floyd"),
            ("artist_3", "Radiohead"),
            ("artist_4", "Nirvana"),
            ("artist_5", "Queen"),
            ("artist_6", "David Bowie"),
            ("artist_7", "The Rolling Stones"),
            ("artist_8", "Fleetwood Mac"),
            ("artist_9", "Arctic Monkeys"),
        ]
        
        # Generate sample playlists
        import random
        random.seed(42)
        
        for pid in range(1000):
            num_tracks = random.randint(10, 50)
            tracks = []
            
            for pos in range(num_tracks):
                artist_uri, artist_name = random.choice(artists)
                track_id = f"track_{pid}_{pos}"
                
                tracks.append({
                    "pos": pos,
                    "track_uri": f"spotify:track:{track_id}",
                    "track_name": f"Song {pos} by {artist_name}",
                    "artist_uri": f"spotify:artist:{artist_uri}",
                    "artist_name": artist_name,
                    "album_uri": f"spotify:album:album_{artist_uri}_{pos // 3}",
                    "album_name": f"Album {pos // 3}",
                    "duration_ms": random.randint(180000, 300000)
                })
            
            sample_data["playlists"].append({
                "pid": pid,
                "name": f"Sample Playlist {pid}",
                "num_tracks": num_tracks,
                "num_artists": len(set(t["artist_uri"] for t in tracks)),
                "num_albums": len(set(t["album_uri"] for t in tracks)),
                "tracks": tracks
            })
        
        # Save sample data
        sample_file = self.data_dir / "mpd.slice.0-999.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Created sample dataset: {sample_file}")
        logger.info("Sample contains 1000 playlists for testing")
        logger.info("For production use, download the full MPD dataset")
    
    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics.
        
        Returns:
            Dict with dataset statistics
        """
        stats = {
            'num_playlists': 0,
            'num_tracks': 0,
            'unique_tracks': set(),
            'unique_artists': set(),
            'unique_albums': set(),
            'total_track_occurrences': 0,
            'avg_playlist_length': 0
        }
        
        for playlist in self.iter_playlists():
            stats['num_playlists'] += 1
            stats['total_track_occurrences'] += len(playlist['tracks'])
            
            for track in playlist['tracks']:
                stats['unique_tracks'].add(track['track_uri'])
                stats['unique_artists'].add(track['artist_uri'])
                stats['unique_albums'].add(track['album_uri'])
        
        stats['num_unique_tracks'] = len(stats['unique_tracks'])
        stats['num_unique_artists'] = len(stats['unique_artists'])
        stats['num_unique_albums'] = len(stats['unique_albums'])
        stats['avg_playlist_length'] = stats['total_track_occurrences'] / max(stats['num_playlists'], 1)
        
        # Remove sets from output (not JSON serializable)
        del stats['unique_tracks']
        del stats['unique_artists']
        del stats['unique_albums']
        
        return stats


def main():
    """Main entry point for MPD loader."""
    parser = argparse.ArgumentParser(description="Spotify MPD Dataset Loader")
    parser.add_argument("--download", action="store_true", help="Show download instructions")
    parser.add_argument("--data-dir", type=str, default="./data/mpd", help="MPD data directory")
    parser.add_argument("--stats", action="store_true", help="Compute dataset statistics")
    parser.add_argument("--sample", action="store_true", help="Create sample dataset")
    
    args = parser.parse_args()
    
    loader = MPDLoader(data_dir=args.data_dir)
    
    if args.download:
        loader.download_instructions()
        return
    
    if args.sample:
        loader._create_sample_data()
        return
    
    if args.stats:
        logger.info("Computing dataset statistics...")
        stats = loader.get_statistics()
        
        print("\n" + "="*60)
        print("SPOTIFY MPD STATISTICS")
        print("="*60)
        print(f"Playlists:       {stats['num_playlists']:,}")
        print(f"Unique Tracks:   {stats['num_unique_tracks']:,}")
        print(f"Unique Artists:  {stats['num_unique_artists']:,}")
        print(f"Unique Albums:   {stats['num_unique_albums']:,}")
        print(f"Avg Playlist Len: {stats['avg_playlist_length']:.1f} tracks")
        print("="*60 + "\n")
        return
    
    # Default: show first playlist
    loader_iter = loader.iter_playlists()
    first_playlist = next(loader_iter, None)
    
    if first_playlist:
        print(f"\nFirst playlist (pid={first_playlist['pid']}):")
        print(f"Name: {first_playlist['name']}")
        print(f"Tracks: {first_playlist['num_tracks']}")
        print(f"\nFirst 3 tracks:")
        for track in first_playlist['tracks'][:3]:
            print(f"  - {track['track_name']} by {track['artist_name']}")
    else:
        logger.error("No playlists found. Run with --download for instructions.")


if __name__ == "__main__":
    main()
