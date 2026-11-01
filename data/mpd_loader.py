"""
Spotify Million Playlist Dataset (MPD) Loader

Downloads and parses the Spotify MPD dataset.
Dataset: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Iterator
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
        
        # Shared track catalog: every artist has TRACKS_PER_ARTIST tracks, and the same
        # track appears in many playlists, so co-occurrence statistics learned on the
        # training split carry over to validation and test (as in the real MPD).
        import random
        rng = random.Random(42)
        TRACKS_PER_ARTIST = 40
        catalog = {
            artist_uri: [f"spotify:track:{artist_uri}_{i}" for i in range(TRACKS_PER_ARTIST)]
            for artist_uri, _ in artists
        }
        # Within an artist, low-index tracks are more popular (Zipf-like weights).
        track_weights = [1.0 / (i + 1) for i in range(TRACKS_PER_ARTIST)]
        artist_names = dict(artists)
        artist_uris = [a for a, _ in artists]

        for pid in range(1000):
            num_tracks = rng.randint(10, 50)
            # Each playlist has 2-3 favourite artists; 80% of its tracks come from them.
            favourites = rng.sample(artist_uris, rng.randint(2, 3))
            tracks = []
            used = set()

            while len(tracks) < num_tracks:
                artist_uri = rng.choice(favourites) if rng.random() < 0.8 else rng.choice(artist_uris)
                track_uri = rng.choices(catalog[artist_uri], weights=track_weights)[0]
                if track_uri in used:
                    continue
                used.add(track_uri)
                pos = len(tracks)
                idx = int(track_uri.rsplit("_", 1)[1])
                artist_name = artist_names[artist_uri]

                tracks.append({
                    "pos": pos,
                    "track_uri": track_uri,
                    "track_name": f"Song {idx} by {artist_name}",
                    "artist_uri": f"spotify:artist:{artist_uri}",
                    "artist_name": artist_name,
                    "album_uri": f"spotify:album:album_{artist_uri}_{idx // 10}",
                    "album_name": f"Album {idx // 10}",
                    "duration_ms": rng.randint(180000, 300000)
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
        logger.info("Sample contains 1000 playlists over a 400-track catalog (10 artists x 40 tracks)")
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
        print("\nFirst 3 tracks:")
        for track in first_playlist['tracks'][:3]:
            print(f"  - {track['track_name']} by {track['artist_name']}")
    else:
        logger.error("No playlists found. Run with --download for instructions.")


if __name__ == "__main__":
    main()
