"""Data model shared by every insights module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

AUDIO_KEYS = (
    "danceability", "energy", "valence", "acousticness", "instrumentalness",
    "liveness", "speechiness", "loudness", "tempo",
)


def norm_key(text: str) -> str:
    """Lower-case, strip bracketed suffixes ('(Remastered 2011)', '- Live') and punctuation."""
    t = text.lower()
    t = re.sub(r"\s*[\(\[].*?[\)\]]", "", t)
    t = re.sub(r"\s+-\s+.*$", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Track:
    title: str
    artist: str
    artists: List[str] = field(default_factory=list)
    album: str = ""
    duration_ms: Optional[int] = None
    spotify_uri: Optional[str] = None
    artist_uri: Optional[str] = None
    isrc: Optional[str] = None
    release_date: Optional[str] = None
    explicit: Optional[bool] = None
    added_at: Optional[str] = None
    added_by: Optional[str] = None
    popularity: Optional[float] = None
    genres: List[str] = field(default_factory=list)
    audio: Dict[str, float] = field(default_factory=dict)
    # Filled by insights.enrich
    lyrics: Optional[str] = None
    instrumental: Optional[bool] = None
    bpm: Optional[float] = None
    gain: Optional[float] = None
    deezer_id: Optional[int] = None
    deezer_rank: Optional[int] = None
    preview_url: Optional[str] = None
    cover_url: Optional[str] = None

    def __post_init__(self):
        if not self.artists:
            self.artists = [self.artist]

    @property
    def key(self) -> str:
        return f"{norm_key(self.artist)}|{norm_key(self.title)}"

    @property
    def spotify_id(self) -> Optional[str]:
        if self.spotify_uri and self.spotify_uri.startswith("spotify:track:"):
            return self.spotify_uri.rsplit(":", 1)[1]
        return None

    @property
    def year(self) -> Optional[int]:
        if self.release_date and re.match(r"^\d{4}", self.release_date):
            y = int(self.release_date[:4])
            return y if y > 0 else None
        return None

    @property
    def tempo(self) -> Optional[float]:
        """BPM from Deezer if known (and non-zero), else the audio-features tempo."""
        if self.bpm:
            return self.bpm
        return self.audio.get("tempo") or None


@dataclass
class ArtistInfo:
    name: str
    tags: Dict[str, float] = field(default_factory=dict)   # tag -> weight in [0, 1]
    similar: List[Tuple[str, float]] = field(default_factory=list)  # (name, match in [0, 1])
    country: Optional[str] = None       # ISO 3166-1 alpha-2, from MusicBrainz
    begin_year: Optional[int] = None
    kind: Optional[str] = None          # Person / Group / ...
    mbid: Optional[str] = None
    deezer_id: Optional[int] = None
    fans: Optional[int] = None
    bio: Optional[str] = None
    image_url: Optional[str] = None
    wiki_url: Optional[str] = None


@dataclass
class Playlist:
    name: str
    tracks: List[Track]
    owner: str = "me"
    source: str = ""
    artists: Dict[str, ArtistInfo] = field(default_factory=dict)  # keyed by artist name

    def artist_info(self, name: str) -> ArtistInfo:
        if name not in self.artists:
            self.artists[name] = ArtistInfo(name=name)
        return self.artists[name]

    def tags_for(self, track: Track) -> Dict[str, float]:
        """Genre/tag profile of a track: its own genres, else its primary artist's tags."""
        if track.genres:
            return {g.lower(): 1.0 for g in track.genres}
        info = self.artists.get(track.artist)
        return dict(info.tags) if info else {}
