"""Build an artist's lyric corpus: studio albums from MusicBrainz, lyrics from LRCLIB."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..models import norm_key

logger = logging.getLogger(__name__)
MB = "https://musicbrainz.org/ws/2"


@dataclass
class Song:
    title: str
    album: str
    year: int
    order: int            # album index in release order, 0-based
    position: int         # track number on the album
    duration_s: Optional[int]
    lyrics: Optional[str] = None
    instrumental: bool = False
    status: str = "pending"   # lyrics | instrumental | missing | duplicate


@dataclass
class Corpus:
    artist: str
    mbid: Optional[str]
    albums: List[Dict]                    # {title, year, order, tracks}
    songs: List[Song] = field(default_factory=list)
    source: str = ""

    @property
    def with_lyrics(self) -> List[Song]:
        return [s for s in self.songs if s.status == "lyrics"]


def _year(date: str) -> Optional[int]:
    m = re.match(r"^(\d{4})", date or "")
    return int(m.group(1)) if m else None


def find_artist(http, name: str) -> Optional[Dict]:
    res = http.get_json(f"{MB}/artist", {"query": f'artist:"{name}"', "fmt": "json", "limit": 5})
    arts = (res or {}).get("artists") or []
    exact = [a for a in arts if norm_key(a.get("name", "")) == norm_key(name)]
    best = (exact or arts or [None])[0]
    if best is None or int(best.get("score", 0)) < 90:
        return None
    return best


def studio_albums(http, mbid: str) -> List[Dict]:
    groups, offset = [], 0
    while True:
        res = http.get_json(f"{MB}/release-group", {"artist": mbid, "type": "album", "fmt": "json",
                                                    "limit": 100, "offset": offset})
        page = (res or {}).get("release-groups") or []
        groups += page
        if len(page) < 100:
            break
        offset += 100
    out = []
    for g in groups:
        if g.get("primary-type") != "Album" or g.get("secondary-types"):
            continue
        y = _year(g.get("first-release-date", ""))
        if y:
            out.append({"id": g["id"], "title": g["title"], "year": y, "date": g.get("first-release-date")})
    out.sort(key=lambda g: (g["date"], g["title"]))
    return out


def canonical_tracklist(http, rg_id: str) -> List[Dict]:
    """Tracks of the release whose track count is the most common among official releases
    (so deluxe bonus tracks do not count), earliest such release first."""
    res = http.get_json(f"{MB}/release", {"release-group": rg_id, "status": "official",
                                          "inc": "recordings+media", "fmt": "json", "limit": 100})
    rels = (res or {}).get("releases") or []
    if not rels:
        return []

    def tracks(r):
        return [t for m in r.get("media") or [] for t in m.get("tracks") or []]

    counts = Counter(len(tracks(r)) for r in rels if tracks(r))
    if not counts:
        return []
    mode = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    pick = sorted((r for r in rels if len(tracks(r)) == mode), key=lambda r: r.get("date") or "9999")[0]
    out = []
    for i, t in enumerate(tracks(pick)):
        length = t.get("length") or (t.get("recording") or {}).get("length")
        out.append({"title": t.get("title") or (t.get("recording") or {}).get("title", ""), "position": i + 1,
                    "duration_s": round(length / 1000) if length else None})
    return out


def build(src, artist: str, max_albums: int = 20, progress=None) -> Optional[Corpus]:
    http = src.http
    a = find_artist(http, artist)
    if not a:
        logger.warning("MusicBrainz: no confident match for %r", artist)
        return None
    albums = studio_albums(http, a["id"])[:max_albums]
    corpus = Corpus(artist=a.get("name", artist), mbid=a["id"], albums=[], source="MusicBrainz + LRCLIB")
    seen = set()
    for order, alb in enumerate(albums):
        tl = canonical_tracklist(http, alb["id"])
        corpus.albums.append({"title": alb["title"], "year": alb["year"], "order": order, "tracks": len(tl)})
        for t in tl:
            s = Song(title=t["title"], album=alb["title"], year=alb["year"], order=order, position=t["position"],
                     duration_s=t["duration_s"])
            key = norm_key(t["title"])
            if key in seen:              # re-recording / reprise of an earlier title
                s.status = "duplicate"
                corpus.songs.append(s)
                continue
            seen.add(key)
            hit = src.lrclib.lyrics(t["title"], corpus.artist, alb["title"], t["duration_s"])
            if hit and hit.get("instrumental"):
                s.status, s.instrumental = "instrumental", True
            elif hit and (hit.get("plainLyrics") or "").strip():
                # Reject a search hit for a different song: titles must match after normalising.
                if norm_key(hit.get("trackName", t["title"])) == key or hit.get("trackName") is None:
                    s.lyrics, s.status = hit["plainLyrics"], "lyrics"
                else:
                    s.status = "missing"
            else:
                s.status = "missing"
            corpus.songs.append(s)
        if progress:
            progress("albums", order + 1, len(albums))
    # Albums with no lyrics at all (e.g. instrumental records) are dropped from comparisons.
    have = {s.order for s in corpus.with_lyrics}
    corpus.albums = [al for al in corpus.albums if al["order"] in have]
    remap = {al["order"]: i for i, al in enumerate(corpus.albums)}
    for al in corpus.albums:
        al["order"] = remap[al["order"]]
    for s in corpus.songs:
        s.order = remap.get(s.order, -1)
    return corpus
