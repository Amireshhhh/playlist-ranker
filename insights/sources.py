"""Clients for the free music APIs. Each method returns parsed data or None.

| Source      | Key needed            | Used for                                             |
|-------------|-----------------------|------------------------------------------------------|
| LRCLIB      | no                    | plain + synced lyrics, instrumental flag             |
| Deezer      | no (public data)      | BPM, gain, popularity rank, 30 s previews, covers,   |
|             |                       | album genres, related artists, artist top tracks,    |
|             |                       | public playlists                                     |
| MusicBrainz | no (User-Agent)       | artist country, type, start year, community tags     |
| Last.fm     | free key (LASTFM_API_KEY) | artist tags (weighted), similar artists          |
| ReccoBeats  | no                    | Spotify-style audio features (energy, valence, ...)  |
| Wikipedia   | no                    | artist summary and image                             |
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from .httpcache import CachedHTTP


def _ok(body: Any) -> Optional[Any]:
    """Deezer reports errors as HTTP 200 with an 'error' object."""
    if body is None or (isinstance(body, dict) and "error" in body):
        return None
    return body


class LRCLib:
    BASE = "https://lrclib.net/api"

    def __init__(self, http: CachedHTTP):
        self.http = http

    def lyrics(self, title: str, artist: str, album: str = "", duration_s: Optional[int] = None) -> Optional[Dict]:
        """Exact match via /api/get (needs duration within ~2 s), else best /api/search hit."""
        if duration_s:
            hit = self.http.get_json(f"{self.BASE}/get", {
                "track_name": title, "artist_name": artist, "album_name": album or None, "duration": duration_s})
            if isinstance(hit, dict) and (hit.get("plainLyrics") or hit.get("instrumental")):
                return hit
        results = self.http.get_json(f"{self.BASE}/search", {"track_name": title, "artist_name": artist})
        if not isinstance(results, list) or not results:
            return None
        with_text = [r for r in results if r.get("plainLyrics")]
        if duration_s and with_text:
            with_text.sort(key=lambda r: abs((r.get("duration") or 0) - duration_s))
        return (with_text or results)[0]


class Deezer:
    BASE = "https://api.deezer.com"

    def __init__(self, http: CachedHTTP):
        self.http = http

    def _get(self, path: str, params=None):
        return _ok(self.http.get_json(f"{self.BASE}{path}", params))

    def find_track(self, title: str, artist: str, isrc: Optional[str] = None) -> Optional[Dict]:
        if isrc:
            t = self._get(f"/track/isrc:{isrc}")
            if t and t.get("id"):
                return t
        res = self._get("/search", {"q": f'artist:"{artist}" track:"{title}"', "limit": 5})
        if not res or not res.get("data"):
            res = self._get("/search", {"q": f"{artist} {title}", "limit": 5})
        data = (res or {}).get("data") or []
        return data[0] if data else None

    def track(self, track_id: int) -> Optional[Dict]:
        return self._get(f"/track/{track_id}")

    def album(self, album_id: int) -> Optional[Dict]:
        return self._get(f"/album/{album_id}")

    def artist(self, artist_id: int) -> Optional[Dict]:
        return self._get(f"/artist/{artist_id}")

    def search_artist(self, name: str) -> Optional[Dict]:
        res = self._get("/search/artist", {"q": name, "limit": 1})
        data = (res or {}).get("data") or []
        return data[0] if data else None

    def related(self, artist_id: int, limit: int = 20) -> List[Dict]:
        res = self._get(f"/artist/{artist_id}/related", {"limit": limit})
        return (res or {}).get("data") or []

    def top_tracks(self, artist_id: int, limit: int = 5) -> List[Dict]:
        res = self._get(f"/artist/{artist_id}/top", {"limit": limit})
        return (res or {}).get("data") or []

    def playlist(self, playlist_id: int) -> Optional[Dict]:
        return self._get(f"/playlist/{playlist_id}")

    def playlist_tracks(self, playlist_id: int, page: int = 100) -> Iterator[Dict]:
        index = 0
        while True:
            res = self._get(f"/playlist/{playlist_id}/tracks", {"index": index, "limit": page})
            data = (res or {}).get("data") or []
            yield from data
            if len(data) < page:
                return
            index += page


class MusicBrainz:
    BASE = "https://musicbrainz.org/ws/2"

    def __init__(self, http: CachedHTTP):
        self.http = http

    def artist(self, name: str) -> Optional[Dict]:
        res = self.http.get_json(f"{self.BASE}/artist", {"query": f'artist:"{name}"', "fmt": "json", "limit": 3})
        artists = (res or {}).get("artists") or []
        if not artists:
            return None
        exact = [a for a in artists if a.get("name", "").lower() == name.lower()]
        best = (exact or artists)[0]
        if int(best.get("score", 100)) < 80:
            return None
        return best


class LastFM:
    BASE = "https://ws.audioscrobbler.com/2.0/"

    def __init__(self, http: CachedHTTP, api_key: Optional[str] = None):
        self.http = http
        self.api_key = api_key or os.environ.get("LASTFM_API_KEY")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _call(self, method: str, **params):
        if not self.enabled:
            return None
        body = self.http.get_json(self.BASE, {"method": method, "api_key": self.api_key, "format": "json",
                                              "autocorrect": 1, **params})
        return None if (body is None or "error" in body) else body

    def top_tags(self, artist: str) -> Dict[str, float]:
        body = self._call("artist.gettoptags", artist=artist)
        tags = ((body or {}).get("toptags") or {}).get("tag") or []
        if isinstance(tags, dict):
            tags = [tags]
        out = {}
        for t in tags:
            c = float(t.get("count", 0) or 0)
            if c > 0:
                out[t["name"].lower()] = c / 100.0  # Last.fm scales the top tag to 100
        return out

    def similar(self, artist: str, limit: int = 30) -> List[tuple]:
        body = self._call("artist.getsimilar", artist=artist, limit=limit)
        arts = ((body or {}).get("similarartists") or {}).get("artist") or []
        if isinstance(arts, dict):
            arts = [arts]
        return [(a["name"], float(a.get("match", 0) or 0)) for a in arts if a.get("name")]


class ReccoBeats:
    BASE = "https://api.reccobeats.com/v1"

    def __init__(self, http: CachedHTTP):
        self.http = http

    def audio_features(self, spotify_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Spotify track id -> features. Ids unknown to ReccoBeats are absent."""
        out: Dict[str, Dict[str, float]] = {}
        for i in range(0, len(spotify_ids), 40):
            chunk = spotify_ids[i:i + 40]
            body = self.http.get_json(f"{self.BASE}/audio-features", {"ids": ",".join(chunk)})
            for item in (body or {}).get("content") or []:
                href = item.get("href") or ""
                sid = href.rstrip("/").rsplit("/", 1)[-1] if "spotify.com/track/" in href else None
                if sid in chunk:
                    out[sid] = {k: float(v) for k, v in item.items()
                                if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return out


class Wikipedia:
    BASE = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def __init__(self, http: CachedHTTP):
        self.http = http

    def summary(self, title: str) -> Optional[Dict]:
        body = self.http.get_json(self.BASE + title.replace(" ", "_"))
        if not body or body.get("type") == "disambiguation":
            return None
        return body


class Sources:
    """All clients behind one cache. ``offline=True`` serves only cached responses."""

    def __init__(self, cache_dir: str = ".cache/insights", offline: bool = False,
                 lastfm_key: Optional[str] = None):
        self.http = CachedHTTP(cache_dir=cache_dir, offline=offline)
        self.lrclib = LRCLib(self.http)
        self.deezer = Deezer(self.http)
        self.musicbrainz = MusicBrainz(self.http)
        self.lastfm = LastFM(self.http, lastfm_key)
        self.reccobeats = ReccoBeats(self.http)
        self.wikipedia = Wikipedia(self.http)
