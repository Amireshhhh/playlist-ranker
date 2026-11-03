"""Load a playlist from free sources into :class:`insights.models.Playlist`.

Supported inputs
- Exportify CSV (https://github.com/watsonbox/exportify): Spotify playlists, Liked Songs.
- Any CSV with a title column and an artist column (name variants are detected).
- An MPD slice JSON (the dataset this repo trains on), one playlist selected by ``pid``.
- A JSON list of {"title", "artist", ...} objects.
- A public Deezer playlist (no account or key needed), see :func:`load_deezer_playlist`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import AUDIO_KEYS, Playlist, Track

TITLE_COLS = ("track name", "track_name", "title", "track", "song", "name")
ARTIST_COLS = ("artist name(s)", "artist_name", "artist", "artists", "artist name")


def _find(header: Dict[str, str], candidates) -> Optional[str]:
    for c in candidates:
        if c in header:
            return header[c]
    return None


def _split_artists(names: str, uris: str) -> List[str]:
    """Exportify joins multiple artists with ','. Artist names can contain commas
    ('Tyler, The Creator'), so split only when the URI column has the same count."""
    if not names:
        return []
    parts = [p.strip() for p in names.split(",") if p.strip()]
    if not uris:
        return parts
    n_uris = len([u for u in uris.split(",") if u.strip()])
    return parts if n_uris == len(parts) else [names.strip()]


def _float(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def load_csv(path: str, name: Optional[str] = None, owner: str = "me") -> Playlist:
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path}: empty CSV")
        header = {h.strip().lower(): h for h in reader.fieldnames}
        title_col = _find(header, TITLE_COLS)
        artist_col = _find(header, ARTIST_COLS)
        if not title_col or not artist_col:
            raise ValueError(
                f"{path}: need a title column (one of {TITLE_COLS}) and an artist column "
                f"(one of {ARTIST_COLS}); found {reader.fieldnames}"
            )
        col = lambda *names: _find(header, names)  # noqa: E731
        uri_col, auri_col = col("track uri", "track_uri", "uri"), col("artist uri(s)", "artist_uri")
        tracks = []
        for row in reader:
            title = (row.get(title_col) or "").strip()
            artist_field = (row.get(artist_col) or "").strip()
            if not title or not artist_field:
                continue
            artist_uris = row.get(auri_col, "") if auri_col else ""
            artists = _split_artists(artist_field, artist_uris)
            genres_field = row.get(col("artist genres", "genres", "genre") or "", "") or ""
            dur = _float(row.get(col("track duration (ms)", "duration_ms", "duration") or "", ""))
            explicit = row.get(col("explicit?", "explicit") or "", "")
            audio = {}
            for k in AUDIO_KEYS:
                v = _float(row.get(header.get(k, ""), ""))
                if v is not None:
                    audio[k] = v
            tracks.append(Track(
                title=title,
                artist=artists[0],
                artists=artists,
                album=(row.get(col("album name", "album") or "", "") or "").strip(),
                duration_ms=int(dur) if dur else None,
                spotify_uri=(row.get(uri_col, "") or None) if uri_col else None,
                artist_uri=(artist_uris.split(",")[0].strip() or None) if artist_uris else None,
                isrc=(row.get(col("isrc") or "", "") or None),
                release_date=(row.get(col("album release date", "release_date", "year") or "", "") or None),
                explicit=(explicit.strip().lower() == "true") if explicit else None,
                added_at=(row.get(col("added at", "added_at") or "", "") or None),
                added_by=(row.get(col("added by", "added_by") or "", "") or None),
                popularity=_float(row.get(col("popularity") or "", "")),
                genres=[g.strip() for g in genres_field.split(",") if g.strip()],
                audio=audio,
            ))
    if not tracks:
        raise ValueError(f"{path}: no rows with both a title and an artist")
    return Playlist(name=name or path.stem.replace("_", " "), tracks=tracks, owner=owner,
                    source=f"csv:{path.name}")


def _mpd_track(t: Dict) -> Track:
    return Track(
        title=t.get("track_name", ""), artist=t.get("artist_name", ""), album=t.get("album_name", ""),
        duration_ms=t.get("duration_ms"), spotify_uri=t.get("track_uri"), artist_uri=t.get("artist_uri"),
    )


def load_mpd_playlist(path: str, pid: int, owner: str = "me") -> Playlist:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for p in data.get("playlists", []):
        if p.get("pid") == pid:
            return Playlist(name=p.get("name") or f"MPD {pid}", tracks=[_mpd_track(t) for t in p["tracks"]],
                            owner=owner, source=f"mpd:{Path(path).name}#{pid}")
    raise KeyError(f"pid {pid} not found in {path}")


def load_json(path: str, name: Optional[str] = None, owner: str = "me") -> Playlist:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "playlists" in data:
        raise ValueError("This is an MPD slice; pass --pid to choose a playlist")
    if isinstance(data, dict):
        name = name or data.get("name")
        owner = data.get("owner", owner)
        data = data.get("tracks", [])
    tracks = []
    for t in data:
        title = t.get("title") or t.get("track_name")
        artist = t.get("artist") or t.get("artist_name")
        if title and artist:
            extra = {k: t[k] for k in ("album", "duration_ms", "spotify_uri", "isrc", "release_date",
                                       "explicit", "added_at", "genres", "audio", "lyrics") if k in t}
            tracks.append(Track(title=title, artist=artist, **extra))
    if not tracks:
        raise ValueError(f"{path}: no tracks with title and artist")
    return Playlist(name=name or Path(path).stem, tracks=tracks, owner=owner, source=f"json:{Path(path).name}")


def load_deezer_playlist(playlist_id: int, deezer, owner: Optional[str] = None) -> Playlist:
    """Read a public Deezer playlist through :class:`insights.sources.Deezer` (no key needed)."""
    meta = deezer.playlist(playlist_id)
    if not meta:
        raise ValueError(f"Deezer playlist {playlist_id} not found or not public")
    tracks = []
    for t in deezer.playlist_tracks(playlist_id):
        tracks.append(Track(
            title=t.get("title", ""), artist=t.get("artist", {}).get("name", ""),
            album=t.get("album", {}).get("title", ""),
            duration_ms=(t["duration"] * 1000) if t.get("duration") else None,
            explicit=t.get("explicit_lyrics"), deezer_id=t.get("id"), deezer_rank=t.get("rank"),
            preview_url=t.get("preview") or None, cover_url=t.get("album", {}).get("cover_medium"),
            added_at=(str(t["time_add"]) if t.get("time_add") else None),
        ))
    return Playlist(name=meta.get("title", f"Deezer {playlist_id}"), tracks=tracks,
                    owner=owner or meta.get("creator", {}).get("name", "me"), source=f"deezer:{playlist_id}")


def load(path_or_ref: str, pid: Optional[int] = None, name: Optional[str] = None,
         owner: str = "me", deezer=None) -> Playlist:
    """Dispatch on the reference: 'deezer:<id>', '*.csv', '*.json' (MPD slice with pid)."""
    if path_or_ref.startswith("deezer:"):
        if deezer is None:
            raise ValueError("deezer:<id> needs network access (drop --offline)")
        return load_deezer_playlist(int(path_or_ref.split(":", 1)[1]), deezer, owner=owner)
    p = Path(path_or_ref)
    if p.suffix.lower() == ".csv":
        return load_csv(p, name=name, owner=owner)
    if p.suffix.lower() == ".json":
        if pid is not None:
            pl = load_mpd_playlist(p, pid, owner=owner)
            if name:
                pl.name = name
            return pl
        return load_json(p, name=name, owner=owner)
    raise ValueError(f"Unsupported input {path_or_ref!r} (use .csv, .json or deezer:<playlist id>)")
