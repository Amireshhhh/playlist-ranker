"""Fill a Playlist with data from the free sources (lyrics, tempo, tags, origin, ...).

Every step is optional and fails soft: a track missing from a source keeps its field None,
and the report says how many tracks each source covered (see :func:`coverage`).
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

from .models import Playlist
from .sources import Sources

logger = logging.getLogger(__name__)

MUSIC_WORDS = re.compile(r"\b(band|singer|rapper|musician|songwriter|music|DJ|composer|group|duo|producer|"
                         r"vocalist|guitarist|artist|ensemble|orchestra)\b", re.I)
LRC_TIME = re.compile(r"^\[\d+:\d+(?:\.\d+)?\]\s*")


def lyrics_from_synced(synced: str) -> str:
    return "\n".join(LRC_TIME.sub("", line) for line in synced.splitlines())


def enrich_lyrics(pl: Playlist, src: Sources, progress: Optional[Callable] = None):
    for i, t in enumerate(pl.tracks):
        if t.lyrics is None:
            hit = src.lrclib.lyrics(t.title, t.artist, t.album,
                                    round(t.duration_ms / 1000) if t.duration_ms else None)
            if hit:
                t.instrumental = bool(hit.get("instrumental"))
                text = hit.get("plainLyrics") or (lyrics_from_synced(hit["syncedLyrics"])
                                                  if hit.get("syncedLyrics") else None)
                t.lyrics = text if text and text.strip() else None
        if progress:
            progress("lyrics", i + 1, len(pl.tracks))


def enrich_deezer(pl: Playlist, src: Sources, progress: Optional[Callable] = None):
    album_genres: Dict[int, list] = {}
    for i, t in enumerate(pl.tracks):
        hit = None
        if t.deezer_id is None:
            hit = src.deezer.find_track(t.title, t.artist, t.isrc)
            if hit:
                t.deezer_id = hit.get("id")
        if t.deezer_id:
            full = src.deezer.track(t.deezer_id) or hit or {}
            t.bpm = float(full["bpm"]) if full.get("bpm") else t.bpm       # Deezer uses 0 for unknown
            t.gain = float(full["gain"]) if full.get("gain") is not None else t.gain
            t.deezer_rank = full.get("rank", t.deezer_rank)
            t.preview_url = t.preview_url or full.get("preview") or None
            album = full.get("album") or {}
            t.cover_url = t.cover_url or album.get("cover_medium")
            t.release_date = t.release_date or full.get("release_date") or album.get("release_date")
            if t.explicit is None and "explicit_lyrics" in full:
                t.explicit = bool(full["explicit_lyrics"])
            if t.duration_ms is None and full.get("duration"):
                t.duration_ms = int(full["duration"]) * 1000
            art = full.get("artist") or {}
            if art.get("id"):
                info = pl.artist_info(t.artist)
                info.deezer_id = info.deezer_id or art["id"]
            if not t.genres and album.get("id"):
                aid = album["id"]
                if aid not in album_genres:
                    a = src.deezer.album(aid) or {}
                    album_genres[aid] = [g["name"] for g in (a.get("genres") or {}).get("data", [])]
                t.genres = list(album_genres[aid])
        if progress:
            progress("deezer", i + 1, len(pl.tracks))


def enrich_artists(pl: Playlist, src: Sources, progress: Optional[Callable] = None, wiki: bool = True):
    names = list(dict.fromkeys(t.artist for t in pl.tracks))
    for i, name in enumerate(names):
        info = pl.artist_info(name)
        mb = src.musicbrainz.artist(name)
        if mb:
            info.mbid = mb.get("id")
            info.country = mb.get("country") or (mb.get("area") or {}).get("iso-3166-1-codes", [None])[0]
            info.kind = mb.get("type")
            begin = (mb.get("life-span") or {}).get("begin") or ""
            info.begin_year = int(begin[:4]) if re.match(r"^\d{4}", begin) else None
            if not info.tags and mb.get("tags"):
                top = max(int(tg.get("count", 0)) for tg in mb["tags"]) or 1
                info.tags = {tg["name"].lower(): max(int(tg.get("count", 0)), 0) / top
                             for tg in mb["tags"] if int(tg.get("count", 0)) > 0}
        if src.lastfm.enabled:
            tags = src.lastfm.top_tags(name)
            if tags:
                info.tags = dict(list(tags.items())[:15])
            info.similar = src.lastfm.similar(name)
        if info.deezer_id is None:
            d = src.deezer.search_artist(name)
            if d and d.get("name", "").lower() == name.lower():
                info.deezer_id = d.get("id")
        if info.deezer_id:
            d = src.deezer.artist(info.deezer_id) or {}
            info.fans = d.get("nb_fan", info.fans)
            info.image_url = info.image_url or d.get("picture_medium")
            if not info.similar:
                rel = src.deezer.related(info.deezer_id)
                # Deezer gives an ordered list without scores: use a linear rank weight.
                info.similar = [(r["name"], 1.0 - k / max(len(rel), 1)) for k, r in enumerate(rel)]
        if wiki and info.bio is None:
            for title in (name, f"{name} (band)", f"{name} (musician)", f"{name} (singer)", f"{name} (rapper)"):
                s = src.wikipedia.summary(title)
                if s and MUSIC_WORDS.search(s.get("extract", "")):
                    info.bio = s.get("extract")
                    info.wiki_url = ((s.get("content_urls") or {}).get("desktop") or {}).get("page")
                    info.image_url = info.image_url or (s.get("thumbnail") or {}).get("source")
                    break
        if progress:
            progress("artists", i + 1, len(names))


def enrich_audio(pl: Playlist, src: Sources):
    ids = [t.spotify_id for t in pl.tracks if t.spotify_id and not t.audio]
    if not ids:
        return
    feats = src.reccobeats.audio_features(ids)
    for t in pl.tracks:
        if t.spotify_id in feats and not t.audio:
            t.audio = feats[t.spotify_id]


def enrich(pl: Playlist, src: Sources, lyrics: bool = True, wiki: bool = True,
           progress: Optional[Callable] = None) -> Playlist:
    """Run all enrichment stages. Stages hit different hosts, so they run concurrently;
    the per-host rate limits in CachedHTTP still apply."""
    jobs = [lambda: enrich_deezer(pl, src, progress), lambda: enrich_audio(pl, src)]
    if lyrics:
        jobs.append(lambda: enrich_lyrics(pl, src, progress))
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for f in [ex.submit(j) for j in jobs]:
            f.result()
    # Artists last: Deezer track lookups give us artist ids for free.
    enrich_artists(pl, src, progress, wiki=wiki)
    return pl


def coverage(pl: Playlist) -> Dict[str, int]:
    """How many tracks / artists each enrichment produced data for."""
    ts = pl.tracks
    arts = [pl.artists.get(n) for n in dict.fromkeys(t.artist for t in ts)]
    arts = [a for a in arts if a]
    return {
        "tracks": len(ts),
        "lyrics": sum(1 for t in ts if t.lyrics),
        "instrumental": sum(1 for t in ts if t.instrumental),
        "bpm": sum(1 for t in ts if t.tempo),
        "audio_features": sum(1 for t in ts if t.audio.get("energy") is not None),
        "release_year": sum(1 for t in ts if t.year),
        "genres_or_tags": sum(1 for t in ts if pl.tags_for(t)),
        "preview": sum(1 for t in ts if t.preview_url),
        "artists": len({t.artist for t in ts}),
        "artist_country": sum(1 for a in arts if a.country),
        "artist_similar": sum(1 for a in arts if a.similar),
        "artist_bio": sum(1 for a in arts if a.bio),
    }
