"""Get a playlist from a link or an export file.

| Input | How | Needs |
|---|---|---|
| Deezer playlist link | public API | nothing |
| YouTube / YouTube Music playlist link | YouTube Data API v3 playlistItems (1 quota unit per 50 tracks) | free API key, ``YOUTUBE_API_KEY`` |
| Spotify playlist link | Web API ``/playlists/{id}/items`` with the *visitor's* login | ``SPOTIFY_CLIENT_ID``; since Feb 2026 only playlists the logged-in user owns or collaborates on return tracks |
| Apple Music | export from the Music app: File > Library > Export Playlist (.txt or .xml) | nothing |
| CSV / JSON | Exportify, or any file with title and artist columns | nothing |

Only fixed API hosts are ever contacted; a pasted link is parsed for an id, never fetched.
"""

from __future__ import annotations

import csv
import io
import plistlib
import re
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .models import Playlist, Track


class InputError(ValueError):
    """A problem with what the user gave us, with a message they can act on."""


# ----------------------------------------------------------------------------- links

def parse_link(url: str) -> Tuple[str, str]:
    """Return (service, playlist id) for a supported playlist link, else raise InputError."""
    url = url.strip()
    m = re.fullmatch(r"spotify:playlist:([A-Za-z0-9]{10,40})", url)
    if m:
        return "spotify", m.group(1)
    u = urlparse(url if "://" in url else "https://" + url)
    host = (u.hostname or "").lower()
    path = u.path
    if host in ("open.spotify.com", "play.spotify.com"):
        m = re.search(r"/playlist/([A-Za-z0-9]{10,40})", path)
        if m:
            return "spotify", m.group(1)
    if host.endswith("deezer.com") or host == "deezer.page.link":
        m = re.search(r"/playlist/(\d+)", path)
        if m:
            return "deezer", m.group(1)
        if host == "deezer.page.link" or host == "link.deezer.com":
            raise InputError("Short Deezer share links can't be read; open the link and paste the full deezer.com/playlist/... address.")
    if host in ("www.youtube.com", "youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"):
        lid = parse_qs(u.query).get("list", [None])[0]
        if lid and re.fullmatch(r"[A-Za-z0-9_-]{10,64}", lid):
            return "youtube", lid
    if host == "music.apple.com":
        raise InputError("Apple Music links can't be read without a paid Apple developer account. In the Music app choose "
                         "File > Library > Export Playlist and upload the .txt or .xml file instead.")
    raise InputError("Paste a playlist link from Spotify, Deezer or YouTube / YouTube Music, or upload a file.")


# ----------------------------------------------------------------------------- YouTube

TOPIC = re.compile(r"\s+-\s+Topic$")
NOISE = re.compile(r"\s*[\(\[](official|lyric|lyrics|audio|video|visualizer|hd|4k|remaster|mv)[^\)\]]*[\)\]]", re.I)


def youtube_track(title: str, channel: str) -> Optional[Tuple[str, str]]:
    """(song, artist) from a YouTube video title and its channel.

    YouTube Music's auto-generated uploads sit on '<Artist> - Topic' channels and carry the
    bare song title; other uploads are usually 'Artist - Song (Official Video)'.
    """
    if not title or title in ("Private video", "Deleted video"):
        return None
    if TOPIC.search(channel or ""):
        return title.strip(), TOPIC.sub("", channel).strip()
    clean = NOISE.sub("", title).strip()
    for sep in (" - ", " – ", " — ", " | "):
        if sep in clean:
            a, s = clean.split(sep, 1)
            return s.strip().strip('"'), a.strip()
    channel = re.sub(r"(VEVO|Official)$", "", channel or "", flags=re.I).strip()
    return (clean, channel) if channel else None


def load_youtube(http, playlist_id: str, api_key: Optional[str], max_tracks: int = 500) -> Playlist:
    if not api_key:
        raise InputError("YouTube playlists need a free YouTube Data API key on the server (YOUTUBE_API_KEY).")
    base = "https://www.googleapis.com/youtube/v3"
    meta = http.get_json(f"{base}/playlists", {"part": "snippet", "id": playlist_id, "key": api_key}) or {}
    items = meta.get("items") or []
    if not items:
        raise InputError("That YouTube playlist is private, deleted, or the link is wrong.")
    name = items[0]["snippet"].get("title", "YouTube playlist")
    owner = items[0]["snippet"].get("channelTitle", "me")
    tracks, token = [], None
    while len(tracks) < max_tracks:
        page = http.get_json(f"{base}/playlistItems", {"part": "snippet", "playlistId": playlist_id, "maxResults": 50,
                                                       "pageToken": token, "key": api_key}) or {}
        for it in page.get("items") or []:
            sn = it.get("snippet") or {}
            got = youtube_track(sn.get("title", ""), sn.get("videoOwnerChannelTitle", ""))
            if got:
                tracks.append(Track(title=got[0], artist=got[1], added_at=sn.get("publishedAt")))
        token = page.get("nextPageToken")
        if not token:
            break
    if not tracks:
        raise InputError("No playable tracks found in that YouTube playlist.")
    return Playlist(name=name, tracks=tracks[:max_tracks], owner=owner, source=f"youtube:{playlist_id}")


# ----------------------------------------------------------------------------- Spotify

def load_spotify(http_get, playlist_id: str, token: Optional[str], max_tracks: int = 500) -> Playlist:
    """``http_get(url, params, headers)`` returns (status, json). ``token`` is the visitor's user token."""
    if not token:
        raise InputError("Log in with Spotify first. Spotify only shares the tracks of playlists you own or collaborate on.")
    base = "https://api.spotify.com/v1"
    status, meta = http_get(f"{base}/playlists/{playlist_id}", {"fields": "name,owner(display_name)"}, token)
    if status == 404:
        raise InputError("Spotify can't find that playlist.")
    if status in (401, 403):
        raise InputError("Spotify refused access. You can analyse playlists you own or collaborate on (a Spotify rule since February 2026).")
    if status != 200 or not meta:
        raise InputError(f"Spotify returned an error ({status}). Try again, or upload an Exportify CSV.")
    tracks, offset = [], 0
    while len(tracks) < max_tracks:
        status, page = http_get(f"{base}/playlists/{playlist_id}/items", {"limit": 50, "offset": offset}, token)
        if status in (401, 403):
            raise InputError("Spotify returned the playlist but not its tracks: it only shares tracks of playlists you own or "
                             "collaborate on. Upload an Exportify CSV instead.")
        if status != 200 or not page:
            break
        rows = page.get("items") or []
        for row in rows:
            t = row.get("item") or row.get("track") or {}   # 'track' was renamed 'item' in Feb 2026
            if not t or t.get("type", "track") != "track" or not t.get("name"):
                continue
            artists = [a.get("name", "") for a in t.get("artists") or [] if a.get("name")]
            if not artists:
                continue
            alb = t.get("album") or {}
            tracks.append(Track(title=t["name"], artist=artists[0], artists=artists, album=alb.get("name", ""),
                                duration_ms=t.get("duration_ms"), spotify_uri=t.get("uri"),
                                isrc=(t.get("external_ids") or {}).get("isrc"), release_date=alb.get("release_date"),
                                explicit=t.get("explicit"), added_at=row.get("added_at")))
        if len(rows) < 50:
            break
        offset += 50
    if not tracks:
        raise InputError("Spotify returned no tracks for that playlist (it may be empty, or not yours).")
    return Playlist(name=meta.get("name", "Spotify playlist"), tracks=tracks[:max_tracks],
                    owner=(meta.get("owner") or {}).get("display_name", "me"), source=f"spotify:{playlist_id}")


# ----------------------------------------------------------------------------- files

def _decode(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    if data[:3] == b"\xef\xbb\xbf":
        return data[3:].decode("utf-8")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def load_apple_xml(data: bytes, max_tracks: int = 2000) -> Playlist:
    """Music app / iTunes XML export (a property list with 'Tracks' and 'Playlists')."""
    try:
        pl = plistlib.loads(data)
    except Exception as e:  # noqa: BLE001 - any parse failure is the user's file
        raise InputError(f"That XML file isn't a Music app export ({type(e).__name__}).")
    tracks_by_id: Dict[str, Dict] = {str(k): v for k, v in (pl.get("Tracks") or {}).items()}
    lists = pl.get("Playlists") or []
    order = [str(i.get("Track ID")) for i in (lists[0].get("Playlist Items") or [])] if lists else list(tracks_by_id)
    name = lists[0].get("Name", "Apple Music playlist") if lists else "Apple Music library"
    tracks = []
    for tid in order[:max_tracks]:
        t = tracks_by_id.get(tid)
        if t and t.get("Name") and t.get("Artist"):
            tracks.append(Track(title=t["Name"], artist=t["Artist"], album=t.get("Album", ""),
                                duration_ms=t.get("Total Time"), release_date=str(t["Year"]) if t.get("Year") else None,
                                added_at=t["Date Added"].isoformat() if hasattr(t.get("Date Added"), "isoformat") else None,
                                genres=[t["Genre"]] if t.get("Genre") else []))
    if not tracks:
        raise InputError("No tracks with a name and an artist in that XML file.")
    return Playlist(name=name, tracks=tracks, source="apple-music-xml")


def load_table(data: bytes, name: str, max_tracks: int = 2000) -> Playlist:
    """CSV or tab-separated text (Exportify, the Music app's 'Unicode Text' export, anything with title + artist)."""
    from .ingest import load_csv
    text = _decode(data)
    first = text.splitlines()[0] if text else ""
    delim = "\t" if first.count("\t") > first.count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    if not rows:
        raise InputError("That file is empty.")
    # normalise to a comma CSV in memory and reuse the CSV loader
    buf = io.StringIO()
    w = csv.writer(buf)
    for r in rows[: max_tracks + 1]:
        w.writerow(r)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "upload.csv"
        p.write_text(buf.getvalue(), encoding="utf-8")
        try:
            pl = load_csv(str(p), name=name)
        except ValueError as e:
            raise InputError(str(e).replace(str(p) + ": ", ""))
    # Music app text export: 'Time' is seconds
    header = [h.strip().lower() for h in rows[0]]
    if "time" in header and not any(t.duration_ms for t in pl.tracks):
        ti = header.index("time")
        for t, r in zip(pl.tracks, rows[1:]):
            try:
                t.duration_ms = int(float(r[ti]) * 1000)
            except (ValueError, IndexError):
                pass
    pl.source = "file"
    return pl


def load_upload(filename: str, data: bytes) -> Playlist:
    name = re.sub(r"\.[A-Za-z0-9]+$", "", filename or "playlist").replace("_", " ")[:80] or "playlist"
    low = (filename or "").lower()
    if low.endswith(".xml") or data.lstrip()[:5] == b"<?xml":
        return load_apple_xml(data)
    if low.endswith(".json"):
        import json
        import tempfile
        from pathlib import Path

        from .ingest import load_json
        try:
            json.loads(_decode(data))
        except ValueError:
            raise InputError("That JSON file can't be parsed.")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "upload.json"
            p.write_bytes(data)
            try:
                pl = load_json(str(p), name=name)
            except ValueError as e:
                raise InputError(str(e).replace(str(p) + ": ", ""))
        pl.source = "file"
        return pl
    return load_table(data, name)
