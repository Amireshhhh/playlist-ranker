"""Recommendations for a playlist.

Two engines:

1. ``collaborative`` - the repo's own trained ranker. Candidates come from track
   co-occurrence in the training playlists (MPD), are scored by the PyTorch model, and are
   re-ranked with the same fatigue/diversity re-ranker the API uses. Only works for tracks
   whose Spotify URIs are in the processed training data.
2. ``similar_artists`` - free-API engine for any playlist. Each playlist artist votes for
   its similar artists (Last.fm match scores, or Deezer related-artist order) with weight
   = its share of the playlist; the winners' Deezer top tracks become candidates. A
   ``discovery`` knob in [0, 1] trades votes against popularity (higher -> less popular).
   The repo's greedy re-ranker then spreads picks across artists.

Plus ``deep_cuts``: popular songs by the playlist's own artists that are not in it yet.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from models.rerank import rerank

from .models import Playlist, norm_key


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _track_row(t: Dict, score: float, reason: str) -> Dict:
    return {
        "title": t.get("title", ""), "artist": (t.get("artist") or {}).get("name", ""),
        "album": (t.get("album") or {}).get("title", ""), "preview": t.get("preview") or None,
        "cover": (t.get("album") or {}).get("cover_small"), "link": t.get("link"),
        "deezer_rank": t.get("rank"), "score": round(float(score), 3), "reason": reason,
    }


def similar_artists(pl: Playlist, src, n: int = 20, discovery: float = 0.3, per_artist: int = 3,
                    max_artists: int = 25, diversity: float = 2.0,
                    max_per_artist: int = 2) -> List[Dict]:
    counts = Counter(t.artist for t in pl.tracks)
    total = sum(counts.values())
    in_playlist = {a.lower() for a in counts}
    votes: Dict[str, float] = defaultdict(float)
    voters: Dict[str, Counter] = defaultdict(Counter)
    for artist, c in counts.items():
        info = pl.artists.get(artist)
        for other, match in (info.similar if info else []):
            if other.lower() in in_playlist:
                continue
            votes[other] += (c / total) * match
            voters[other][artist] += (c / total) * match
    if not votes:
        return []
    have = {t.key for t in pl.tracks}
    cands, raw_vote, pop, artists = [], [], [], []
    for other, v in sorted(votes.items(), key=lambda kv: -kv[1])[:max_artists]:
        d = src.deezer.search_artist(other)
        if not d or norm_key(d.get("name", "")) != norm_key(other):
            continue
        for t in src.deezer.top_tracks(d["id"], limit=per_artist):
            key = f"{norm_key(other)}|{norm_key(t.get('title', ''))}"
            if key in have:
                continue
            have.add(key)
            because = ", ".join(a for a, _ in voters[other].most_common(2))
            cands.append((t, f"similar to {because}"))
            raw_vote.append(v)
            pop.append(math.log10(t["rank"]) if t.get("rank") else 0.0)
            artists.append(other)
    if not cands:
        return []
    score = _z(np.array(raw_vote)) - discovery * _z(np.array(pop))
    order = rerank(score.tolist(), artists, set(), len(cands), diversity=diversity)
    picked, per = [], Counter()
    for i in order:  # hard cap on top of the soft diversity penalty
        if per[artists[i]] < max_per_artist:
            picked.append(i)
            per[artists[i]] += 1
        if len(picked) == n:
            break
    return [_track_row(cands[i][0], score[i], cands[i][1]) for i in picked]


def deep_cuts(pl: Playlist, src, n: int = 10, top_artists: int = 8) -> List[Dict]:
    counts = Counter(t.artist for t in pl.tracks)
    have = {t.key for t in pl.tracks}
    rows = []
    for artist, c in counts.most_common(top_artists):
        info = pl.artists.get(artist)
        aid = info.deezer_id if info else None
        if not aid:
            d = src.deezer.search_artist(artist)
            aid = d["id"] if d and norm_key(d.get("name", "")) == norm_key(artist) else None
        if not aid:
            continue
        for t in src.deezer.top_tracks(aid, limit=10):
            key = f"{norm_key(artist)}|{norm_key(t.get('title', ''))}"
            if key not in have:
                have.add(key)
                rows.append(_track_row(t, c, f"you have {c} songs by {artist}"))
                break
    return rows[:n]


def collaborative(pl: Playlist, data_dir: str, checkpoint: str, n: int = 20,
                  fatigue: float = 0.5, diversity: float = 0.5) -> Optional[Dict]:
    """Rank with the repo's trained model. Returns None if data or checkpoint are missing or
    no playlist track is in the training catalog."""
    data_dir, checkpoint = Path(data_dir), Path(checkpoint)
    if not (data_dir / "track_to_artist.json").exists() or not checkpoint.exists():
        return None
    import torch

    from features.build_features import FeatureBuilder
    from models.ranker import RankingModel

    fb = FeatureBuilder(data_dir=str(data_dir))
    uris = [t.spotify_uri for t in pl.tracks if t.spotify_uri in fb.track_to_artist]
    if not uris:
        return None
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ck.get("config", {})
    model = RankingModel(input_dim=cfg.get("input_dim", 32), hidden_dims=cfg.get("hidden_dims", [128, 64, 32]),
                         dropout=cfg.get("dropout", 0.3))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    prefix = [{"track_uri": u, "artist_uri": fb.track_to_artist.get(u, "unknown"), "pos": i}
              for i, u in enumerate(uris)]
    cands = fb.candidates_from_cooccurrence(uris, top_n=100)
    feats = np.stack([fb.extract_features(prefix, c) for c in cands])
    with torch.no_grad():
        scores = model(torch.from_numpy(feats)).squeeze(-1).tolist()
    artists = [fb.track_to_artist.get(c, "unknown") for c in cands]
    order = rerank(scores, artists, {p["artist_uri"] for p in prefix[-5:]}, min(n, len(cands)),
                   fatigue=fatigue, diversity=diversity)
    names = {}
    for t in pl.tracks:
        if t.artist_uri:
            names[t.artist_uri] = t.artist
    return {
        "matched_tracks": len(uris),
        "rows": [{"track_uri": cands[i], "artist_uri": artists[i], "artist": names.get(artists[i], artists[i]),
                  "score": round(float(scores[i]), 3),
                  "link": ("https://open.spotify.com/track/" + cands[i].rsplit(":", 1)[1])
                  if cands[i].startswith("spotify:track:") and "_" not in cands[i] else None}
                 for i in order],
    }
