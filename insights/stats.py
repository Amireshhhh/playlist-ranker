"""Descriptive statistics of a playlist, reusing the repo's playlist-health metrics."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from evaluation.metrics import PlaylistHealthMetrics as PHM

from .models import Playlist


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    if s.isdigit():  # Deezer time_add: unix seconds
        return datetime.fromtimestamp(int(s), tz=timezone.utc)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def lorenz(counts: List[int]) -> List[float]:
    """Cumulative share of tracks held by the least-played artists (0 ... 1)."""
    c = np.sort(np.asarray(counts, dtype=float))
    return [0.0] + list(np.cumsum(c) / c.sum()) if c.sum() else [0.0]


def basic(pl: Playlist) -> Dict:
    ts = pl.tracks
    artists = [t.artist for t in ts]
    ac = Counter(artists)
    dur = [t.duration_ms for t in ts if t.duration_ms]
    years = [t.year for t in ts if t.year]
    explicit = [t.explicit for t in ts if t.explicit is not None]
    out = {
        "n_tracks": len(ts),
        "n_unique_tracks": len({t.key for t in ts}),
        "duplicates": [k for k, c in Counter(t.key for t in ts).items() if c > 1],
        "n_artists": len(ac),
        "n_albums": len({(t.artist, t.album) for t in ts if t.album}),
        "total_ms": int(sum(dur)) if dur else None,
        "duration_known": len(dur),
        "mean_track_ms": float(np.mean(dur)) if dur else None,
        "longest": max(ts, key=lambda t: t.duration_ms or 0) if dur else None,
        "shortest": min((t for t in ts if t.duration_ms), key=lambda t: t.duration_ms) if dur else None,
        "top_artists": ac.most_common(15),
        "artist_counts": dict(ac),
        "artist_entropy_bits": PHM.artist_entropy(artists),
        "max_entropy_bits": math.log2(len(ts)) if ts else 0.0,
        "artist_coverage": PHM.artist_coverage(artists),
        "repeat_rate": PHM.repeat_rate(artists),
        "gini": float(PHM.gini_coefficient(list(ac.values()))),
        "lorenz": lorenz(list(ac.values())),
        "one_hit_artists": sum(1 for c in ac.values() if c == 1),
        "years": years,
        "median_year": int(np.median(years)) if years else None,
        "decades": dict(sorted(Counter(y // 10 * 10 for y in years).items())),
        "oldest": min((t for t in ts if t.year), key=lambda t: t.year) if years else None,
        "newest": max((t for t in ts if t.year), key=lambda t: t.year) if years else None,
        "explicit_share": (sum(explicit) / len(explicit)) if explicit else None,
    }
    # When the user added songs (Exportify 'Added At' / Deezer 'time_add').
    added = [d for d in (_parse_dt(t.added_at) for t in ts) if d]
    if added:
        out["added"] = sorted(added)
        out["added_by_weekday_hour"] = Counter((d.weekday(), d.hour) for d in added)
        out["added_span_days"] = (max(added) - min(added)).days
    ranks = [t.deezer_rank for t in ts if t.deezer_rank]
    pops = [t.popularity for t in ts if t.popularity is not None]
    if ranks:
        out["deezer_rank_median"] = float(np.median(ranks))
    if pops:
        out["popularity_mean"] = float(np.mean(pops))
    return out


def mainstream_score(pl: Playlist) -> Optional[Dict]:
    """Where the playlist sits between deep cuts and hits.

    Uses Spotify popularity (0-100, Exportify) when present, else Deezer rank divided by
    1,000,000 and clipped to [0, 1]. (A log scale, log10(rank) / 6, was tried first: on a real 24-song playlist
    it averaged 0.97, too close to the ceiling to tell playlists apart.)
    This is a relative score for comparing playlists processed the same way, not a calibrated one.
    """
    scored = [(t.popularity / 100, t) for t in pl.tracks if t.popularity is not None]
    src = "Spotify popularity"
    if not scored:
        scored = [(min(t.deezer_rank / 1_000_000, 1.0), t) for t in pl.tracks
                  if t.deezer_rank and t.deezer_rank > 0]
        src = "Deezer rank / 1,000,000"
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return {"score": float(np.mean([p for p, _ in scored])), "source": src, "n": len(scored),
            "deepest_cut": scored[0][1], "biggest_hit": scored[-1][1]}


def flow(pl: Playlist) -> Optional[Dict]:
    """How smoothly tempo moves from one song to the next, and a greedy smoother order.

    Tempo distance is min(|a-b|, |2a-b|, |a-2b|): a DJ can mix 70 BPM into 140 BPM
    (half/double time), so that pair counts as close. The suggested order is a nearest-
    neighbour tour started from each song in turn, keeping the tour with the lowest total.
    """
    idx = [i for i, t in enumerate(pl.tracks) if t.tempo]
    if len(idx) < 3:
        return None
    bpm = np.array([pl.tracks[i].tempo for i in idx], dtype=float)

    def dist(a, b):
        return min(abs(a - b), abs(2 * a - b), abs(a - 2 * b))

    D = np.array([[dist(a, b) for b in bpm] for a in bpm])
    current = float(np.mean([D[k, k + 1] for k in range(len(idx) - 1)]))
    best_tour, best_cost = None, math.inf
    for start in range(len(idx)):
        tour, left = [start], set(range(len(idx))) - {start}
        while left:
            nxt = min(left, key=lambda j: (D[tour[-1], j], j))
            tour.append(nxt)
            left.remove(nxt)
        cost = sum(D[a, b] for a, b in zip(tour, tour[1:]))
        if cost < best_cost:
            best_tour, best_cost = tour, cost
    return {
        "track_indices": idx, "bpm": bpm.tolist(),
        "mean_jump_current": current,
        "mean_jump_suggested": float(best_cost / (len(idx) - 1)),
        "suggested_order": [idx[k] for k in best_tour],
        "biggest_jump": max(((D[k, k + 1], idx[k], idx[k + 1]) for k in range(len(idx) - 1))),
    }


def mood_quadrants(pl: Playlist, lyrics_report=None) -> Optional[Dict]:
    """Songs on an energy (arousal) x positivity (valence) plane (Russell's circumplex).

    Uses audio-feature energy/valence when present (Exportify or ReccoBeats). Otherwise
    falls back to tempo (normalised within the playlist) x lyric sentiment, and says so.
    """
    pts = []
    sent = {s.index: s.sentiment["compound"] for s in (lyrics_report.songs if lyrics_report else [])}
    has_audio = sum(1 for t in pl.tracks if "energy" in t.audio and "valence" in t.audio)
    if has_audio >= 3:
        for i, t in enumerate(pl.tracks):
            if "energy" in t.audio and "valence" in t.audio:
                pts.append((i, t.audio["valence"], t.audio["energy"]))
        basis = ("valence (audio)", "energy (audio)")
    else:
        tempos = [t.tempo for t in pl.tracks if t.tempo]
        if len(tempos) < 3 or len(sent) < 3:
            return None
        lo, hi = min(tempos), max(tempos)
        for i, t in enumerate(pl.tracks):
            if t.tempo and i in sent:
                pts.append((i, (sent[i] + 1) / 2, (t.tempo - lo) / (hi - lo) if hi > lo else 0.5))
        basis = ("lyric sentiment (VADER, rescaled to 0-1)", "tempo (min-max within playlist)")
    if len(pts) < 3:
        return None
    quad = Counter()
    for _, x, y in pts:
        quad[("happy / energetic" if x >= 0.5 else "angry / tense") if y >= 0.5 else
             ("calm / content" if x >= 0.5 else "sad / melancholic")] += 1
    return {"points": pts, "basis": basis, "quadrants": dict(quad)}


def overlap(playlists: List[Playlist]) -> Dict:
    """Jaccard overlap of tracks and of artists between every pair of a user's playlists."""
    tk = [{t.key for t in p.tracks} for p in playlists]
    ak = [{t.artist.lower() for t in p.tracks} for p in playlists]
    n = len(playlists)

    def jac(a, b):
        return len(a & b) / len(a | b) if a | b else 0.0

    return {
        "names": [p.name for p in playlists],
        "tracks": [[jac(tk[i], tk[j]) for j in range(n)] for i in range(n)],
        "artists": [[jac(ak[i], ak[j]) for j in range(n)] for i in range(n)],
    }
