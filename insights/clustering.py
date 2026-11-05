"""Group songs and artists.

Each song gets a vector made of up to three blocks:
  tags   - genre/tag profile (Exportify genres, Last.fm or MusicBrainz tags, Deezer album genres)
  lyrics - TF-IDF of the lyrics reduced to <= 20 dimensions with truncated SVD
  sound  - standardised tempo, energy, valence, danceability, acousticness, gain, release year
Every block is L2-normalised per song and weighted, so no block dominates just by having
more dimensions. Missing blocks are zero vectors. k-means is run for k = 2..8 and the k with
the best silhouette score is kept.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Playlist

SOUND_KEYS = ("tempo", "energy", "valence", "danceability", "acousticness", "gain", "year")


def _l2(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return np.divide(M, n, out=np.zeros_like(M), where=n > 0)


def tag_block(pl: Playlist) -> Tuple[np.ndarray, List[str]]:
    profiles = [pl.tags_for(t) for t in pl.tracks]
    df = Counter(tag for p in profiles for tag in p)
    vocab = [t for t, c in df.most_common(200)]
    if not vocab:
        return np.zeros((len(pl.tracks), 0)), []
    idx = {t: j for j, t in enumerate(vocab)}
    n = len(profiles)
    M = np.zeros((n, len(vocab)))
    for i, p in enumerate(profiles):
        for tag, w in p.items():
            if tag in idx:
                M[i, idx[tag]] = w * np.log((1 + n) / (1 + df[tag]) + 1)  # smoothed idf
    return _l2(M), vocab


def sound_values(t) -> Dict[str, Optional[float]]:
    return {
        "tempo": t.tempo, "energy": t.audio.get("energy"), "valence": t.audio.get("valence"),
        "danceability": t.audio.get("danceability"), "acousticness": t.audio.get("acousticness"),
        "gain": t.gain, "year": float(t.year) if t.year else None,
    }


def sound_block(pl: Playlist) -> np.ndarray:
    rows = [sound_values(t) for t in pl.tracks]
    M = np.full((len(rows), len(SOUND_KEYS)), np.nan)
    for i, r in enumerate(rows):
        for j, k in enumerate(SOUND_KEYS):
            if r[k] is not None:
                M[i, j] = r[k]
    keep = [j for j in range(M.shape[1]) if np.isfinite(M[:, j]).sum() >= 2]
    M = M[:, keep]
    if M.shape[1] == 0:
        return np.zeros((len(rows), 0))
    mu, sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
    sd[sd == 0] = 1
    Z = (M - mu) / sd
    return np.nan_to_num(Z, nan=0.0)  # missing = playlist average


def lyrics_block(pl: Playlist, report) -> np.ndarray:
    n = len(pl.tracks)
    if report is None or report.tfidf_matrix is None or report.tfidf_matrix.shape[0] < 3:
        return np.zeros((n, 0))
    X = report.tfidf_matrix
    k = min(20, X.shape[0] - 1, X.shape[1] - 1)
    if k < 1:
        return np.zeros((n, 0))
    from sklearn.decomposition import TruncatedSVD
    R = TruncatedSVD(n_components=k, random_state=0).fit_transform(X)
    out = np.zeros((n, k))
    for s, row in zip(report.songs, R):
        out[s.index] = row
    return _l2(out)


@dataclass
class ClusterResult:
    labels: np.ndarray            # cluster id per track (-1 if not clustered)
    coords: np.ndarray            # 2-D PCA coordinates per track
    k: int
    silhouette: Optional[float]
    blocks_used: List[str]
    summaries: List[Dict]
    explained_variance: List[float]


def song_vectors(pl: Playlist, lyrics_report=None, weights=None) -> Tuple[np.ndarray, List[str], List[str]]:
    weights = weights or {"tags": 1.0, "lyrics": 0.7, "sound": 0.8}
    T, vocab = tag_block(pl)
    blocks, used = [], []
    for name, B in (("tags", T), ("lyrics", lyrics_block(pl, lyrics_report)), ("sound", _l2(sound_block(pl)))):
        if B.shape[1] and np.any(B):
            blocks.append(B * weights[name])
            used.append(name)
    X = np.hstack(blocks) if blocks else np.zeros((len(pl.tracks), 0))
    return X, used, vocab


def cluster_songs(pl: Playlist, lyrics_report=None, k_range=range(2, 9), seed: int = 0) -> Optional[ClusterResult]:
    X, used, vocab = song_vectors(pl, lyrics_report)
    n = X.shape[0]
    if n < 4 or X.shape[1] == 0:
        return None
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    rows = np.any(X != 0, axis=1)
    Xr = X[rows]
    best = None
    # At least ~4 songs per group on average: silhouette alone picked k = 8 for 24 real
    # songs, leaving several 2-song groups.
    k_max = max(2, len(Xr) // 4)
    for k in k_range:
        if k >= len(Xr) or k > k_max:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xr)
        if len(set(km.labels_)) < 2:
            continue
        s = float(silhouette_score(Xr, km.labels_))
        if best is None or s > best[0]:
            best = (s, k, km.labels_)
    labels = np.full(n, -1)
    sil, k = None, 1
    if best:
        sil, k, lab = best
        labels[rows] = lab
    else:
        labels[rows] = 0
    nc = min(2, X.shape[1], n)
    pca = PCA(n_components=nc, random_state=seed).fit(X)
    coords = pca.transform(X)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((n, 1))])

    T, _ = tag_block(pl)
    summaries = []
    for c in sorted(set(labels) - {-1}):
        members = [i for i in range(n) if labels[i] == c]
        tag_mass = T[members].sum(axis=0) if T.shape[1] else np.zeros(0)
        tags = [vocab[j] for j in np.argsort(-tag_mass)[:3] if tag_mass[j] > 0] if len(tag_mass) else []
        words = []
        if lyrics_report is not None:
            wc = Counter()
            for s in lyrics_report.songs:
                if s.index in members:
                    wc.update(dict(s.distinctive))
            words = [w for w, _ in wc.most_common(4)]
        sv = [sound_values(pl.tracks[i]) for i in members]
        tempo = [v["tempo"] for v in sv if v["tempo"]]
        years = [v["year"] for v in sv if v["year"]]
        arts = Counter(pl.tracks[i].artist for i in members)
        name = " / ".join(tags[:2]) if tags else (", ".join(words[:2]) if words else f"Group {c + 1}")
        summaries.append({
            "cluster": int(c), "name": name, "size": len(members), "tags": tags, "words": words,
            "top_artists": [a for a, _ in arts.most_common(3)],
            "median_bpm": float(np.median(tempo)) if tempo else None,
            "median_year": int(np.median(years)) if years else None,
            "members": members,
        })
    return ClusterResult(labels=labels, coords=coords, k=k, silhouette=sil, blocks_used=used,
                         summaries=summaries, explained_variance=[float(v) for v in pca.explained_variance_ratio_])


@dataclass
class ArtistGraph:
    names: List[str]
    coords: np.ndarray
    edges: List[Tuple[int, int, float]]   # (i, j, cosine similarity)
    counts: List[int]


def artist_knn(pl: Playlist, lyrics_report=None, k: int = 3, min_sim: float = 0.05) -> Optional[ArtistGraph]:
    """k-nearest-neighbour graph of the playlist's artists by cosine similarity of their
    mean song vectors. Known similarity from Last.fm/Deezer ('similar artists') between two
    playlist artists is added as an extra edge source."""
    X, used, _ = song_vectors(pl, lyrics_report)
    names = list(dict.fromkeys(t.artist for t in pl.tracks))
    if len(names) < 3:
        return None
    idx = {a: i for i, a in enumerate(names)}
    counts = Counter(t.artist for t in pl.tracks)
    A = np.zeros((len(names), X.shape[1]))
    for i, t in enumerate(pl.tracks):
        A[idx[t.artist]] += X[i]
    A = _l2(A)
    S = A @ A.T
    np.fill_diagonal(S, -1)
    edges = {}
    if X.shape[1]:
        for i in range(len(names)):
            for j in np.argsort(-S[i])[:k]:
                if S[i, j] >= min_sim:
                    a, b = sorted((i, int(j)))
                    edges[(a, b)] = float(S[i, j])
    for a in names:
        info = pl.artists.get(a)
        for other, match in (info.similar if info else []):
            if other in idx and other != a:
                key = tuple(sorted((idx[a], idx[other])))
                edges[key] = max(edges.get(key, 0.0), float(match))
    coords = _spring_layout(len(names), edges)
    return ArtistGraph(names=names, coords=coords, edges=[(a, b, w) for (a, b), w in edges.items()],
                       counts=[counts[a] for a in names])


def _spring_layout(n: int, edges: Dict, iters: int = 300, seed: int = 0) -> np.ndarray:
    """Fruchterman-Reingold force layout (small graphs only: O(n^2) per iteration)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    kk = 1.0 / np.sqrt(max(n, 1))
    temp = 0.1
    for _ in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(d, axis=2) + 1e-9
        disp = (d / dist[..., None] * (kk * kk / dist)[..., None]).sum(axis=1)
        for (a, b), w in edges.items():
            delta = pos[a] - pos[b]
            dd = np.linalg.norm(delta) + 1e-9
            f = delta / dd * (dd * dd / kk) * w
            disp[a] -= f
            disp[b] += f
        disp -= pos * (2.0 * kk)  # gravity: keeps disconnected components on screen
        length = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-9
        pos += disp / length * np.minimum(length, temp)
        pos -= pos.mean(axis=0)  # keep disconnected nodes from drifting off
        temp *= 0.985
    return pos
