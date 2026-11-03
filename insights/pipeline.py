"""Glue: playlist -> enrichment -> analysis dict -> dashboards."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from . import clustering, dashboard, enrich, lyrics, recommend, stats
from .models import Playlist

logger = logging.getLogger(__name__)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "playlist"


def analyze(pl: Playlist, src=None, recs: bool = True, discovery: float = 0.3,
            data_dir: Optional[str] = None, checkpoint: Optional[str] = None) -> Dict:
    lr = lyrics.analyze(pl.tracks)
    out = {
        "stats": stats.basic(pl),
        "lyrics": lr,
        "clusters": clustering.cluster_songs(pl, lr),
        "artist_graph": clustering.artist_knn(pl, lr),
        "mood": stats.mood_quadrants(pl, lr),
        "flow": stats.flow(pl),
        "mainstream": stats.mainstream_score(pl),
        "coverage": enrich.coverage(pl),
        "recommendations": {"discovery": discovery},
    }
    if recs and src is not None:
        out["recommendations"]["similar"] = recommend.similar_artists(pl, src, discovery=discovery)
        out["recommendations"]["deep_cuts"] = recommend.deep_cuts(pl, src)
    if data_dir and checkpoint:
        out["recommendations"]["collaborative"] = recommend.collaborative(pl, data_dir, checkpoint)
    return out


def summary_json(pl: Playlist, a: Dict) -> Dict:
    """Machine-readable digest (numbers only, no lyrics text) saved next to each dashboard."""
    st = a["stats"]
    lr = a["lyrics"]
    return {
        "name": pl.name, "owner": pl.owner, "source": pl.source, "coverage": a["coverage"],
        "stats": {k: v for k, v in st.items() if k in (
            "n_tracks", "n_unique_tracks", "n_artists", "n_albums", "total_ms", "artist_entropy_bits",
            "max_entropy_bits", "artist_coverage", "repeat_rate", "gini", "median_year", "decades",
            "explicit_share", "top_artists", "one_hit_artists")},
        "lyrics": None if lr is None else {
            "songs": len(lr.songs), "total_words": lr.total_words, "vocabulary": lr.vocabulary,
            "top_words": lr.word_counts.most_common(50), "pronouns": lr.pronouns, "themes": lr.themes,
            "per_song": [{"title": s.title, "artist": s.artist, "words": s.n_words, "mattr": s.mattr,
                          "repetitiveness": s.repetitiveness, "sentiment": s.sentiment["compound"],
                          "distinctive": [w for w, _ in s.distinctive]} for s in lr.songs]},
        "clusters": None if a["clusters"] is None else {
            "k": a["clusters"].k, "silhouette": a["clusters"].silhouette, "features": a["clusters"].blocks_used,
            "groups": [{k: v for k, v in s.items() if k != "members"} |
                       {"songs": [pl.tracks[i].title for i in s["members"]]} for s in a["clusters"].summaries]},
        "mood": None if a["mood"] is None else {"basis": a["mood"]["basis"], "quadrants": a["mood"]["quadrants"]},
        "flow": None if a["flow"] is None else {
            "mean_jump_current": a["flow"]["mean_jump_current"], "mean_jump_suggested": a["flow"]["mean_jump_suggested"],
            "suggested_order": [pl.tracks[i].title for i in a["flow"]["suggested_order"]]},
        "mainstream": None if a["mainstream"] is None else {"score": a["mainstream"]["score"], "source": a["mainstream"]["source"]},
        "recommendations": a["recommendations"],
    }


def build_user(user: str, playlists: List[Playlist], out_dir: str, src=None, enrich_data: bool = True,
               inline_js: bool = False, **analyze_kw) -> Path:
    """Write <out_dir>/<user>/index.html and one dashboard per playlist."""
    root = Path(out_dir) / slug(user)
    entries, analyses = [], []
    for pl in playlists:
        if enrich_data and src is not None:
            enrich.enrich(pl, src, progress=_progress)
        a = analyze(pl, src, **analyze_kw)
        analyses.append(a)
        sub = root / slug(pl.name)
        dashboard.render_playlist(pl, a, sub / "index.html", inline_js=inline_js, user_index_href="../index.html")
        (sub / "summary.json").write_text(json.dumps(summary_json(pl, a), indent=2, default=dashboard._json_default),
                                          encoding="utf-8")
        entries.append({"name": pl.name, "href": f"{slug(pl.name)}/index.html", "stats": a["stats"]})
        logger.info("wrote %s", sub / "index.html")
    all_tracks = [t for p in playlists for t in p.tracks]
    wc = Counter()
    for a in analyses:
        if a["lyrics"]:
            wc.update(a["lyrics"].word_counts)
    combined = {
        "n_tracks": len(all_tracks), "n_unique_tracks": len({t.key for t in all_tracks}),
        "n_artists": len({t.artist for t in all_tracks}),
        "total_ms": sum(t.duration_ms or 0 for t in all_tracks) or None,
        "top_artists": Counter(t.artist for t in all_tracks).most_common(15), "word_counts": wc,
    }
    ov = stats.overlap(playlists) if len(playlists) > 1 else None
    return dashboard.render_user_index(user, entries, ov, combined, root / "index.html", inline_js=inline_js)


_last = {}


def _progress(stage, i, n):
    pct = int(100 * i / n)
    if _last.get(stage, -1) // 25 != pct // 25 or i == n:
        _last[stage] = pct
        logger.info("%s: %d/%d", stage, i, n)
