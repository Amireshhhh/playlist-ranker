"""Build artist pages: corpus -> analysis -> page + summary.json + analysis.json.

analysis.json holds everything a page is drawn from (counts, scores, per-song integer ids;
never lyrics text). With ``prev_dir`` pointing at the previously published site, an artist
whose analysis was made by the same engine version is re-drawn from it without any
network calls, so a change to the page layout republishes in minutes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

from .. import dashboard as D
from ..pipeline import slug
from . import discography, page, textstats

logger = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent


def engine_version() -> str:
    """Hash of the code that decides the numbers (not the page layout)."""
    h = hashlib.sha256()
    for f in (_HERE / "textstats.py", _HERE / "discography.py", _HERE.parent / "lyrics.py"):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _write(res, root: Path, t0: float, inline_js: bool, back_href: Optional[str]) -> Path:
    path = page.render(res, root / "index.html", inline_js=inline_js, back_href=back_href)
    summary = {k: v for k, v in res.items() if k not in ("selfsim", "lookup")}
    summary["timing_s"] = round(time.time() - t0, 1)
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=D._json_default), encoding="utf-8")
    (root / "analysis.json").write_text(json.dumps(res, default=D._json_default), encoding="utf-8")
    return path


def build(name: str, src, out_dir: str, inline_js: bool = False, back_href: Optional[str] = "../../index.html",
          corpus=None, prev_dir: Optional[str] = None, section: str = "artists") -> Optional[Path]:
    t0 = time.time()
    eng = engine_version()
    if prev_dir and corpus is None:
        old = Path(prev_dir) / section / slug(name) / "analysis.json"
        if old.exists():
            res = json.loads(old.read_text(encoding="utf-8"))
            if res.get("engine") == eng:
                logger.info("%s: reusing analysis from the published site (engine %s)", name, eng)
                return _write(res, Path(out_dir) / section / slug(name), t0, inline_js, back_href)
    fetched = corpus is None
    if fetched:
        corpus = discography.build(src, name, progress=lambda s, i, n: logger.info("%s %s: %d/%d", name, s, i, n))
    if corpus is None:
        return None
    res = textstats.analyze(corpus)
    if not res.get("ok"):
        logger.warning("%s: %s", name, res.get("reason"))
        return None
    res["engine"] = eng
    for k in ("domain", "source_note"):
        if getattr(corpus, k, None):
            res[k] = getattr(corpus, k)
    if res.get("domain"):
        res["headline"] = {k: page.relabel(v, res["domain"]) for k, v in res["headline"].items()}
        for t in res["tests"]:
            for k in ("claim", "stat", "method"):
                t[k] = page.relabel(t[k], res["domain"])
    # Folder named after the requested name, so a later run can find it again without a lookup.
    path = _write(res, Path(out_dir) / section / slug(name if fetched else corpus.artist), t0, inline_js, back_href)
    logger.info("%s: %d songs, %.0f s -> %s", corpus.artist, res["facts"]["songs"], time.time() - t0, path)
    return path
