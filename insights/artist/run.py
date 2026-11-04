"""Build artist pages: corpus -> analysis -> page + summary.json."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .. import dashboard as D
from ..pipeline import slug
from . import discography, page, textstats

logger = logging.getLogger(__name__)


def build(name: str, src, out_dir: str, inline_js: bool = False, back_href: Optional[str] = "../../index.html",
          corpus=None) -> Optional[Path]:
    t0 = time.time()
    if corpus is None:
        corpus = discography.build(src, name, progress=lambda s, i, n: logger.info("%s %s: %d/%d", name, s, i, n))
    if corpus is None:
        return None
    res = textstats.analyze(corpus)
    root = Path(out_dir) / "artists" / slug(corpus.artist)
    if not res.get("ok"):
        logger.warning("%s: %s", name, res.get("reason"))
        return None
    path = page.render(res, root / "index.html", inline_js=inline_js, back_href=back_href)
    summary = {k: v for k, v in res.items() if k not in ("selfsim", "lookup")}
    summary["timing_s"] = round(time.time() - t0, 1)
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=D._json_default), encoding="utf-8")
    logger.info("%s: %d songs, %.0f s -> %s", corpus.artist, res["facts"]["songs"], time.time() - t0, path)
    return path
