"""A non-music corpus for the same engine: US presidential inaugural addresses, 1789-2025.

The texts are works of the US federal government, so they are in the public domain. The
source is the NLTK "inaugural" corpus (compiled from C-SPAN), one file per address named
``<year>-<President>.txt``. Each address is one unit ("speech"); addresses are grouped into
six eras so the album-trend tests become era-trend tests.

  python -m insights speeches path/to/inaugural   # the unzipped corpus folder
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .artist.discography import Corpus, Song

ERAS = [(1789, 1829, "Founding era"), (1830, 1869, "Antebellum & Civil War"), (1870, 1909, "Gilded Age"),
        (1910, 1949, "World wars"), (1950, 1989, "Cold War"), (1990, 2100, "Since 1990")]
URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/inaugural.zip"


def sentences(text: str):
    """One sentence per line, so line-level measures (sentiment, repeats) work per sentence."""
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", flat) if len(s.strip()) > 1]


def load(folder: str) -> Optional[Corpus]:
    files = sorted(Path(folder).glob("[0-9][0-9][0-9][0-9]-*.txt"))
    if not files:
        return None
    c = Corpus(artist="US presidents", mbid=None, albums=[], source="NLTK inaugural corpus (C-SPAN)")
    c.domain = "speeches"
    c.source_note = (f"{len(files)} inaugural addresses, {files[0].stem[:4]}–{files[-1].stem[:4]}, from the NLTK inaugural corpus "
                     "(compiled from C-SPAN). US government works, public domain. Grouped into six eras; each sentence counts as a line.")
    for order, (lo, hi, name) in enumerate(ERAS):
        c.albums.append({"title": name, "year": lo, "order": order, "tracks": 0})
    for f in files:
        year = int(f.stem[:4])
        president = f.stem[5:]
        order = next(i for i, (lo, hi, _) in enumerate(ERAS) if lo <= year <= hi)
        c.albums[order]["tracks"] += 1
        text = f.read_text(encoding="utf-8", errors="replace")
        c.songs.append(Song(title=f"{year} {president}", album=ERAS[order][2], year=year, order=order,
                            position=c.albums[order]["tracks"], duration_s=None, lyrics="\n".join(sentences(text)),
                            status="lyrics"))
    return c
