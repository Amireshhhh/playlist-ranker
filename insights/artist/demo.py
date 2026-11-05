"""Synthetic artist with a planted trend, for the offline demo and tests.

Five albums of eight songs. The share of first-person words falls album by album and
"we" rises, so a correct analysis must find both trends. Nothing here is real data.
"""

import random

from .discography import Corpus, Song

WORDS = ("ocean light sky fire earth whale stone storm silence void planet gravity river forest "
         "dawn shadow burn rise fall life death soul mind time breathe dream").split()
GLUE = "the and of in to a with all is from on we into".split()


def make(seed: int = 7) -> Corpus:
    rng = random.Random(seed)
    albums = [("First Tide", 2001), ("Stone Garden", 2004), ("Iron Sky", 2008), ("Low Orbit", 2012), ("Open Water", 2017)]
    c = Corpus(artist="The Demo Whales", mbid=None, albums=[], source="synthetic demo")
    for order, (title, year) in enumerate(albums):
        c.albums.append({"title": title, "year": year, "order": order, "tracks": 8})
        p_i = 0.30 - 0.06 * order          # first person falls
        p_we = 0.04 + 0.05 * order         # "we" rises
        for k in range(8):
            lines = []
            chorus = " ".join(rng.choice(WORDS + GLUE) for _ in range(6))
            for _ in range(16):
                ws = []
                for _ in range(rng.randint(5, 9)):
                    r = rng.random()
                    ws.append(rng.choice(["i", "my", "me"]) if r < p_i else
                              rng.choice(["we", "us", "our"]) if r < p_i + p_we else rng.choice(WORDS + GLUE))
                lines.append(" ".join(ws))
                if rng.random() < 0.15 + 0.07 * order:
                    lines.append(chorus)
            c.songs.append(Song(title=f"{title.split()[0]} {k + 1}", album=title, year=year, order=order, position=k + 1,
                                duration_s=240, lyrics="\n".join(lines), status="lyrics"))
    c.songs.append(Song(title="Interlude", album="Iron Sky", year=2008, order=2, position=9, duration_s=90,
                        instrumental=True, status="instrumental"))
    return c
