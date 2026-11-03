"""Offline demo: synthetic playlists with invented artists, lyrics and metadata.

Nothing here is real data. It exists so the whole pipeline (analysis, clustering,
recommendations, dashboards) can be run and tested without network access.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from . import pipeline
from .models import ArtistInfo, Playlist, Track

STYLES = {
    "doom metal": dict(bpm=(60, 85), energy=(0.7, 0.95), valence=(0.05, 0.3), year=(1990, 2015),
                       words="ocean whale forest earth fire ash burn dying flesh stone storm silence void abyss "
                             "bleeding planet gravity drown iron", countries=["FR", "US", "SE"]),
    "synth pop": dict(bpm=(110, 128), energy=(0.6, 0.85), valence=(0.6, 0.95), year=(1981, 1989),
                      words="neon heart dance tonight lights city love baby forever dream kiss shine radio "
                            "electric summer midnight", countries=["GB", "DE", "US"]),
    "indie folk": dict(bpm=(70, 105), energy=(0.2, 0.45), valence=(0.3, 0.6), year=(2005, 2022),
                       words="river home mountain winter morning letter coffee window road quiet father garden "
                             "train autumn snow", countries=["US", "CA", "IE"]),
    "hip hop": dict(bpm=(85, 98), energy=(0.55, 0.8), valence=(0.35, 0.7), year=(1994, 2024),
                    words="money street block hustle crew flow mic rhyme city grind gold real dream boss "
                          "hood", countries=["US", "GB"]),
}
ARTISTS = {
    "doom metal": ["Abyssal Tide", "Grey Monolith", "Ashen Choir"],
    "synth pop": ["Neon Parade", "Velvet Circuit", "The Polaroids"],
    "indie folk": ["Willow & Pine", "June Harbor", "The Quiet Rooms"],
    "hip hop": ["MC Lowfreq", "Block Theory", "Kali Verse"],
}
GLUE = "i you we the and my your in of to all never no one can feel know night time".split()


def _lyrics(rng, words):
    ws = words.split()
    chorus = [" ".join(rng.choice(ws + GLUE) for _ in range(rng.randint(5, 8))) for _ in range(2)]
    parts = []
    for v in range(3):
        parts.append(f"[Verse {v + 1}]")
        parts += [" ".join(rng.choice(ws + GLUE + GLUE) for _ in range(rng.randint(6, 10))) for _ in range(4)]
        parts.append("[Chorus]")
        parts += chorus * rng.randint(1, 2)
    if rng.random() < 0.5:
        parts.append("i love this beautiful day" if rng.random() < 0.5 else "i hate the pain and the fear")
    return "\n".join(parts)


def make_playlist(name, mix, n, seed, owner="demo-user", start=datetime(2024, 1, 5, tzinfo=timezone.utc)):
    rng = random.Random(seed)
    tracks, t0 = [], start
    styles, weights = zip(*mix.items())
    for i in range(n):
        style = rng.choices(styles, weights)[0]
        s = STYLES[style]
        artist = rng.choice(ARTISTS[style])
        t0 += timedelta(hours=rng.randint(2, 120))
        instrumental = rng.random() < 0.05
        tracks.append(Track(
            title=f"{rng.choice(s['words'].split()).title()} {rng.choice(['Song', 'Dream', 'Light', 'Tide', 'Road', 'Line'])} {i + 1}",
            artist=artist, album=f"{artist} LP {rng.randint(1, 3)}", duration_ms=rng.randint(150_000, 420_000),
            release_date=f"{rng.randint(*s['year'])}-01-01", explicit=(style == "hip hop" and rng.random() < 0.6),
            added_at=t0.isoformat(), deezer_rank=int(10 ** rng.uniform(4, 6)),
            bpm=round(rng.uniform(*s["bpm"]), 1),
            audio={"energy": rng.uniform(*s["energy"]), "valence": rng.uniform(*s["valence"]),
                   "danceability": rng.uniform(0.3, 0.9), "acousticness": rng.uniform(0, 0.8),
                   "instrumentalness": 0.9 if instrumental else rng.uniform(0, 0.1),
                   "speechiness": 0.3 if style == "hip hop" else rng.uniform(0.03, 0.08), "liveness": rng.uniform(0.05, 0.3)},
            instrumental=instrumental, lyrics=None if instrumental else _lyrics(rng, s["words"]),
        ))
    pl = Playlist(name=name, tracks=tracks, owner=owner, source="synthetic demo data")
    for style, names in ARTISTS.items():
        for k, a in enumerate(names):
            others = [x for x in names if x != a] + [f"{a.split()[-1]} Revival", f"New {style.title()} Collective",
                                                     f"Echoes of {a.split()[0]}"]
            pl.artists[a] = ArtistInfo(
                name=a, tags={style: 1.0, style.split()[-1]: 0.6, "demo": 0.2}, country=STYLES[style]["countries"][k % len(STYLES[style]["countries"])],
                begin_year=STYLES[style]["year"][0] + k * 3, kind="Group", similar=[(o, 1 - j * 0.15) for j, o in enumerate(others)],
                bio=f"{a} is an invented {style} act used for the offline demo.")
    return pl


class _FakeDeezer:
    def search_artist(self, name):
        return {"id": abs(hash(name)) % 10**6, "name": name}

    def top_tracks(self, artist_id, limit=5):
        rng = random.Random(artist_id)
        return [{"title": f"Demo Single {artist_id % 97}-{k}", "artist": {"name": None}, "rank": int(10 ** rng.uniform(4, 6)),
                 "album": {"title": "Demo EP"}} for k in range(limit)]


class FakeSources:
    """Stands in for insights.sources.Sources in the demo: artist names come back unchanged."""

    def __init__(self):
        self.deezer = _FakeDeezer()
        orig = self.deezer.top_tracks
        names = {}
        search = self.deezer.search_artist

        def search_and_remember(name):
            r = search(name)
            names[r["id"]] = name
            return r

        def top_with_names(aid, limit=5):
            rows = orig(aid, limit)
            for r in rows:
                r["artist"] = {"name": names.get(aid, "?")}
            return rows

        self.deezer.search_artist = search_and_remember
        self.deezer.top_tracks = top_with_names


def run(out_dir="reports", inline_js=False):
    pls = [
        make_playlist("Demo - Heavy Days", {"doom metal": 0.7, "hip hop": 0.2, "indie folk": 0.1}, 40, 1),
        make_playlist("Demo - Night Drive", {"synth pop": 0.6, "hip hop": 0.3, "doom metal": 0.1}, 35, 2),
        make_playlist("Demo - Sunday Morning", {"indie folk": 0.7, "synth pop": 0.3}, 30, 3),
    ]
    return pipeline.build_user("demo-user", pls, out_dir, src=FakeSources(), enrich_data=False, inline_js=inline_js)
