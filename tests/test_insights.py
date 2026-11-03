"""Tests for the insights package (no network: API clients are exercised with fakes)."""

import csv
import json
from collections import Counter

import numpy as np
import pytest

from insights import clustering, ingest, lyrics, recommend, stats
from insights.httpcache import CachedHTTP
from insights.models import ArtistInfo, Playlist, Track, norm_key
from insights.sources import Deezer, LastFM, LRCLib, MusicBrainz, ReccoBeats


# ----------------------------------------------------------------------------- ingest

EXPORTIFY_HEADER = ["Track URI", "Track Name", "Artist URI(s)", "Artist Name(s)", "Album Name", "Album Release Date",
                    "Track Duration (ms)", "Explicit?", "Popularity", "Added At", "Artist Genres", "Energy", "Valence",
                    "Tempo"]


def write_exportify(path):
    rows = [
        ["spotify:track:aaa", "Stranded", "spotify:artist:g1", "Gojira", "Magma", "2016-06-17", "271000", "false", "60",
         "2024-01-01T10:00:00Z", "french metal,progressive death metal", "0.9", "0.2", "105.0"],
        ["spotify:track:bbb", "See You Again", "spotify:artist:t1", "Tyler, The Creator", "Flower Boy", "2017-07-21",
         "180000", "true", "80", "2024-01-02T11:00:00Z", "hip hop", "", "", ""],
        ["spotify:track:ccc", "Collab", "spotify:artist:x1,spotify:artist:x2", "Artist A,Artist B", "Alb", "2001",
         "200000", "false", "", "", "", "", "", ""],
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EXPORTIFY_HEADER)
        w.writerows(rows)


def test_exportify_csv(tmp_path):
    p = tmp_path / "my_list.csv"
    write_exportify(p)
    pl = ingest.load(str(p), owner="amir")
    assert pl.name == "my list" and pl.owner == "amir" and len(pl.tracks) == 3
    g, t, c = pl.tracks
    assert g.artist == "Gojira" and g.year == 2016 and g.duration_ms == 271000 and g.explicit is False
    assert g.genres == ["french metal", "progressive death metal"]
    assert g.audio == {"energy": 0.9, "valence": 0.2, "tempo": 105.0} and g.tempo == 105.0
    assert g.spotify_id == "aaa" and g.popularity == 60
    # One URI -> the comma belongs to the name
    assert t.artists == ["Tyler, The Creator"] and t.explicit is True
    # Two URIs and two names -> split
    assert c.artists == ["Artist A", "Artist B"] and c.artist == "Artist A" and c.year == 2001


def test_simple_csv_and_json(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("title,artist\nA,X\n,Y\nB,Z\n", encoding="utf-8")
    pl = ingest.load(str(p))
    assert [t.title for t in pl.tracks] == ["A", "B"]  # row without a title is skipped
    j = tmp_path / "p.json"
    j.write_text(json.dumps({"name": "J", "tracks": [{"title": "T", "artist": "A", "release_date": "1999"}]}))
    pl = ingest.load(str(j))
    assert pl.name == "J" and pl.tracks[0].year == 1999


def test_csv_without_columns_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="title column"):
        ingest.load(str(p))


def test_mpd_slice(tmp_path):
    from data.mpd_loader import MPDLoader
    MPDLoader(data_dir=str(tmp_path))._create_sample_data()
    f = tmp_path / "mpd.slice.0-999.json"
    pl = ingest.load(str(f), pid=3)
    raw = json.loads(f.read_text())["playlists"][3]
    assert len(pl.tracks) == raw["num_tracks"]
    assert pl.tracks[0].spotify_uri == raw["tracks"][0]["track_uri"]
    with pytest.raises(ValueError, match="pid"):
        ingest.load(str(f))


def test_norm_key():
    assert norm_key("Stranded (Remastered 2011)") == "stranded"
    assert norm_key("Song - Live at Wembley") == "song"
    assert Track("Hey!", "AC/DC").key == "acdc|hey"


# ----------------------------------------------------------------------------- lyrics

def test_tokenize_and_stopwords():
    toks = lyrics.tokenize("I'm walkin' — on the OCEAN, ocean’s floor 42")
    assert toks == ["i'm", "walkin", "on", "the", "ocean", "ocean's", "floor"]
    assert lyrics.content_words(toks) == ["walkin", "ocean", "ocean's", "floor"]


def test_mattr_exact():
    toks = ["a", "b"] * 30  # every 50-token window holds 2 types
    assert lyrics.mattr(toks, window=50) == pytest.approx(2 / 50)
    assert lyrics.mattr(toks[:10], window=50) is None
    distinct = [f"w{i}" for i in range(60)]
    assert lyrics.mattr(distinct, window=50) == pytest.approx(1.0)


def test_repetitiveness_orders_texts():
    chorus = "\n".join(["we will rise again tonight"] * 40)
    rng = np.random.default_rng(0)
    vocab = [f"word{i}" for i in range(500)]
    varied = "\n".join(" ".join(rng.choice(vocab, 5)) for _ in range(40))
    assert lyrics.repetitiveness(chorus) > 0.9 > lyrics.repetitiveness(varied)
    assert lyrics.repetitiveness("short") is None


def test_sentiment_sign():
    assert lyrics.sentiment("I love this wonderful happy day")["compound"] > 0.5
    assert lyrics.sentiment("I hate this terrible awful pain")["compound"] < -0.5


def test_analyze_counts():
    ts = [
        Track("Ocean Song", "Band", lyrics="[Chorus]\nocean ocean fire\nyou and me"),
        Track("Fire Song", "Band", lyrics="fire burns\nwe are fire"),
        Track("Silent", "Band", lyrics=None),
        Track("Instr", "Band", lyrics="la la", instrumental=True),
    ]
    rep = lyrics.analyze(ts, n_themes=2)
    assert len(rep.songs) == 2
    # '[Chorus]' is dropped, 'band' would be an artist word, stopwords removed
    assert rep.word_counts == Counter({"ocean": 2, "fire": 3, "burns": 1})
    assert rep.pronouns["you"] == 1 and rep.pronouns["I / me"] == 1 and rep.pronouns["we / us"] == 1
    assert rep.total_words == 6 + 5  # 'ocean ocean fire you and me' + 'fire burns we are fire'
    assert lyrics.analyze([Track("x", "y")]) is None


# ----------------------------------------------------------------------------- stats

def test_basic_stats_and_health():
    ts = [Track(f"t{i}", a, duration_ms=60_000, release_date=f"{y}-01-01")
          for i, (a, y) in enumerate([("A", 1991), ("A", 1995), ("B", 2001), ("C", 2012)])]
    st = stats.basic(Playlist("p", ts))
    assert st["n_tracks"] == 4 and st["n_artists"] == 3 and st["total_ms"] == 240_000
    assert st["decades"] == {1990: 2, 2000: 1, 2010: 1}
    assert st["repeat_rate"] == pytest.approx(1 / 3)  # (A, A) is 1 of 3 adjacent pairs
    assert st["artist_entropy_bits"] == pytest.approx(1.5)  # p = 1/2, 1/4, 1/4
    assert st["lorenz"] == pytest.approx([0, 0.25, 0.5, 1.0])
    assert st["one_hit_artists"] == 2


def test_flow_handles_double_time_and_improves():
    bpms = [120, 60, 121, 90, 61, 119]
    pl = Playlist("p", [Track(f"t{i}", "a", bpm=b) for i, b in enumerate(bpms)])
    f = stats.flow(pl)
    assert f["mean_jump_suggested"] <= f["mean_jump_current"]
    assert sorted(f["suggested_order"]) == list(range(6))
    # The reported mean matches the order it describes
    d = lambda a, b: min(abs(a - b), abs(2 * a - b), abs(a - 2 * b))  # noqa: E731
    order = [bpms[i] for i in f["suggested_order"]]
    assert f["mean_jump_suggested"] == pytest.approx(np.mean([d(a, b) for a, b in zip(order, order[1:])]))
    # 60 -> 120 -> 60 is all double/half time: zero jump
    pl2 = Playlist("p", [Track(f"t{i}", "a", bpm=b) for i, b in enumerate([60, 120, 60])])
    assert stats.flow(pl2)["mean_jump_current"] == 0


def test_mainstream_prefers_spotify_popularity():
    pl = Playlist("p", [Track("a", "x", popularity=20, deezer_rank=10**6), Track("b", "y", popularity=80)])
    m = stats.mainstream_score(pl)
    assert m["source"] == "Spotify popularity" and m["score"] == pytest.approx(0.5)
    assert m["deepest_cut"].title == "a" and m["biggest_hit"].title == "b"
    pl = Playlist("p", [Track("a", "x", deezer_rank=1000), Track("b", "y", deezer_rank=10**6)])
    assert stats.mainstream_score(pl)["score"] == pytest.approx((3 / 6 + 1.0) / 2)


def test_overlap():
    a = Playlist("a", [Track("1", "x"), Track("2", "y")])
    b = Playlist("b", [Track("2", "y"), Track("3", "z")])
    ov = stats.overlap([a, b])
    assert ov["tracks"][0][1] == pytest.approx(1 / 3) and ov["tracks"][0][0] == 1.0


# ----------------------------------------------------------------------------- clustering

def two_scene_playlist():
    ts = []
    for i in range(10):
        ts.append(Track(f"m{i}", f"Metal{i % 3}", bpm=70 + i, audio={"energy": 0.9, "valence": 0.1}))
        ts.append(Track(f"p{i}", f"Pop{i % 3}", bpm=120 + i, audio={"energy": 0.5, "valence": 0.9}))
    pl = Playlist("p", ts)
    for k in range(3):
        pl.artists[f"Metal{k}"] = ArtistInfo(f"Metal{k}", tags={"metal": 1.0, "doom": 0.5})
        pl.artists[f"Pop{k}"] = ArtistInfo(f"Pop{k}", tags={"pop": 1.0, "synth": 0.5})
    return pl


def test_kmeans_recovers_two_scenes():
    pl = two_scene_playlist()
    res = clustering.cluster_songs(pl)
    assert res.k == 2 and res.silhouette > 0.5
    metal = {res.labels[i] for i, t in enumerate(pl.tracks) if t.title.startswith("m")}
    pop = {res.labels[i] for i, t in enumerate(pl.tracks) if t.title.startswith("p")}
    assert len(metal) == 1 and len(pop) == 1 and metal != pop
    assert {s["name"] for s in res.summaries} == {"metal / doom", "pop / synth"}


def test_artist_knn_links_within_scene():
    g = clustering.artist_knn(two_scene_playlist(), k=2)
    for a, b, w in g.edges:
        assert g.names[a][:3] == g.names[b][:3]  # Metal-Metal or Pop-Pop only
    assert np.isfinite(g.coords).all()


# ----------------------------------------------------------------------------- http + sources

class FakeHTTP:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get_json(self, url, params=None):
        self.calls.append((url, params))
        for key, val in self.routes.items():
            if key in url:
                return val(params) if callable(val) else val
        return None


def test_cache_and_offline(tmp_path, monkeypatch):
    http = CachedHTTP(cache_dir=str(tmp_path), offline=False)

    class R:
        status_code = 200

        def json(self):
            return {"ok": 1}

    class S:
        headers = {}
        n = 0

        def get(self, url, timeout):
            S.n += 1
            return R()

    http._session = S()
    assert http.get_json("https://example.org/x", {"q": 1, "api_key": "SECRET"}) == {"ok": 1}
    assert http.get_json("https://example.org/x", {"q": 1, "api_key": "OTHER"}) == {"ok": 1}
    assert S.n == 1  # second call served from cache, key not part of the cache key
    assert "SECRET" not in "".join(p.read_text() for p in tmp_path.rglob("*.json"))
    off = CachedHTTP(cache_dir=str(tmp_path), offline=True)
    assert off.get_json("https://example.org/x", {"q": 1}) == {"ok": 1}
    assert off.get_json("https://example.org/missing") is None


def test_lrclib_prefers_exact_then_search():
    http = FakeHTTP({"/api/get": {"plainLyrics": "exact"}})
    assert LRCLib(http).lyrics("t", "a", duration_s=200)["plainLyrics"] == "exact"
    http = FakeHTTP({"/api/search": [{"duration": 300, "plainLyrics": "far"}, {"duration": 201, "plainLyrics": "near"},
                                     {"duration": 200, "plainLyrics": None}]})
    assert LRCLib(http).lyrics("t", "a", duration_s=200)["plainLyrics"] == "near"


def test_deezer_error_object_is_none():
    http = FakeHTTP({"/track/": {"error": {"type": "DataException", "code": 800}}})
    assert Deezer(http).track(1) is None


def test_lastfm_tags_scaled_and_disabled_without_key():
    http = FakeHTTP({"audioscrobbler": {"toptags": {"tag": [{"name": "Metal", "count": 100},
                                                            {"name": "French", "count": 40}]}}})
    assert LastFM(http, api_key="k").top_tags("Gojira") == {"metal": 1.0, "french": 0.4}
    assert LastFM(http, api_key="").top_tags("Gojira") == {} or not LastFM(http, api_key="").enabled


def test_musicbrainz_score_threshold():
    http = FakeHTTP({"musicbrainz": {"artists": [{"name": "Gojyra", "score": 60}]}})
    assert MusicBrainz(http).artist("Gojira") is None
    http = FakeHTTP({"musicbrainz": {"artists": [{"name": "Gojira", "score": 100, "country": "FR"}]}})
    assert MusicBrainz(http).artist("gojira")["country"] == "FR"


def test_reccobeats_maps_by_spotify_href():
    http = FakeHTTP({"reccobeats": {"content": [
        {"id": "uuid-1", "href": "https://open.spotify.com/track/aaa", "energy": 0.8, "valence": 0.3, "tempo": 120}]}})
    assert ReccoBeats(http).audio_features(["aaa", "bbb"]) == {"aaa": {"energy": 0.8, "valence": 0.3, "tempo": 120.0}}


# ----------------------------------------------------------------------------- recommendations

class FakeSrc:
    class deezer:
        @staticmethod
        def search_artist(name):
            return {"id": name, "name": name}

        @staticmethod
        def top_tracks(aid, limit=5):
            return [{"title": f"{aid} hit {k}", "artist": {"name": aid}, "rank": 1000 * (k + 1)} for k in range(limit)]


def test_similar_artists_excludes_known_and_caps():
    pl = Playlist("p", [Track("s1", "A"), Track("s2", "A"), Track("s3", "B")])
    pl.artists["A"] = ArtistInfo("A", similar=[("B", 0.9), ("C", 1.0), ("D", 0.5)])
    pl.artists["B"] = ArtistInfo("B", similar=[("C", 0.8), ("E", 0.9)])
    recs = recommend.similar_artists(pl, FakeSrc, n=10, per_artist=3, max_per_artist=2)
    arts = Counter(r["artist"] for r in recs)
    assert "A" not in arts and "B" not in arts
    assert max(arts.values()) <= 2
    assert recs[0]["artist"] == "C"  # C gets votes from both A (2/3 * 1.0) and B (1/3 * 0.8)
    assert "A" in recs[0]["reason"] and "B" in recs[0]["reason"]


def test_collaborative_returns_none_without_data(tmp_path):
    assert recommend.collaborative(Playlist("p", [Track("a", "b")]), str(tmp_path), str(tmp_path / "x.pt")) is None


# ----------------------------------------------------------------------------- end to end

def test_demo_dashboards(tmp_path):
    from insights import demo
    index = demo.run(str(tmp_path))
    html = index.read_text()
    assert "Taste across playlists" in html
    pages = sorted(tmp_path.glob("demo-user/demo-*/index.html"))
    assert len(pages) == 3
    page = pages[0].read_text()
    for section in ("Overview", "Lyrics", "Song groups", "Recommendations", "Mood map"):
        assert section in page
    figs = json.loads(page.split("const FIGS = ", 1)[1].split(";\nconst PAL", 1)[0])
    assert "f_clusters" in figs and "f_words" in figs
    # Lyrics text must never be written into the page: no full lyric line appears in it.
    pl = demo.make_playlist("Demo - Heavy Days", {"doom metal": 0.7, "hip hop": 0.2, "indie folk": 0.1}, 40, 1)
    lines = [ln for t in pl.tracks if t.lyrics for ln in lyrics.clean_lines(t.lyrics) if len(ln.split()) >= 5]
    heavy = (tmp_path / "demo-user" / "demo-heavy-days" / "index.html").read_text()
    assert lines and not any(ln in heavy for ln in lines)
    summary = json.loads((tmp_path / "demo-user" / "demo-heavy-days" / "summary.json").read_text())
    assert summary["stats"]["n_tracks"] == 40
