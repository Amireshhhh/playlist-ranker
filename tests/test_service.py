"""Tests for the playlist service: link parsing, file formats, Spotify/YouTube parsing, and the web app."""

import plistlib
import time
from datetime import datetime

import pytest

from insights import connectors as C


@pytest.mark.parametrize("url,expected", [
    ("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc", ("spotify", "37i9dQZF1DXcBWIGoYBM5M")),
    ("spotify:playlist:37i9dQZF1DXcBWIGoYBM5M", ("spotify", "37i9dQZF1DXcBWIGoYBM5M")),
    ("https://www.deezer.com/en/playlist/908622995", ("deezer", "908622995")),
    ("deezer.com/playlist/1234567", ("deezer", "1234567")),
    ("https://music.youtube.com/playlist?list=PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG", ("youtube", "PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG")),
    ("https://www.youtube.com/watch?v=abc&list=PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG", ("youtube", "PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG")),
])
def test_parse_link(url, expected):
    assert C.parse_link(url) == expected


def test_parse_link_rejects_and_explains():
    with pytest.raises(C.InputError, match="Export Playlist"):
        C.parse_link("https://music.apple.com/us/playlist/x/pl.u-abc")
    with pytest.raises(C.InputError):
        C.parse_link("https://evil.example.com/playlist/123")
    with pytest.raises(C.InputError):
        C.parse_link("http://169.254.169.254/latest/meta-data")


def test_youtube_titles():
    assert C.youtube_track("Stranded", "Gojira - Topic") == ("Stranded", "Gojira")
    assert C.youtube_track("Gojira - Stranded (Official Video)", "Gojira") == ("Stranded", "Gojira")
    assert C.youtube_track("Kendrick Lamar – HUMBLE. [Official Music Video]", "KendrickLamarVEVO") == ("HUMBLE.", "Kendrick Lamar")
    assert C.youtube_track("Deleted video", "") is None


def test_apple_xml_export():
    data = plistlib.dumps({
        "Tracks": {"1": {"Name": "Stranded", "Artist": "Gojira", "Album": "Magma", "Total Time": 271000, "Year": 2016,
                         "Date Added": datetime(2024, 1, 2, 3, 4, 5), "Genre": "Metal"},
                   "2": {"Name": "Dreams", "Artist": "Fleetwood Mac", "Total Time": 254000},
                   "3": {"Name": "No artist"}},
        "Playlists": [{"Name": "Mix", "Playlist Items": [{"Track ID": 2}, {"Track ID": 1}, {"Track ID": 3}]}]})
    pl = C.load_upload("Mix.xml", data)
    assert pl.name == "Mix" and [t.title for t in pl.tracks] == ["Dreams", "Stranded"]
    assert pl.tracks[1].year == 2016 and pl.tracks[1].duration_ms == 271000 and pl.tracks[1].genres == ["Metal"]


def test_apple_text_export_utf16_tabs():
    text = "Name\tArtist\tAlbum\tTime\tYear\nStranded\tGojira\tMagma\t271\t2016\nHurt\tJohnny Cash\tAmerican IV\t218\t2002\n"
    pl = C.load_upload("Mix.txt", text.encode("utf-16"))
    assert [(t.title, t.artist) for t in pl.tracks] == [("Stranded", "Gojira"), ("Hurt", "Johnny Cash")]
    assert pl.tracks[0].duration_ms == 271000 and pl.tracks[1].year == 2002


def test_spotify_items_shape():
    calls = []

    def get(url, params, token):
        calls.append(url)
        if url.endswith("/items"):
            return 200, {"items": [
                {"added_at": "2024-01-01T00:00:00Z", "item": {"type": "track", "name": "Stranded", "uri": "spotify:track:x",
                 "artists": [{"name": "Gojira"}], "album": {"name": "Magma", "release_date": "2016-06-17"},
                 "duration_ms": 271000, "external_ids": {"isrc": "FR123"}}},
                {"item": {"type": "episode", "name": "A podcast"}},
                {"track": {"type": "track", "name": "Hurt", "artists": [{"name": "Johnny Cash"}], "album": {}}}]}
        return 200, {"name": "Mine", "owner": {"display_name": "amir"}}

    pl = C.load_spotify(get, "abc", "token")
    assert pl.name == "Mine" and [t.title for t in pl.tracks] == ["Stranded", "Hurt"]
    assert pl.tracks[0].isrc == "FR123" and pl.tracks[0].year == 2016
    with pytest.raises(C.InputError, match="Log in"):
        C.load_spotify(get, "abc", None)
    with pytest.raises(C.InputError, match="own or collaborate"):
        C.load_spotify(lambda u, p, t: (403, None), "abc", "token")


# ----------------------------------------------------------------------------- web app

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import serving.insights_app as A
    from insights.sources import Sources
    monkeypatch.setattr(A, "OFFLINE", True)
    monkeypatch.setattr(A, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(A, "_src", Sources(cache_dir=str(tmp_path / "cache"), offline=True))
    monkeypatch.setattr(A, "PER_HOUR", 4)
    A.recent.clear()
    A.jobs.clear()
    return TestClient(A.app), A


def _wait(c, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_upload_to_report(client):
    c, A = client
    assert "Analyse a playlist" in c.get("/").text
    csv = "Track Name,Artist Name(s),Album Name,Track Duration (ms)\n" + "\n".join(
        f"Song {i},Artist {i % 4},Album,{200000 + i}" for i in range(12))
    r = c.post("/api/analyze", files={"file": ("road trip.csv", csv.encode(), "text/csv")})
    assert r.status_code == 200
    j = _wait(c, r.json()["job"])
    assert j["status"] == "done", j
    page = c.get(j["report"])
    assert page.status_code == 200 and "road trip" in page.text and "Overview" in page.text


def test_bad_inputs(client):
    c, A = client
    assert c.post("/api/analyze", data={"url": ""}).status_code == 400
    r = c.post("/api/analyze", data={"url": "https://music.apple.com/us/playlist/x/pl.u-1"})
    assert r.status_code == 400 and "Export Playlist" in r.json()["detail"]
    r = c.post("/api/analyze", data={"url": "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"})
    assert r.status_code == 400 and "Exportify" in r.json()["detail"]     # Spotify not configured here
    r = c.post("/api/analyze", files={"file": ("x.csv", b"foo,bar\n1,2\n", "text/csv")})
    assert _wait(c, r.json()["job"])["status"] == "error"
    # every request counts toward the hourly limit, valid or not: this is the 5th (limit 4)
    assert c.post("/api/analyze", data={"url": "x"}).status_code == 429
    assert c.get("/r/../../etc/passwd").status_code == 404
    assert c.get("/api/jobs/nope").status_code == 404


def test_upload_size_limit(client, monkeypatch):
    c, A = client
    monkeypatch.setattr(A, "MAX_UPLOAD", 100)
    r = c.post("/api/analyze", files={"file": ("big.csv", b"x" * 500, "text/csv")})
    assert r.status_code == 413
