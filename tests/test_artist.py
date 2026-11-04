"""Tests for artist pages: statistics helpers, planted-effect recovery, discography parsing."""

import itertools
import math

import numpy as np
import pytest

from insights.artist import demo, discography, textstats as T


def test_spearman_exact_minimum_p():
    rho, p, exact = T.spearman_exact([1, 2, 3, 4, 5])
    assert exact and rho == pytest.approx(1.0)
    assert p == pytest.approx(2 / math.factorial(5))  # only the identity and its reverse reach |rho| = 1


def test_spearman_exact_matches_enumeration():
    y = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0]
    rho, p, _ = T.spearman_exact(y)
    from scipy.stats import spearmanr
    vals = [abs(spearmanr(range(6), [y[i] for i in perm]).statistic) for perm in itertools.permutations(range(6))]
    assert p == pytest.approx(np.mean(np.array(vals) >= abs(rho) - 1e-12))


def test_bh_matches_definition():
    p = [0.01, 0.04, 0.03, 0.2]
    # sorted: .01 .03 .04 .2 -> raw q: .04 .06 .0533 .2 -> monotone: .04 .0533 .0533 .2
    assert T.bh(p) == pytest.approx([0.04, 0.05333333, 0.05333333, 0.2])


def test_score_families_and_verdicts():
    tests = [{"p": 0.01, "family": "a"}, {"p": 0.04, "family": "a"}, {"p": 0.03, "family": "b"},
             {"p": float("nan"), "family": "c"}, {"p": float("nan"), "family": "fit"}]
    out = T.score(tests)
    assert [t["result"] for t in out] == ["Held", "Held", "Held", "Not testable", "Fit"]
    assert out[1]["q"] == pytest.approx(0.04)


def test_contractions_and_unique_lines():
    assert T.expand("I'm sure you're right, it's what we've lost, can't stop") == \
        "i am sure you are right, it is what we have lost, can not stop"
    assert T.unique_lines(["We rise", "we rise!", "We fall"]) == ["We rise", "We fall"]


@pytest.fixture(scope="module")
def res():
    return T.analyze(demo.make())


def test_rarefaction_bounds(res):
    for r in res["rarefaction"]["rows"]:
        assert 0 < r["expected"] <= res["facts"]["types"]


def test_planted_effects_found_and_nothing_else_in_trends(res):
    held = {t["claim"] for t in res["tests"] if t["result"] == "Held"}
    assert "“I” rate changes with album order" in held
    assert "“we” rate changes with album order" in held
    assert res["pronouns"]["I"]["rho"] == pytest.approx(-1.0)
    assert res["headline"]["title"] == "The Demo Whales stopped saying “I”"
    # no theme is planted, so none may hold
    assert not any(t["claim"].startswith("Theme") and t["result"] == "Held" for t in res["tests"])
    # lyrics text never leaks into the result except as counts/ids
    assert "selfsim" in res and all(isinstance(i, int) for i in res["selfsim"][0]["ids"])


def test_artist_page_renders_without_lyrics_text(res, tmp_path):
    from insights.artist import page
    out = page.render(res, tmp_path / "a.html")
    html = out.read_text()
    assert "stopped saying" in html and "Everything tested" in html
    lines = [ln for s in demo.make().songs if s.lyrics for ln in s.lyrics.splitlines() if len(ln.split()) >= 7]
    assert lines and not any(ln in html for ln in lines)


class FakeHTTP:
    def __init__(self, routes):
        self.routes = routes

    def get_json(self, url, params=None):
        for k, v in self.routes.items():
            if url.endswith(k):
                return v(params) if callable(v) else v
        return None


def test_discography_picks_studio_albums_and_standard_tracklist():
    rgs = {"release-groups": [
        {"id": "g2", "title": "Later", "primary-type": "Album", "secondary-types": [], "first-release-date": "2010-05-01"},
        {"id": "g1", "title": "Debut", "primary-type": "Album", "secondary-types": [], "first-release-date": "2001"},
        {"id": "g3", "title": "Live!", "primary-type": "Album", "secondary-types": ["Live"], "first-release-date": "2005"},
        {"id": "g4", "title": "A Single", "primary-type": "Single", "secondary-types": [], "first-release-date": "2003"}]}

    def releases(params):
        std = {"date": "2001-01-01", "media": [{"tracks": [{"title": "A", "length": 200000}, {"title": "B", "length": 180000}]}]}
        deluxe = {"date": "2002-01-01", "media": [{"tracks": [{"title": "A"}, {"title": "B"}, {"title": "Bonus"}]}]}
        return {"releases": [deluxe, std, dict(std, date="2003-01-01")]}

    http = FakeHTTP({"/release-group": rgs, "/release": releases})
    albums = discography.studio_albums(http, "x")
    assert [a["title"] for a in albums] == ["Debut", "Later"]
    tl = discography.canonical_tracklist(http, "g1")
    assert [t["title"] for t in tl] == ["A", "B"] and tl[0]["duration_s"] == 200
