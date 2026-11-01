"""Tests for the FastAPI service and its re-ranking logic. Run with: pytest tests/"""

import pytest
from fastapi.testclient import TestClient

import serving.api as api


@pytest.fixture
def client(monkeypatch, tmp_path):
    # No checkpoint and no data: the service starts with an untrained model and mock scores.
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.pt")
    monkeypatch.setattr(api, "DATA_DIR", tmp_path / "missing")
    with TestClient(api.app) as c:
        yield c


def test_health_reports_untrained_model(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model_trained"] is False


def test_empty_prefix_is_400(client):
    r = client.post("/rank", json={"playlist_prefix": []})
    assert r.status_code == 400
    assert "playlist_prefix" in r.json()["detail"]


def test_unknown_constraint_key_is_400(client):
    r = client.post("/rank", json={"playlist_prefix": ["t1"], "constraint_weights": {"exposure": 1.0}})
    assert r.status_code == 400
    assert "exposure" in r.json()["detail"]


def test_negative_constraint_weight_is_400(client):
    r = client.post("/rank", json={"playlist_prefix": ["t1"], "constraint_weights": {"fatigue": -1}})
    assert r.status_code == 400


def test_rank_returns_top_k(client):
    r = client.post("/rank", json={"playlist_prefix": ["t1"], "top_k": 5, "constraint_weights": {"diversity": 1}})
    assert r.status_code == 200
    body = r.json()
    assert [t["rank"] for t in body["ranked_tracks"]] == [1, 2, 3, 4, 5]
    assert body["metadata"]["mock_scores"] is True
    assert body["metadata"]["constraint_weights_applied"] == {"diversity": 1.0}


def test_entry_point_targets_importable_module():
    """The console script runs uvicorn on 'serving.api:app'; that path must import."""
    import importlib
    import inspect

    assert '"serving.api:app"' in inspect.getsource(api.main)
    assert importlib.import_module("serving.api").app is api.app


class TestRerank:
    def test_no_weights_is_plain_sort(self):
        assert api._rerank([0.1, 0.9, 0.5], ["a", "b", "c"], set(), 3) == [1, 2, 0]

    def test_fatigue_demotes_recent_artists(self):
        scores = [3.0, 2.9, 1.0, 0.9]
        artists = ["recent", "recent", "new", "new"]
        assert api._rerank(scores, artists, set(), 2) == [0, 1]
        assert api._rerank(scores, artists, {"recent"}, 2, fatigue=10.0) == [2, 3]

    def test_diversity_spreads_artists(self):
        scores = [3.0, 2.9, 2.8, 1.0]
        artists = ["a", "a", "a", "b"]
        assert api._rerank(scores, artists, set(), 2) == [0, 1]
        assert api._rerank(scores, artists, set(), 2, diversity=10.0) == [0, 3]

    def test_handles_empty_and_constant_scores(self):
        assert api._rerank([], [], set(), 5) == []
        assert sorted(api._rerank([1.0, 1.0], ["a", "b"], set(), 2)) == [0, 1]

