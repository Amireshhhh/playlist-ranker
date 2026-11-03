#  Constraint-Aware Playlist Ranking System

> Playlist continuation built for Spotify's Million Playlist Dataset (MPD): a learning-to-rank model trained with differentiable artist-fatigue, diversity and exposure-fairness losses, plus a FastAPI service with inference-time re-ranking.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  What This Does

Given a partial playlist, the system retrieves candidate tracks and ranks them while trading relevance against:
-  **Artist fatigue**: repeating artists from the last few tracks
-  **Diversity**: concentrating the list on few artists
-  **Exposure fairness**: deviating from the artists' popularity prior

The trade-off is set in two places: loss weights at training time (`training/config.yaml`) and re-ranking weights at request time (`constraint_weights` in the API).

---

##  Quick Start

### Prerequisites
- Python 3.10 or newer (tested on 3.11 and 3.13)

### Installation

```bash
git clone https://github.com/Amireshhhh/playlist-ranker.git
cd playlist-ranker

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements-dev.txt   # runtime + test dependencies
pip install -e .
```

Or run everything up to feature extraction on sample data with `./quickstart.sh`.

### Data

The MPD must be downloaded by hand (free AIcrowd account):

```bash
python data/mpd_loader.py --download   # prints the download and extraction instructions
```

For a quick run without the MPD, generate synthetic sample data (1,000 playlists over a 400-track catalog):

```bash
python data/mpd_loader.py --sample
```

### Pipeline

```bash
# 1. Prefix/continuation pairs and track-artist mappings
python data/splits.py --input-dir ./data/mpd --output-dir ./data/processed --max-playlists 1000

# 2. Features (and the artist popularity prior)
python features/build_features.py --data-dir ./data/processed

# 3. Train: baseline without constraints, and the full model
python training/train.py --experiment baseline --no-constraints
python training/train.py --experiment full_constraints

# 4. Evaluate on the test split
python evaluation/evaluate.py \
  --checkpoint baseline=checkpoints/baseline/best_model.pt \
  --checkpoint full=checkpoints/full_constraints/best_model.pt \
  --fatigue 0.5 --diversity 0.5

# 5. Serve
uvicorn serving.api:app --host 0.0.0.0 --port 8000   # or: playlist-ranker
```

Drop `--max-playlists` to use the whole dataset. Interactive API docs are at http://localhost:8000/docs.

### Tests

```bash
pytest tests/
```

---

##  Playlist Insights Dashboards

`python -m insights` analyses a playlist and writes a static HTML dashboard per playlist, plus an index per user. It covers lyrics word counts, themes and sentiment, k-means song groups, an artist kNN graph, tempo flow, mood, artist origins, and recommendations from free APIs and this repo's ranker. Every source is free (LRCLIB, Deezer, MusicBrainz, Last.fm, ReccoBeats, Wikipedia), and the analysis runs locally.

```bash
pip install -r requirements-insights.txt
python -m insights demo                               # offline demo, synthetic data
python -m insights analyze exports/*.csv --user amir  # Exportify CSVs
```

See [docs/INSIGHTS.md](docs/INSIGHTS.md) for inputs, methods and data sources.

---

##  System Architecture

```
┌─────────────────┐
│ Playlist Prefix │
└────────┬────────┘
         ▼
┌─────────────────────┐
│ Candidate Retrieval │  ← top-100 by co-occurrence with the prefix
└────────┬────────────┘    (filled with the most popular tracks if needed)
         ▼
┌─────────────────────┐
│ Feature Engineering │  ← co-occurrence, artist repetition, recency,
└────────┬────────────┘    popularity, prefix diversity (32 features)
         ▼
┌─────────────────────┐
│   Ranking Model     │  ← PyTorch MLP trained with the multi-objective loss
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Greedy Re-ranking   │  ← optional fatigue / diversity weights per request
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Ranked Tracks      │  + playlist-health diagnostics
└─────────────────────┘
```

---

##  How the Constraints Work

Training is grouped: one **group** is the candidate list for one prefix, made of its continuation tracks (positives) and `num_negatives` random tracks per positive.

```
L_total = L_rank + λ₁·L_fatigue + λ₂·L_diversity + λ₃·L_exposure
```

- **L_rank**: pairwise margin loss over every (positive, negative) pair of the group.
- The constraint terms act on the group's **soft ranking distribution**
  `p = softmax(z / temperature)`, where `z` are the group's standardised scores, and on the soft
  artist distribution `q_a = Σ_{i: artist_i = a} p_i`:
  - **L_fatigue** = `Σ_i p_i · 1[artist_i in the last 5 prefix tracks]`
  - **L_diversity** = `−H(q)`
  - **L_exposure** = `Σ_a (q_a − π_a)²`, with `π` the artists' share of training occurrences, renormalised over the group's artists

Both choices matter for the constraints to have any effect. A penalty computed on a hard top-K list (argsort, counting) has zero gradient, so it cannot change the weights. Standardising the scores keeps the softmax from saturating as the margin loss grows them. `tests/test_model.py` checks that each term has a gradient and that training with a fatigue weight lowers the mass on recent-artist tracks.

**Leakage control.** Co-occurrence and popularity features for a training pair are computed with that pair's whole playlist removed from the statistics. Otherwise every positive counts itself, and the model learns a signal that unseen playlists do not have.

At **inference**, `constraint_weights` re-ranks greedily: each step picks the candidate maximising `z_i − fatigue·[artist in the last 5 prefix tracks] − diversity·(earlier picks of the same artist)`, with `z` the standardised model scores.

---

##  Results

Measured with the commands under "Pipeline" on the **synthetic sample data** (1,000 playlists, `--max-playlists 1000`), Python 3.11, CPU, test split: 1,148 prefixes, 100 candidates each, K = 10. Re-ranking weights: fatigue 0.5, diversity 0.5.

| Method | NDCG@10 | Recall@10 | MRR | Artist entropy@10 (bits) | Repeat rate@10 | Artist coverage@10 |
|---|---|---|---|---|---|---|
| Co-occurrence order (no model) | 0.266 | 0.196 | 0.477 | 2.70 | 0.088 | 0.73 |
| Baseline model (no constraints) | 0.380 | 0.272 | 0.611 | 1.40 | 0.263 | 0.30 |
| Baseline + re-ranking | 0.330 | 0.219 | 0.608 | 2.88 | 0.014 | 0.78 |
| Full-constraint model | 0.352 | 0.245 | 0.604 | 2.30 | 0.127 | 0.59 |
| Full-constraint + re-ranking | 0.207 | 0.184 | 0.326 | 3.27 | 0.001 | 0.97 |

Candidate retrieval finds 67.4% of the true continuation tracks; no ranker can score the rest.

The constraint losses trade relevance for diversity: against the baseline, NDCG@10 drops from 0.380 to 0.352 while artist entropy rises from 1.40 to 2.30 bits and the repeat rate falls from 0.263 to 0.127. With 10 tracks the entropy can be at most log₂ 10 ≈ 3.32 bits.

These numbers describe the synthetic sample only (10 artists, 400 tracks); **nothing has been measured on the real MPD yet**, and API latency/throughput have not been benchmarked. To reproduce or extend them, run `evaluation/evaluate.py` (add `--output results.json` to save them).

---

##  Project Structure

```
playlist-ranker/
├── data/
│   ├── mpd_loader.py         # MPD parsing, download instructions, synthetic sample data
│   └── splits.py             # Train/val/test prefix-continuation pairs + mappings
├── features/
│   └── build_features.py     # Features, co-occurrence retrieval, negative sampling
├── models/
│   ├── ranker.py             # PyTorch model and (differentiable) losses
│   └── rerank.py             # Inference-time greedy re-ranking
├── training/
│   ├── train.py              # Grouped training loop
│   └── config.yaml           # Hyperparameters
├── evaluation/
│   ├── metrics.py            # NDCG, MRR, Recall, playlist health
│   └── evaluate.py           # Offline evaluation on the test split
├── serving/
│   ├── api.py                # FastAPI application
│   └── Dockerfile            # Container image
├── insights/                 # playlist analysis + dashboards (python -m insights)
├── docs/INSIGHTS.md          # insights inputs, methods, free data sources
├── tests/                    # pytest suite (model, losses, metrics, API, insights)
├── requirements.txt          # runtime dependencies
├── requirements-dev.txt      # + test dependencies
├── quickstart.sh
├── setup.py
├── SYSTEM_SPEC.md            # Design specification
└── README.md
```

---

##  API Documentation

### Rank Endpoint

**POST** `/rank`

**Request Body:**
```json
{
  "playlist_prefix": ["spotify:track:...", "spotify:track:..."],
  "candidate_tracks": ["spotify:track:..."],
  "top_k": 10,
  "constraint_weights": {
    "fatigue": 0.5,
    "diversity": 0.5
  }
}
```

- `candidate_tracks`: optional; defaults to the top 100 tracks by co-occurrence with the prefix.
- `top_k`: 1–100.
- `constraint_weights`: optional; only `fatigue` and `diversity` are accepted (values ≥ 0, in units of score standard deviations). Other keys, negative values or an empty prefix return **400**.

**Response:**
```json
{
  "ranked_tracks": [
    {
      "track_uri": "spotify:track:...",
      "score": 0.74,
      "artist_uri": "spotify:artist:...",
      "rank": 1
    }
  ],
  "diagnostics": {
    "artist_entropy": 2.32,
    "repeat_rate": 0.0,
    "artist_coverage": 1.0,
    "gini_coefficient": 0.0
  },
  "metadata": {
    "num_candidates": 100,
    "processing_time_ms": 12.4,
    "prefix_length": 25,
    "model_version": "1.0.0",
    "model_trained": true,
    "mock_scores": false,
    "constraint_weights_applied": {"fatigue": 0.5, "diversity": 0.5}
  }
}
```
(Values are illustrative.)

### Health Check

**GET** `/health` returns `status`, `model_loaded`, `model_trained` (false when no checkpoint was found and an untrained model is in use), `feature_builder_loaded` and `device`.

### Configuration

| Variable | Default |
|---|---|
| `MODEL_PATH` | `./checkpoints/full_constraints/best_model.pt` |
| `DATA_DIR` | `./data/processed` |
| `HOST` / `PORT` (for `playlist-ranker`) | `0.0.0.0` / `8000` |

Without `DATA_DIR` data the API returns random mock scores and says so in `metadata.mock_scores`.

---

##  Docker

```bash
docker build -f serving/Dockerfile -t playlist-ranker .
docker run -p 8000:8000 \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/data/processed:/app/data/processed" \
  playlist-ranker
```

Build from the repository root. The checkpoint and processed data are mounted, not baked into the image. The image has not been built or tested yet.

---

##  Evaluation Metrics

### Ranking Quality
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Recall@K**: Fraction of the continuation tracks in the top K
- **Candidate recall**: Fraction of the continuation tracks the retrieval step found at all

### Playlist Health (of the top K)
- **Artist Coverage**: Unique artists / tracks
- **Artist Entropy**: −Σ p(a)·log₂ p(a), in bits
- **Repeat Rate**: Fraction of consecutive same-artist pairs
- **Gini Coefficient**: Inequality of artist appearance counts (API diagnostics)

---

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run `pytest tests/`
4. Commit and push, then open a Pull Request

---

##  License

MIT, see [LICENSE](LICENSE).
