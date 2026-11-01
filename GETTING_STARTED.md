# Getting Started with Playlist Ranker

## 🎯 Overview

This guide walks you through setting up and running the Constraint-Aware Playlist Ranking System from scratch: data, features, training, evaluation and the API.

## 📋 Prerequisites

- Python 3.10 or newer (tested on 3.11 and 3.13)
- Git

## 🚀 Installation

### Option 1: Quick Start (sample data)

```bash
git clone https://github.com/Amireshhhh/playlist-ranker.git
cd playlist-ranker

chmod +x quickstart.sh
./quickstart.sh
```

This will:
- Create a virtual environment (`venv/`)
- Install the dependencies and the package
- Generate synthetic sample data (1,000 playlists over a 400-track catalog)
- Create train/val/test splits
- Build features

### Option 2: Manual Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements-dev.txt
pip install -e .

python data/mpd_loader.py --sample
python data/splits.py --input-dir ./data/mpd --output-dir ./data/processed --max-playlists 1000
python features/build_features.py --data-dir ./data/processed
```

## 📊 Using the Full Spotify Dataset

1. **Download** the dataset from https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
   (free account; `python data/mpd_loader.py --download` prints the steps).

2. **Extract** it into `./data/mpd`:
   ```bash
   unzip spotify_million_playlist_dataset.zip -d ./data/mpd
   ```
   Remove the sample file `data/mpd/mpd.slice.0-999.json` first if you generated sample data:
   the loader reads every `*.json` file under `data/mpd`.

3. **Process** it:
   ```bash
   python data/splits.py --input-dir ./data/mpd --output-dir ./data/processed
   python features/build_features.py --data-dir ./data/processed
   ```
   Use `--max-playlists N` on `splits.py` to work with a subset first.

## 🏋️ Training Models

### Baseline (no constraints)

```bash
python training/train.py --experiment baseline --no-constraints
```

### Full model (with constraints)

```bash
python training/train.py --experiment full_constraints
```

Checkpoints go to `checkpoints/<experiment>/best_model.pt` (best validation loss). Each epoch logs every loss component and the validation NDCG@10, for example:

```
INFO:__main__:Epoch 2/50  lr=1.00e-03
INFO:__main__:  Train - Total: 0.2465, Rank: 0.2823, Fatigue: 0.5743, Diversity: -1.9173, Exposure: 0.0894
INFO:__main__:  Val   - Total: 0.1749, Rank: 0.2083, Fatigue: 0.5986, Diversity: -1.9169, Exposure: 0.0842, NDCG@10: 0.6140
```

Use `--epochs N` to override the configured number of epochs.

## 📈 Evaluation

```bash
python evaluation/evaluate.py \
  --checkpoint baseline=checkpoints/baseline/best_model.pt \
  --checkpoint full=checkpoints/full_constraints/best_model.pt \
  --fatigue 0.5 --diversity 0.5 \
  --output results.json
```

For each test prefix this retrieves 100 candidates by co-occurrence, ranks them with every checkpoint (and with re-ranking when `--fatigue`/`--diversity` are set), and reports NDCG@10, Recall@10, MRR and the playlist-health metrics of the top 10. The README's "Results" section shows the numbers for the sample data.

## 🧪 Testing

```bash
pytest tests/
```

## 🌐 Running the API Server

```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000
# or, after pip install -e .:
playlist-ranker
```

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

The server loads `checkpoints/full_constraints/best_model.pt` and `data/processed/` (override with `MODEL_PATH` and `DATA_DIR`). `/health` reports `"model_trained": false` if no checkpoint was found.

### Test the API

Use track URIs that exist in your data (for the sample data, e.g. `spotify:track:artist_3_0`):

```bash
curl -X POST "http://localhost:8000/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "playlist_prefix": ["spotify:track:artist_3_0", "spotify:track:artist_3_1"],
    "top_k": 10,
    "constraint_weights": {"fatigue": 0.5, "diversity": 0.5}
  }'
```

## 🐳 Docker Deployment

```bash
docker build -f serving/Dockerfile -t playlist-ranker .
docker run -p 8000:8000 \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/data/processed:/app/data/processed" \
  playlist-ranker
curl http://localhost:8000/health
```

## 🎨 Customization

### Constraint weights (training)

`training/config.yaml`:

```yaml
lambda_fatigue: 0.1      # mass on artists from the last 5 prefix tracks
lambda_diversity: 0.05   # negative entropy of the soft artist distribution
lambda_exposure: 0.03    # distance from the artist popularity prior
temperature: 1.0         # softmax temperature of the soft ranking distribution
```

### Constraint weights (inference)

Pass `constraint_weights` (`fatigue`, `diversity`) to `/rank`, or `--fatigue` / `--diversity` to `evaluation/evaluate.py`.

### Model architecture

```yaml
hidden_dims: [256, 128, 64]
dropout: 0.4
```

### Features

Edit `FeatureBuilder.extract_features` in `features/build_features.py`; keep the vector at `input_dim` (32) values or change `input_dim` in the config.

## 🐛 Troubleshooting

**"No JSON files found"**: generate sample data (`python data/mpd_loader.py --sample`) or extract the MPD into `data/mpd`.

**`FileNotFoundError: ... artist_popularity.json` when training**: run `python features/build_features.py` first.

**`/health` shows `"model_trained": false`**: train a model (`python training/train.py --experiment full_constraints`) or set `MODEL_PATH`.

**Out of memory**: lower `batch_size` (groups per batch) in `training/config.yaml`, or use `--max-playlists`.

## 📄 License

MIT License, see LICENSE.
