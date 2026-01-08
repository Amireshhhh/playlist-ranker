# Getting Started with Playlist Ranker

## 🎯 Overview

This guide will walk you through setting up and running the Constraint-Aware Playlist Ranking System from scratch.

## 📋 Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum
- 20GB disk space (for full dataset) or 1GB (for sample)
- Git (for cloning repository)

## 🚀 Installation

### Option 1: Quick Start (Recommended for Demo)

```bash
# Clone repository
git clone https://github.com/yourusername/playlist-ranker.git
cd playlist-ranker

# Run quick start script
chmod +x quickstart.sh
./quickstart.sh
```

This will:
- Create virtual environment
- Install dependencies
- Generate sample data (1000 playlists)
- Create train/val/test splits
- Build features

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Generate sample data
python data/mpd_loader.py --sample

# Create splits
python data/splits.py --input-dir ./data/mpd --output-dir ./data/processed --max-playlists 1000

# Build features
python features/build_features.py --data-dir ./data/processed
```

## 📊 Using Full Spotify Dataset (Optional)

For production-scale experiments:

1. **Download Dataset**
   - Visit: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
   - Create account (free)
   - Download: `spotify_million_playlist_dataset.zip` (~5GB)

2. **Extract Data**
   ```bash
   unzip spotify_million_playlist_dataset.zip -d ./data/mpd
   ```

3. **Process Full Dataset**
   ```bash
   # This will take 10-30 minutes
   python data/splits.py --input-dir ./data/mpd --output-dir ./data/processed
   python features/build_features.py --data-dir ./data/processed
   ```

## 🏋️ Training Models

### Train Baseline (No Constraints)

```bash
python training/train.py \
  --experiment baseline \
  --no-constraints \
  --data-dir ./data/processed \
  --checkpoint-dir ./checkpoints
```

**Expected output:**
```
Epoch 1/50
  Train - Total: 0.4523, Rank: 0.4523
  Val   - Total: 0.4201, Rank: 0.4201
...
✓ Saved best model (val_loss: 0.3801)
```

### Train Full Model (With Constraints)

```bash
python training/train.py \
  --experiment full_constraints \
  --data-dir ./data/processed \
  --checkpoint-dir ./checkpoints
```

**Expected output:**
```
Epoch 1/50
  Train - Total: 0.4821, Rank: 0.4312, Fatigue: 0.0321, Diversity: 0.0188
  Val   - Total: 0.4512, Rank: 0.4021, Fatigue: 0.0298, Diversity: 0.0193
...
```

**Training time:**
- Sample data (1000 playlists): ~5-10 minutes
- Full dataset (1M playlists): ~2-4 hours (GPU) or ~8-12 hours (CPU)

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_model.py::TestRankingModel::test_model_creation PASSED
tests/test_model.py::TestRankingModel::test_forward_pass PASSED
...
========== 15 passed in 2.34s ==========
```

## 🌐 Running API Server

### Start Server

```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

Server will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Test API

**Using curl:**

```bash
curl -X POST "http://localhost:8000/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "playlist_prefix": ["track_1", "track_2", "track_3"],
    "top_k": 10
  }'
```

**Using Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/rank",
    json={
        "playlist_prefix": ["track_1", "track_2", "track_3"],
        "top_k": 10
    }
)

print(response.json())
```

**Expected response:**

```json
{
  "ranked_tracks": [
    {
      "track_uri": "spotify:track:abc123",
      "score": 0.87,
      "artist_uri": "spotify:artist:xyz456",
      "rank": 1
    }
  ],
  "diagnostics": {
    "artist_entropy": 3.72,
    "repeat_rate": 0.12,
    "artist_coverage": 0.80
  },
  "metadata": {
    "num_candidates": 100,
    "processing_time_ms": 45.2
  }
}
```

## 📓 Running Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/demo.ipynb
```

The demo notebook includes:
- Data exploration
- Feature visualization
- Model inference
- Evaluation metrics
- Playlist health analysis

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t playlist-ranker:latest -f serving/Dockerfile .

# Run container
docker run -p 8000:8000 playlist-ranker:latest

# Test
curl http://localhost:8000/health
```

## 📈 Evaluation

Evaluate trained model:

```python
from evaluation.metrics import EvaluationSuite
import json

# Load test data
with open('./data/processed/test_features.json', 'r') as f:
    test_data = json.load(f)

# Initialize evaluator
evaluator = EvaluationSuite()

# Run evaluation
# (see notebooks/demo.ipynb for complete example)
```

## 🎨 Customization

### Adjust Constraint Weights

Edit `training/config.yaml`:

```yaml
lambda_fatigue: 0.1      # Increase to penalize artist repetition more
lambda_diversity: 0.05   # Increase for more diversity
lambda_exposure: 0.03    # Increase for more fairness
```

### Change Model Architecture

Edit `training/config.yaml`:

```yaml
hidden_dims: [256, 128, 64]  # Larger model
dropout: 0.4                  # More regularization
```

### Modify Features

Edit `features/build_features.py` to add custom features.

## 🐛 Troubleshooting

### Issue: "No JSON files found"

**Solution:** Run data generation:
```bash
python data/mpd_loader.py --sample
```

### Issue: "Model not found"

**Solution:** Train model first:
```bash
python training/train.py --experiment baseline
```

### Issue: "Out of memory"

**Solution:** Reduce batch size in `training/config.yaml`:
```yaml
batch_size: 256  # Default is 512
```

### Issue: "API returns 500 error"

**Solution:** Check logs and ensure model is trained:
```bash
python training/train.py --experiment full_constraints
```

## 📊 Expected Performance

### Sample Data (1000 playlists)
- NDCG@10: ~0.30-0.35
- Training time: 5-10 minutes
- API latency: <50ms

### Full Dataset (1M playlists)
- NDCG@10: ~0.35-0.40
- Training time: 2-4 hours (GPU)
- API latency: <100ms

## 📝 Next Steps

1. **Experiment with constraints:** Try different λ values
2. **Add features:** Extend feature engineering
3. **Deploy:** Use Docker for production
4. **Monitor:** Add MLflow/W&B tracking
5. **Scale:** Implement batch prediction

## 🤝 Contributing

See CONTRIBUTING.md (to be created) for guidelines.

## 📧 Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/playlist-ranker/issues
- Email: your.email@example.com

## 📄 License

MIT License - see LICENSE file.

---

**Built with ❤️ for production-grade music recommendation**
