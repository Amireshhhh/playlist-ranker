# Constraint-Aware Playlist Ranking System (Spotify-Scale)

## Executive Summary

A production-grade playlist continuation system built on the Spotify Million Playlist Dataset (MPD), implementing learning-to-rank with multi-objective constraints for diversity, artist fatigue control, and exposure fairness.

> **Status.** This is the design specification. Sections marked *planned* are not
> implemented in this repository. Measured results (on the synthetic sample data) are in
> README.md under "Results"; nothing has been measured on the full MPD yet.

**Targets (not yet measured on the full MPD):**
- NDCG@10 > 0.35
- Artist entropy of the top 10 > 3.0 bits (the maximum for 10 tracks is log2(10) = 3.32 bits)
- Repeat rate < 15%

---

## 1. Problem Statement

### Input
- Partial playlist (prefix of tracks)
- Candidate pool of continuation tracks

### Output
- Ranked list of recommended tracks
- Constraint diagnostic scores

### Constraints
1. **Artist Fatigue Control**: Limit consecutive same-artist tracks
2. **Diversity**: Maintain high artist/genre entropy in recommendations
3. **Exposure Fairness**: Balance popularity vs. long-tail artist exposure

---

## 2. Data

### Source
**Spotify Million Playlist Dataset (MPD)**
- 1,000,000 playlists
- 2,262,292 unique tracks
- 295,860 unique artists
- Anonymized, research-grade data

### Splits
- Train: 70% (700k playlists)
- Validation: 15% (150k playlists)
- Test: 15% (150k playlists)

### Preprocessing
- Prefix lengths: 5, 10, 25 tracks
- Continuation length: 10 tracks
- Candidate pool: 100 tracks per query

---

## 3. Architecture

### 3.1 Candidate Generation

**Co-occurrence Retrieval** (implemented: `FeatureBuilder.candidates_from_cooccurrence`)
- Top-N tracks by summed co-occurrence count with the prefix tracks in the training data
- Filled with the most frequent training tracks if fewer than N are found

**Negatives** (implemented)
- Uniform random tracks, excluding the pair's prefix and continuation, `num_negatives` per positive

*Planned:* popular-in-context retrieval, recency weighting, hard negatives from failed retrievals.

**Leakage control** (implemented): co-occurrence and frequency features for training pairs
are computed with the pair's whole playlist left out of the statistics.

### 3.2 Ranking Model

**Feature Engineering**
```
Per (playlist_prefix, track) pair:
- Track co-occurrence frequency
- Artist repetition count in prefix
- Position-aware features (track order)
- Popularity proxy (global frequency)
- Playlist length normalization
- Recency features
```

**Model Architecture**
```
Input: Feature vector (dim=32)
↓
Dense(128) + ReLU + Dropout(0.3)
↓
Dense(64) + ReLU + Dropout(0.21)
↓
Dense(32) + ReLU
↓
Dense(1) → Relevance Score
```

Input features are standardised with the training-set mean and standard deviation; the
statistics are stored in the model's state dict.

**Framework**: PyTorch
**Training**: Pairwise ranking loss (margin-based) over all positive x negative pairs of a group

### 3.3 Constraint Integration

**Multi-Objective Loss**
```
L_total = L_rank + λ₁·L_fatigue + λ₂·L_diversity + λ₃·L_exposure
```

**Component Losses**

Training is grouped: one group is the candidate list for one prefix (its continuation
tracks as positives plus sampled negatives). The constraints act on the group's soft
ranking distribution

```
z = (s - mean(s)) / std(s)          standardised scores of the group
p = softmax(z / temperature)
q_a = Σ_{i: artist_i = a} p_i       soft artist distribution
```

which is differentiable in the scores. (Hard top-K selection has zero gradient, so a
penalty computed on it cannot change the model. Standardising keeps softmax from
saturating as the scores grow.)

1. **Ranking Loss** (Pairwise Margin)
   ```
   L_rank = mean over (pos, neg) pairs of max(0, margin - (score_pos - score_neg))
   ```

2. **Fatigue Penalty**
   ```
   L_fatigue = Σ_i p_i · 𝟙[artist_i ∈ last 5 prefix tracks]
   ```

3. **Diversity Penalty**
   ```
   L_diversity = -H(q) = Σ_a q_a · log q_a
   ```

4. **Exposure Fairness**
   ```
   L_exposure = Σ_a (q_a - π_a)²
   π_a = artist popularity prior (share of training occurrences), renormalised over the group's artists
   ```

**Hyperparameters**
- λ₁ = 0.1 (fatigue)
- λ₂ = 0.05 (diversity)
- λ₃ = 0.03 (exposure)

---

## 4. Training

### Configuration (`training/config.yaml`)
- Batch size: 64 groups
- Learning rate: 1e-3 (Adam), halved after 3 epochs without validation-loss improvement
- Epochs: up to 50
- Early stopping: patience=5 on validation total loss
- Gradient clipping: max_norm=1.0

### Monitoring (implemented)
- Every loss component on train and validation
- Validation NDCG@10 over each group's candidates

*Planned:* random prefix truncation, per-epoch negative resampling, hard negative mining.

---

## 5. Evaluation

### Ranking Metrics
- **NDCG@10**: Primary metric (information retrieval quality)
- **MRR**: First relevant track position
- **Recall@K**: Coverage of ground truth

### Playlist Health Metrics
- **Artist Coverage**: Unique artists in top-K
- **Artist Entropy**: H(artist distribution)
- **Repeat Rate**: % consecutive same-artist tracks
- **Popularity Bias**: Gini coefficient of artist exposure

### Offline Evaluation (implemented: `evaluation/evaluate.py`)
Ranks the retrieved candidates of every test pair with each checkpoint (and optionally
with inference-time re-ranking) and reports NDCG@K, Recall@K, MRR, candidate recall and
the playlist-health metrics of the top K. The co-occurrence order is reported as a baseline.

### Ablation Studies
1. Baseline (no constraints): `train.py --experiment baseline --no-constraints`
2. Full model (all constraints): `train.py --experiment full_constraints`
3. *Planned:* single-constraint runs (set the other λ to 0 in a config copy)

### Sensitivity Analysis (*planned*)
- λ parameter sweeps
- Prefix length impact
- Candidate pool size effects

---

## 6. Deployment

### API Specification

**Endpoint**: `POST /rank`

**Request**
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

`candidate_tracks` is optional (default: top 100 by co-occurrence). `constraint_weights`
is optional and drives a greedy re-ranking at inference time; only `fatigue` and
`diversity` are supported (other keys return 400).

**Response**
```json
{
  "ranked_tracks": [
    {
      "track_uri": "spotify:track:...",
      "score": 1.93,
      "artist_uri": "spotify:artist:...",
      "rank": 1
    }
  ],
  "diagnostics": {
    "artist_entropy": 2.65,
    "repeat_rate": 0.11,
    "artist_coverage": 0.7,
    "gini_coefficient": 0.21
  },
  "metadata": {
    "num_candidates": 100,
    "processing_time_ms": 12.4,
    "prefix_length": 2,
    "model_version": "1.0.0",
    "model_trained": true,
    "mock_scores": false,
    "constraint_weights_applied": {"fatigue": 0.5, "diversity": 0.5}
  }
}
```
(Values are illustrative.)

### Infrastructure
- **Framework**: FastAPI (implemented)
- **Model Serving**: PyTorch (implemented); ONNX Runtime export is *planned*
- **Containerization**: Docker (`serving/Dockerfile`)
- **Monitoring**: Prometheus + Grafana (*planned*)

### Performance Requirements (targets, not yet measured)
- Latency: <100ms (p95)
- Throughput: >100 QPS
- Model size: <50MB

---

## 7. Repository Structure

```
playlist-ranker/
├── data/
│   ├── mpd_loader.py          # MPD parsing and synthetic sample data
│   └── splits.py              # Train/val/test prefix-continuation pairs
├── features/
│   └── build_features.py      # Features, co-occurrence retrieval, negatives
├── models/
│   ├── ranker.py              # PyTorch ranking model and losses
│   └── rerank.py              # Inference-time greedy re-ranking
├── training/
│   ├── train.py               # Training loop
│   └── config.yaml            # Hyperparameters
├── evaluation/
│   ├── metrics.py             # NDCG, MRR, Recall, playlist health
│   └── evaluate.py            # Offline evaluation on the test split
├── serving/
│   ├── api.py                 # FastAPI service
│   └── Dockerfile             # Container image
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── requirements.txt / requirements-dev.txt
├── setup.py
├── SYSTEM_SPEC.md             # This document
└── README.md
```

*Planned:* `notebooks/`, MLflow/W&B tracking, hard-negative mining module.

---

## 8. Technical Decisions & Justifications

### Why No Audio Features?
- **Simplicity**: Co-occurrence signals capture playlist coherence
- **Speed**: No embedding computation bottleneck
- **Interpretability**: Features are human-understandable
- **Industry Reality**: Many production systems start here

### Why Pairwise Ranking Loss?
- Directly optimizes rank order
- Handles implicit feedback (playlist continuations)
- Robust to label noise

### Why Offline Dataset?
- **Reproducibility**: No API rate limits
- **Scale**: 1M playlists is production-grade
- **Credibility**: Spotify's official research dataset

---

## 9. Success Criteria

### MVP (Minimum Viable Product)
- NDCG@10 > 0.30
- API latency < 200ms
- Deployable Docker container

### Target Performance
- 🎯 NDCG@10 > 0.35
- 🎯 Artist entropy (top 10) > 3.0 bits
- 🎯 Repeat rate < 15%
- 🎯 API latency < 100ms

### Stretch Goals
- 🚀 NDCG@10 > 0.40
- 🚀 Real-time constraint tuning
- 🚀 A/B testing framework

---

## 10. Timeline

### Week 1: Data & Infrastructure
- Download & parse MPD
- Generate train/val/test splits
- Implement feature extraction

### Week 2-3: Modeling
- Baseline ranker (no constraints)
- Constraint integration
- Hyperparameter tuning

### Week 4-5: Evaluation & Iteration
- Ablation studies
- Sensitivity analysis
- Performance optimization

### Week 6: Deployment
- FastAPI implementation
- Docker containerization
- Documentation & testing

### Week 7-8: Polish
- Demo interface (Streamlit)
- Presentation materials
- Resume integration

---

## 11. Resume Impact

**One-Liner**
> Designed and deployed a constraint-aware playlist ranking system using Spotify-scale data (1M playlists), balancing relevance, diversity, and exposure fairness with multi-objective learning-to-rank.

**Bullet Points** (fill in the numbers from `evaluation/evaluate.py` on the full MPD; none
of these have been measured yet)
- Built a music recommendation system on Spotify's Million Playlist Dataset (1M playlists, 2M+ tracks)
- Implemented multi-objective learning-to-rank in PyTorch with differentiable fatigue,
  diversity and exposure-fairness losses, trading relevance (NDCG@10: ___) against
  artist diversity (entropy: ___ bits)
- Served the ranker through a FastAPI service with inference-time re-ranking (p95 latency: ___ ms)
- Compared constrained and unconstrained models offline (diversity change: ___ at an NDCG@10 change of ___)

---

## 12. References

- **Dataset**: [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
- **Learning to Rank**: Liu, TY. "Learning to Rank for Information Retrieval"
- **Diversity**: Carbonell & Goldstein. "Maximal Marginal Relevance"
- **Fairness**: Biega et al. "Equity of Attention in Rankings"

---

**Document Version**: 1.0  
**Last Updated**: November 2026  
**Author**: Amirreza Eshraghi  
**Contact**: https://github.com/Amireshhhh
