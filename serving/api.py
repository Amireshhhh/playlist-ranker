"""
FastAPI Application for Playlist Ranking Service

Production-ready API for serving ranking predictions.
"""

import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional
import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.ranker import RankingModel
from models.rerank import rerank as _rerank
from features.build_features import FeatureBuilder
from evaluation.metrics import PlaylistHealthMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(_app):
    await load_model()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Playlist Ranking API",
    description="Constraint-aware playlist continuation with diversity and fairness",
    version="1.0.0"
)

# Paths can be overridden with environment variables.
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "./checkpoints/full_constraints/best_model.pt"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data/processed"))

# Constraint weights accepted by /rank at inference time (see _rerank).
SUPPORTED_CONSTRAINTS = {"fatigue", "diversity"}

# Global model and feature builder (loaded on startup)
model = None
model_trained = False
feature_builder = None
device = None


class Track(BaseModel):
    """Track representation."""
    track_uri: str
    artist_uri: Optional[str] = None
    pos: Optional[int] = None


class RankRequest(BaseModel):
    """Request schema for ranking endpoint."""
    playlist_prefix: List[str] = Field(..., description="List of track URIs in the prefix")
    candidate_tracks: Optional[List[str]] = Field(None, description="Candidate track URIs (if None, use top-100 from co-occurrence)")
    top_k: int = Field(10, description="Number of tracks to return", ge=1, le=100)
    constraint_weights: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Optional re-ranking weights, keys 'fatigue' and/or 'diversity' (>= 0). "
            "fatigue penalises artists from the last 5 prefix tracks; diversity penalises "
            "each earlier pick of the same artist. Weights are in units of score standard deviations."
        )
    )


class RankedTrack(BaseModel):
    """Ranked track with score and metadata."""
    track_uri: str
    score: float
    artist_uri: Optional[str] = None
    rank: int


class Diagnostics(BaseModel):
    """Playlist health diagnostics."""
    artist_entropy: float
    repeat_rate: float
    artist_coverage: float
    gini_coefficient: Optional[float] = None


class RankResponse(BaseModel):
    """Response schema for ranking endpoint."""
    ranked_tracks: List[RankedTrack]
    diagnostics: Diagnostics
    metadata: Dict


async def load_model():
    """Load model and feature builder on startup."""
    global model, model_trained, feature_builder, device
    
    logger.info("Loading model and feature builder...")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    if MODEL_PATH.exists():
        logger.info(f"Loading model from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        cfg = checkpoint.get('config', {})

        model = RankingModel(
            input_dim=cfg.get('input_dim', 32),
            hidden_dims=cfg.get('hidden_dims', [128, 64, 32]),
            dropout=cfg.get('dropout', 0.3),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model_trained = True
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}; using an UNTRAINED model (scores are meaningless)")
        model = RankingModel(input_dim=32, hidden_dims=[128, 64, 32])
        model_trained = False
    model.to(device)
    model.eval()

    # Load feature builder
    data_dir = DATA_DIR
    if (data_dir / "track_to_artist.json").exists():
        feature_builder = FeatureBuilder(data_dir=str(data_dir))
        logger.info("Feature builder loaded")
    else:
        logger.warning(f"Data directory not found: {data_dir}")
        logger.info("API will return mock responses")
    
    logger.info("Startup complete")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Playlist Ranking API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/rank": "POST - Rank tracks for playlist continuation",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_trained": model_trained,
        "feature_builder_loaded": feature_builder is not None,
        "device": str(device)
    }


@app.post("/rank", response_model=RankResponse)
async def rank_tracks(request: RankRequest):
    """
    Rank candidate tracks for playlist continuation.
    
    Args:
        request: RankRequest with prefix and optional candidates
        
    Returns:
        RankResponse with ranked tracks and diagnostics
    """
    start_time = time.time()

    # Validate inputs (outside the try block, so these stay 400s)
    if not request.playlist_prefix:
        raise HTTPException(status_code=400, detail="playlist_prefix cannot be empty")
    weights = request.constraint_weights or {}
    unknown = set(weights) - SUPPORTED_CONSTRAINTS
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported constraint_weights keys: {sorted(unknown)}. Supported: {sorted(SUPPORTED_CONSTRAINTS)}",
        )
    if any(v < 0 for v in weights.values()):
        raise HTTPException(status_code=400, detail="constraint_weights must be >= 0")

    try:
        artist_of = (lambda t: feature_builder.track_to_artist.get(t, 'unknown')) if feature_builder else (lambda t: 'unknown')

        prefix = [
            {'track_uri': track_uri, 'artist_uri': artist_of(track_uri), 'pos': i}
            for i, track_uri in enumerate(request.playlist_prefix)
        ]

        # Get candidates
        if request.candidate_tracks:
            candidates = list(dict.fromkeys(request.candidate_tracks))  # de-duplicate, keep order
        elif feature_builder:
            candidates = feature_builder.candidates_from_cooccurrence(request.playlist_prefix, top_n=100)
        else:
            candidates = [f"track_{i}" for i in range(100)]  # mock candidates

        # Score all candidates in one forward pass
        if feature_builder and candidates:
            features = np.stack([feature_builder.extract_features(prefix, c) for c in candidates])
            with torch.no_grad():
                scores = model(torch.from_numpy(features).to(device)).squeeze(-1).cpu().tolist()
            mock = False
        else:
            scores = np.random.rand(len(candidates)).tolist()
            mock = True

        recent_artists = {t['artist_uri'] for t in prefix[-5:]}
        top_k = min(request.top_k, len(candidates))
        order = _rerank(
            scores,
            [artist_of(c) for c in candidates],
            recent_artists,
            top_k,
            fatigue=weights.get('fatigue', 0.0),
            diversity=weights.get('diversity', 0.0),
        )

        ranked_tracks = [
            RankedTrack(track_uri=candidates[i], score=scores[i], artist_uri=artist_of(candidates[i]), rank=r + 1)
            for r, i in enumerate(order)
        ]

        artist_ids = [t.artist_uri for t in ranked_tracks]
        diagnostics = Diagnostics(
            artist_entropy=PlaylistHealthMetrics.artist_entropy(artist_ids),
            repeat_rate=PlaylistHealthMetrics.repeat_rate(artist_ids),
            artist_coverage=PlaylistHealthMetrics.artist_coverage(artist_ids),
            gini_coefficient=(
                float(PlaylistHealthMetrics.gini_coefficient(list(Counter(artist_ids).values())))
                if artist_ids else None
            ),
        )

        metadata = {
            "num_candidates": len(candidates),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "prefix_length": len(request.playlist_prefix),
            "model_version": "1.0.0",
            "model_trained": model_trained,
            "mock_scores": mock,
            "constraint_weights_applied": {k: float(v) for k, v in weights.items()},
        }

        return RankResponse(ranked_tracks=ranked_tracks, diagnostics=diagnostics, metadata=metadata)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in rank_tracks")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def main():
    """Main entry point for running the API."""
    import uvicorn
    
    logger.info("Starting Playlist Ranking API...")
    
    uvicorn.run(
        "serving.api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
