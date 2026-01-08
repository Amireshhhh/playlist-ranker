"""
FastAPI Application for Playlist Ranking Service

Production-ready API for serving ranking predictions.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.ranker import RankingModel
from features.build_features import FeatureBuilder
from evaluation.metrics import PlaylistHealthMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Playlist Ranking API",
    description="Constraint-aware playlist continuation with diversity and fairness",
    version="1.0.0"
)

# Global model and feature builder (loaded on startup)
model = None
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
        description="Custom constraint weights {fatigue, diversity, exposure}"
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


@app.on_event("startup")
async def load_model():
    """Load model and feature builder on startup."""
    global model, feature_builder, device
    
    logger.info("Loading model and feature builder...")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    model_path = Path("./checkpoints/full_constraints/best_model.pt")
    
    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        model = RankingModel(input_dim=32, hidden_dims=[128, 64, 32])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model not found at {model_path}")
        logger.info("Creating dummy model for demonstration")
        
        model = RankingModel(input_dim=32, hidden_dims=[128, 64, 32])
        model.to(device)
        model.eval()
    
    # Load feature builder
    data_dir = Path("./data/processed")
    if data_dir.exists():
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
    
    try:
        # Validate inputs
        if not request.playlist_prefix:
            raise HTTPException(status_code=400, detail="playlist_prefix cannot be empty")
        
        # Convert prefix to track dicts
        prefix = [
            {
                'track_uri': track_uri,
                'artist_uri': feature_builder.track_to_artist.get(track_uri, 'unknown') if feature_builder else 'unknown',
                'pos': i
            }
            for i, track_uri in enumerate(request.playlist_prefix)
        ]
        
        # Get candidates
        if request.candidate_tracks:
            candidates = request.candidate_tracks
        else:
            # Use co-occurrence to get candidates
            if feature_builder:
                candidates = _get_candidates_from_cooccurrence(
                    request.playlist_prefix,
                    feature_builder,
                    top_n=100
                )
            else:
                # Mock candidates
                candidates = [f"track_{i}" for i in range(100)]
        
        # Extract features and score
        scores = []
        
        if model and feature_builder:
            for candidate in candidates:
                features = feature_builder.extract_features(prefix, candidate)
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    score = model(features_tensor).item()
                
                scores.append(score)
        else:
            # Mock scores
            scores = np.random.rand(len(candidates)).tolist()
        
        # Rank by score
        ranked_indices = np.argsort(scores)[::-1]
        
        # Get top-K
        top_k = min(request.top_k, len(candidates))
        top_indices = ranked_indices[:top_k]
        
        # Build response
        ranked_tracks = []
        for rank, idx in enumerate(top_indices):
            track_uri = candidates[idx]
            artist_uri = feature_builder.track_to_artist.get(track_uri, 'unknown') if feature_builder else 'unknown'
            
            ranked_tracks.append(RankedTrack(
                track_uri=track_uri,
                score=scores[idx],
                artist_uri=artist_uri,
                rank=rank + 1
            ))
        
        # Compute diagnostics
        artist_ids = [t.artist_uri for t in ranked_tracks]
        
        diagnostics = Diagnostics(
            artist_entropy=PlaylistHealthMetrics.artist_entropy(artist_ids),
            repeat_rate=PlaylistHealthMetrics.repeat_rate(artist_ids),
            artist_coverage=PlaylistHealthMetrics.artist_coverage(artist_ids),
            gini_coefficient=None  # Can be computed if needed
        )
        
        # Metadata
        processing_time_ms = (time.time() - start_time) * 1000
        
        metadata = {
            "num_candidates": len(candidates),
            "processing_time_ms": round(processing_time_ms, 2),
            "prefix_length": len(request.playlist_prefix),
            "model_version": "1.0.0"
        }
        
        return RankResponse(
            ranked_tracks=ranked_tracks,
            diagnostics=diagnostics,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Error in rank_tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_candidates_from_cooccurrence(
    prefix_tracks: List[str],
    feature_builder: FeatureBuilder,
    top_n: int = 100
) -> List[str]:
    """
    Get candidate tracks using co-occurrence.
    
    Args:
        prefix_tracks: Tracks in the prefix
        feature_builder: FeatureBuilder instance
        top_n: Number of candidates to return
        
    Returns:
        List of candidate track URIs
    """
    candidate_scores = {}
    
    for prefix_track in prefix_tracks:
        cooccur_tracks = feature_builder.track_cooccurrence.get(prefix_track, {})
        
        for track, count in cooccur_tracks.items():
            if track not in prefix_tracks:
                candidate_scores[track] = candidate_scores.get(track, 0) + count
    
    # Sort by score and take top-N
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    candidates = [track for track, score in sorted_candidates[:top_n]]
    
    # Fill up to top_n with random tracks if needed
    if len(candidates) < top_n and feature_builder.track_frequencies:
        all_tracks = list(feature_builder.track_frequencies.keys())
        additional_needed = top_n - len(candidates)
        
        candidate_set = set(candidates) | set(prefix_tracks)
        additional_tracks = [t for t in all_tracks if t not in candidate_set][:additional_needed]
        candidates.extend(additional_tracks)
    
    return candidates


def main():
    """Main entry point for running the API."""
    import uvicorn
    
    logger.info("Starting Playlist Ranking API...")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
