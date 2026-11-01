"""
Offline evaluation on the test split.

For every test (prefix, continuation) pair:
  1. retrieve `--num-candidates` candidates by co-occurrence with the prefix
     (the same retrieval the API uses);
  2. rank them with each method below;
  3. score the top K against the true continuation.

Methods:
  * cooccurrence      - retrieval order only (no model); a baseline
  * model:<name>      - each checkpoint passed with --checkpoint name=path
  * ...+rerank        - the same model followed by greedy re-ranking with
                        --fatigue / --diversity weights (if either is > 0)

Metrics (means over test pairs):
  ndcg@K, recall@K, mrr                   relevance = track is in the continuation
  candidate_recall                        share of continuation tracks retrieved at all
  artist_entropy@K (bits), repeat_rate@K, artist_coverage@K   of the top-K list

Usage:
  python evaluation/evaluate.py --data-dir ./data/processed \\
      --checkpoint full=checkpoints/full_constraints/best_model.pt \\
      --checkpoint baseline=checkpoints/baseline/best_model.pt
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import PlaylistHealthMetrics, RankingMetrics  # noqa: E402
from features.build_features import FeatureBuilder  # noqa: E402
from models.ranker import RankingModel  # noqa: E402
from models.rerank import rerank  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(path: str, device: str) -> RankingModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint.get('config', {})
    model = RankingModel(
        input_dim=cfg.get('input_dim', 32),
        hidden_dims=cfg.get('hidden_dims', [128, 64, 32]),
        dropout=cfg.get('dropout', 0.3),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval()


def list_metrics(ranked: List[str], truth: set, artist_of, k: int) -> Dict[str, float]:
    relevance = [1 if t in truth else 0 for t in ranked]
    artists = [artist_of(t) for t in ranked[:k]]
    return {
        f'ndcg@{k}': RankingMetrics.ndcg_at_k(relevance, k),
        f'recall@{k}': RankingMetrics.recall_at_k(relevance, k, len(truth)),
        'mrr': RankingMetrics.mrr(relevance),
        f'artist_entropy@{k}': PlaylistHealthMetrics.artist_entropy(artists),
        f'repeat_rate@{k}': PlaylistHealthMetrics.repeat_rate(artists),
        f'artist_coverage@{k}': PlaylistHealthMetrics.artist_coverage(artists),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate rankers on the test split")
    parser.add_argument("--data-dir", default="./data/processed")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                        help="Model checkpoint to evaluate; repeatable")
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fatigue", type=float, default=0.0, help="Re-ranking fatigue weight")
    parser.add_argument("--diversity", type=float, default=0.0, help="Re-ranking diversity weight")
    parser.add_argument("--output", default=None, help="Optional path for the JSON results")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    builder = FeatureBuilder(data_dir=args.data_dir)
    artist_of = lambda t: builder.track_to_artist.get(t, 'unknown')  # noqa: E731

    models = {}
    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        if not path:
            parser.error(f"--checkpoint expects NAME=PATH, got {spec!r}")
        models[name] = load_model(path, device)

    with open(Path(args.data_dir) / f"{args.split}_pairs.json") as f:
        pairs = json.load(f)

    use_rerank = args.fatigue > 0 or args.diversity > 0
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    candidate_recall = 0.0

    for n, pair in enumerate(pairs, 1):
        prefix = pair['prefix']
        truth = {t['track_uri'] for t in pair['continuation']}
        candidates = builder.candidates_from_cooccurrence([t['track_uri'] for t in prefix], args.num_candidates)
        candidate_recall += len(truth & set(candidates)) / len(truth)

        rankings = {'cooccurrence': candidates}
        if models and candidates:
            features = torch.from_numpy(np.stack([builder.extract_features(prefix, c) for c in candidates])).to(device)
            recent = {t['artist_uri'] for t in prefix[-5:]}
            cand_artists = [artist_of(c) for c in candidates]
            for name, model in models.items():
                with torch.no_grad():
                    scores = model(features).squeeze(-1).cpu().tolist()
                order = rerank(scores, cand_artists, set(), len(candidates))
                rankings[f'model:{name}'] = [candidates[i] for i in order]
                if use_rerank:
                    order = rerank(scores, cand_artists, recent, len(candidates),
                                   fatigue=args.fatigue, diversity=args.diversity)
                    rankings[f'model:{name}+rerank'] = [candidates[i] for i in order]

        for method, ranked in rankings.items():
            for key, value in list_metrics(ranked, truth, artist_of, args.k).items():
                sums[method][key] += value

        if n % 500 == 0:
            logger.info(f"  evaluated {n}/{len(pairs)} pairs")

    results = {
        'split': args.split,
        'num_pairs': len(pairs),
        'num_candidates': args.num_candidates,
        'k': args.k,
        'rerank_weights': {'fatigue': args.fatigue, 'diversity': args.diversity},
        'candidate_recall': candidate_recall / len(pairs),
        'methods': {m: {key: v / len(pairs) for key, v in vals.items()} for m, vals in sums.items()},
    }

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
