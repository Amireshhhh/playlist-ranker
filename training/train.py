"""
Training Script for Ranking Model

Trains the ranking model with multi-objective constraints.

Each training example is a group: the candidate list for one playlist prefix
(see features/build_features.py). The ranking loss compares every positive with
every negative in the group, and the constraint losses (fatigue, diversity,
exposure) act on the group's soft ranking distribution (see models/ranker.py).
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
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import RankingMetrics  # noqa: E402
from models.ranker import ConstraintLosses, RankingModel  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOSS_KEYS = ('total', 'rank', 'fatigue', 'diversity', 'exposure')

DEFAULT_CONFIG = {
    'input_dim': 32,
    'hidden_dims': [128, 64, 32],
    'dropout': 0.3,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'num_epochs': 50,
    'patience': 5,
    'margin': 0.5,
    'max_grad_norm': 1.0,
    'temperature': 1.0,
    'lambda_fatigue': 0.1,
    'lambda_diversity': 0.05,
    'lambda_exposure': 0.03,
    'top_k': 10,
    'seed': 42,
}


class ArtistVocab:
    """Maps artist URIs to integer indices (index 0 is reserved for unknown artists)."""

    def __init__(self, artist_popularity: Dict[str, float]):
        self.index = {a: i + 1 for i, a in enumerate(sorted(artist_popularity))}
        self.popularity = torch.zeros(len(self.index) + 1)
        for artist, i in self.index.items():
            self.popularity[i] = artist_popularity[artist]

    def __call__(self, artist_uri: str) -> int:
        return self.index.get(artist_uri, 0)


class PlaylistGroupDataset(Dataset):
    """One item per (prefix, continuation) group from a features file."""

    def __init__(self, feature_file: str, vocab: ArtistVocab):
        logger.info(f"Loading features from {feature_file}...")
        with open(feature_file, 'r') as f:
            records = json.load(f)

        grouped: Dict[int, List[Dict]] = defaultdict(list)
        for r in records:
            grouped[r['group']].append(r)

        self.groups = []
        for _, rows in sorted(grouped.items()):
            self.groups.append({
                'features': torch.tensor([r['features'] for r in rows], dtype=torch.float32),
                'labels': torch.tensor([r['label'] for r in rows], dtype=torch.long),
                'artist_idx': torch.tensor([vocab(r['artist_uri']) for r in rows], dtype=torch.long),
                'in_recent_prefix': torch.tensor([bool(r['artist_in_recent_prefix']) for r in rows]),
            })

        n_pos = sum(int(g['labels'].sum()) for g in self.groups)
        logger.info(f"Loaded {len(records)} records in {len(self.groups)} groups "
                    f"({n_pos} positives, {len(records) - n_pos} negatives)")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]

    def feature_stats(self):
        """Per-feature mean and standard deviation over all records."""
        x = torch.cat([g['features'] for g in self.groups])
        return x.mean(dim=0), x.std(dim=0)


def collate_groups(batch):
    """Groups have different sizes, so a batch is simply a list of groups."""
    return batch


class Trainer:
    """Trainer for ranking model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        artist_popularity: torch.Tensor,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.artist_popularity = artist_popularity.to(device)

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # The monitored value is a loss, so lower is better: mode='min'.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _group_loss(self, group):
        features = group['features'].to(self.device)
        scores = self.model(features).squeeze(-1)
        loss, parts = ConstraintLosses.compute_group_loss(
            scores,
            group['labels'].to(self.device),
            artist_idx=group['artist_idx'].to(self.device),
            in_recent_prefix=group['in_recent_prefix'].to(self.device),
            artist_popularity=self.artist_popularity,
            lambda_fatigue=self.config.get('lambda_fatigue', 0.0),
            lambda_diversity=self.config.get('lambda_diversity', 0.0),
            lambda_exposure=self.config.get('lambda_exposure', 0.0),
            margin=self.config.get('margin', 0.5),
            temperature=self.config.get('temperature', 1.0),
        )
        return loss, parts, scores

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch; returns average loss components per group."""
        self.model.train()
        totals = dict.fromkeys(LOSS_KEYS, 0.0)
        n_groups = 0

        for batch in self.train_loader:
            batch_loss = 0.0
            for group in batch:
                loss, parts, _ = self._group_loss(group)
                batch_loss = batch_loss + loss
                for k in LOSS_KEYS:
                    totals[k] += parts[k]
            batch_loss = batch_loss / len(batch)

            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
            self.optimizer.step()
            n_groups += len(batch)

        return {k: v / max(n_groups, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Average loss components and NDCG@K over validation groups."""
        self.model.eval()
        totals = dict.fromkeys(LOSS_KEYS, 0.0)
        ndcg_sum = 0.0
        n_groups = 0
        k = self.config.get('top_k', 10)

        for batch in self.val_loader:
            for group in batch:
                _, parts, scores = self._group_loss(group)
                for key in LOSS_KEYS:
                    totals[key] += parts[key]
                order = torch.argsort(scores, descending=True).cpu()
                relevance = group['labels'][order].tolist()
                ndcg_sum += RankingMetrics.ndcg_at_k(relevance, k)
                n_groups += 1

        out = {key: float(v / max(n_groups, 1)) for key, v in totals.items()}
        out[f'ndcg@{k}'] = float(ndcg_sum / max(n_groups, 1))
        return out

    def train(self, num_epochs: int, checkpoint_dir: str):
        """Full training loop with checkpointing and early stopping."""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        k = self.config.get('top_k', 10)

        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Config: {self.config}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            train_losses = self.train_epoch()
            val_losses = self.validate()
            self.scheduler.step(val_losses['total'])

            fmt = lambda d: ", ".join(f"{key.capitalize()}: {d[key]:.4f}" for key in LOSS_KEYS)  # noqa: E731
            logger.info(f"Epoch {epoch + 1}/{num_epochs}  lr={self.optimizer.param_groups[0]['lr']:.2e}")
            logger.info(f"  Train - {fmt(train_losses)}")
            logger.info(f"  Val   - {fmt(val_losses)}, NDCG@{k}: {val_losses[f'ndcg@{k}']:.4f}")

            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                checkpoint_file = checkpoint_path / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    f'val_ndcg@{k}': val_losses[f'ndcg@{k}'],
                    'config': self.config,
                }, checkpoint_file)
                logger.info(f"  Saved best model (val_loss: {val_losses['total']:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.get('patience', 5):
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Model saved to: {checkpoint_path / 'best_model.pt'}")
        logger.info("=" * 60)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train ranking model")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Config file")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--experiment", type=str, default="full_constraints",
                        help="Experiment name; checkpoints go to <checkpoint-dir>/<experiment>/")
    parser.add_argument("--no-constraints", action="store_true", help="Set all constraint weights to 0")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from the config")

    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config_file = Path(args.config)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f) or {})
    else:
        logger.warning(f"Config file not found: {config_file}; using defaults")

    if args.no_constraints:
        config['lambda_fatigue'] = 0.0
        config['lambda_diversity'] = 0.0
        config['lambda_exposure'] = 0.0
        logger.info("Constraints disabled")
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    config['experiment_name'] = args.experiment

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    data_path = Path(args.data_dir)
    with open(data_path / "artist_popularity.json", 'r') as f:
        vocab = ArtistVocab(json.load(f))

    train_dataset = PlaylistGroupDataset(str(data_path / "train_features.json"), vocab)
    val_dataset = PlaylistGroupDataset(str(data_path / "val_features.json"), vocab)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=0, collate_fn=collate_groups)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=0, collate_fn=collate_groups)

    model = RankingModel(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
    )
    model.set_normalization(*train_dataset.feature_stats())
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        artist_popularity=vocab.popularity,
        device=device,
    )
    trainer.train(num_epochs=config['num_epochs'], checkpoint_dir=str(Path(args.checkpoint_dir) / args.experiment))


if __name__ == "__main__":
    main()
