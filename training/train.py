"""
Training Script for Ranking Model

Trains the ranking model with multi-objective constraints.
"""

import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.ranker import RankingModel, ConstraintLosses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlaylistRankingDataset(Dataset):
    """Dataset for playlist ranking."""
    
    def __init__(self, feature_file: str):
        """
        Initialize dataset.
        
        Args:
            feature_file: Path to feature JSON file
        """
        logger.info(f"Loading features from {feature_file}...")
        
        with open(feature_file, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples")
        
        # Separate positives and negatives
        self.positives = [x for x in self.data if x['label'] == 1]
        self.negatives = [x for x in self.data if x['label'] == 0]
        
        logger.info(f"  Positives: {len(self.positives)}")
        logger.info(f"  Negatives: {len(self.negatives)}")
    
    def __len__(self):
        # Length is number of positive samples (pairs)
        return len(self.positives)
    
    def __getitem__(self, idx):
        """
        Get a positive-negative pair.
        
        Returns:
            pos_features, neg_features, pos_label, neg_label
        """
        pos_sample = self.positives[idx]
        
        # Random negative
        neg_idx = np.random.randint(0, len(self.negatives))
        neg_sample = self.negatives[neg_idx]
        
        return {
            'pos_features': torch.tensor(pos_sample['features'], dtype=torch.float32),
            'neg_features': torch.tensor(neg_sample['features'], dtype=torch.float32),
            'pos_label': pos_sample['label'],
            'neg_label': neg_sample['label']
        }


class Trainer:
    """Trainer for ranking model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Ranking model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'rank': 0.0,
            'fatigue': 0.0,
            'diversity': 0.0,
            'exposure': 0.0
        }
        
        num_batches = 0
        
        for batch in self.train_loader:
            pos_features = batch['pos_features'].to(self.device)
            neg_features = batch['neg_features'].to(self.device)
            
            # Forward pass
            pos_scores = self.model(pos_features)
            neg_scores = self.model(neg_features)
            
            # Compute loss
            loss, loss_dict = ConstraintLosses.compute_total_loss(
                pos_scores.squeeze(),
                neg_scores.squeeze(),
                lambda_fatigue=self.config.get('lambda_fatigue', 0.0),
                lambda_diversity=self.config.get('lambda_diversity', 0.0),
                lambda_exposure=self.config.get('lambda_exposure', 0.0),
                margin=self.config.get('margin', 0.5)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            
            num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of average losses
        """
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'rank': 0.0,
            'fatigue': 0.0,
            'diversity': 0.0,
            'exposure': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                pos_features = batch['pos_features'].to(self.device)
                neg_features = batch['neg_features'].to(self.device)
                
                # Forward pass
                pos_scores = self.model(pos_features)
                neg_scores = self.model(neg_features)
                
                # Compute loss
                loss, loss_dict = ConstraintLosses.compute_total_loss(
                    pos_scores.squeeze(),
                    neg_scores.squeeze(),
                    lambda_fatigue=self.config.get('lambda_fatigue', 0.0),
                    lambda_diversity=self.config.get('lambda_diversity', 0.0),
                    lambda_exposure=self.config.get('lambda_exposure', 0.0),
                    margin=self.config.get('margin', 0.5)
                )
                
                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += loss_dict[key]
                
                num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def train(self, num_epochs: int, checkpoint_dir: str):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Config: {self.config}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_losses['total'])
            
            # Logging
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info(f"  Train - Total: {train_losses['total']:.4f}, "
                       f"Rank: {train_losses['rank']:.4f}, "
                       f"Fatigue: {train_losses['fatigue']:.4f}, "
                       f"Diversity: {train_losses['diversity']:.4f}")
            logger.info(f"  Val   - Total: {val_losses['total']:.4f}, "
                       f"Rank: {val_losses['rank']:.4f}, "
                       f"Fatigue: {val_losses['fatigue']:.4f}, "
                       f"Diversity: {val_losses['diversity']:.4f}")
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                
                checkpoint_file = checkpoint_path / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    'config': self.config
                }, checkpoint_file)
                
                logger.info(f"  ✓ Saved best model (val_loss: {val_losses['total']:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 5):
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Model saved to: {checkpoint_path / 'best_model.pt'}")
        logger.info("="*60)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train ranking model")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Config file")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--experiment", type=str, default="baseline", help="Experiment name")
    parser.add_argument("--no-constraints", action="store_true", help="Disable constraints")
    
    args = parser.parse_args()
    
    # Load config
    config_file = Path(args.config)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_file}")
        logger.info("Using default config")
        config = {
            'batch_size': 512,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'num_epochs': 50,
            'patience': 5,
            'margin': 0.5,
            'max_grad_norm': 1.0,
            'lambda_fatigue': 0.1,
            'lambda_diversity': 0.05,
            'lambda_exposure': 0.03
        }
    
    # Override constraints if requested
    if args.no_constraints:
        config['lambda_fatigue'] = 0.0
        config['lambda_diversity'] = 0.0
        config['lambda_exposure'] = 0.0
        logger.info("Constraints disabled")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load datasets
    data_path = Path(args.data_dir)
    
    train_dataset = PlaylistRankingDataset(str(data_path / "train_features.json"))
    val_dataset = PlaylistRankingDataset(str(data_path / "val_features.json"))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = RankingModel(
        input_dim=32,
        hidden_dims=[128, 64, 32],
        dropout=0.3
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment
    trainer.train(
        num_epochs=config['num_epochs'],
        checkpoint_dir=str(checkpoint_dir)
    )


if __name__ == "__main__":
    main()
