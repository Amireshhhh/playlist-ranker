#!/bin/bash
# Quick Start Script for Playlist Ranker

set -e  # Exit on error

echo "=============================================="
echo "Playlist Ranker - Quick Start"
echo "=============================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version
python -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .

echo ""
echo "✓ Installation complete!"
echo ""

# Download/create sample data
echo "Setting up data..."
python data/mpd_loader.py --sample

echo ""
echo "Generating train/val/test splits..."
python data/splits.py --input-dir ./data/mpd --output-dir ./data/processed --max-playlists 1000

echo ""
echo "Building features..."
python features/build_features.py --data-dir ./data/processed

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Train baseline model:"
echo "     python training/train.py --experiment baseline --no-constraints"
echo ""
echo "  2. Train full model with constraints:"
echo "     python training/train.py --experiment full_constraints"
echo ""
echo "  3. Evaluate both on the test split:"
echo "     python evaluation/evaluate.py --checkpoint baseline=checkpoints/baseline/best_model.pt \\"
echo "         --checkpoint full=checkpoints/full_constraints/best_model.pt"
echo ""
echo "  4. Start API server:"
echo "     uvicorn serving.api:app --reload"
echo ""
echo "  5. Visit http://localhost:8000/docs for interactive API"
echo ""
echo "=============================================="
