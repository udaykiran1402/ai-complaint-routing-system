#!/bin/bash

# Setup script for Complaint Auto-Routing System

echo "=========================================="
echo "Complaint Auto-Routing System - Setup"
echo "=========================================="

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Download models
echo ""
echo "Downloading models..."
python scripts/download_models.py

# Generate training data
echo ""
echo "Generating training data..."
python scripts/generate_data.py

# Train models
echo ""
echo "Training models..."
python scripts/train_pipeline.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use the system:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run CLI: python app/cli.py --text 'Your complaint'"
echo "  3. Run web app: python app/web_app.py"
echo ""
