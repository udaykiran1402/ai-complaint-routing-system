"""Configuration for the complaint routing system."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_SAVED_DIR = MODELS_DIR / "saved"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, MODELS_SAVED_DIR, EMBEDDINGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# Officer definitions
OFFICERS = [
    {"id": 1, "name": "Water Supply Officer", "expertise": ["water", "supply", "plumbing", "leak", "pipe"]},
    {"id": 2, "name": "Electricity Officer", "expertise": ["electricity", "power", "outage", "transformer", "voltage"]},
    {"id": 3, "name": "Road Maintenance Officer", "expertise": ["road", "pothole", "street", "pavement", "traffic"]},
    {"id": 4, "name": "Sanitation Officer", "expertise": ["garbage", "waste", "sanitation", "cleaning", "drainage"]},
    {"id": 5, "name": "Building Inspector", "expertise": ["building", "construction", "illegal", "permit", "structure"]},
    {"id": 6, "name": "Noise Pollution Officer", "expertise": ["noise", "sound", "loud", "disturbance", "pollution"]},
    {"id": 7, "name": "Public Health Officer", "expertise": ["health", "disease", "medical", "hospital", "clinic"]},
    {"id": 8, "name": "Parks & Recreation Officer", "expertise": ["park", "garden", "playground", "recreation", "trees"]},
]

# Priority levels
PRIORITY_LEVELS = ["High", "Medium", "Low"]

# Training parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# FAISS parameters
FAISS_INDEX_TYPE = "Flat"  # Options: Flat, IVF, HNSW
SIMILARITY_TOP_K = 5

# Model file paths
OFFICER_ROUTER_PATH = MODELS_SAVED_DIR / "officer_router.pkl"
PRIORITY_CLASSIFIER_PATH = MODELS_SAVED_DIR / "priority_classifier.pkl"
ETA_REGRESSOR_PATH = MODELS_SAVED_DIR / "eta_regressor.pkl"
FAISS_INDEX_PATH = MODELS_SAVED_DIR / "faiss_index.bin"
COMPLAINT_METADATA_PATH = MODELS_SAVED_DIR / "complaint_metadata.pkl"
EMBEDDER_PATH = MODELS_SAVED_DIR / "embedder"
