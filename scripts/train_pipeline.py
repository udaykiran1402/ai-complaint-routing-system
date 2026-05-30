"""Train all models in the complaint routing pipeline."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    DATA_DIR, OFFICER_ROUTER_PATH, PRIORITY_CLASSIFIER_PATH,
    ETA_REGRESSOR_PATH, FAISS_INDEX_PATH, COMPLAINT_METADATA_PATH,
    EMBEDDER_PATH, OFFICERS, RANDOM_SEED
)
from src.training.embedder import ComplaintEmbedder
from src.training.officer_router import OfficerRouter
from src.training.priority_classifier import PriorityClassifier
from src.training.eta_predictor import ETAPredictor
from src.training.similarity_search import SimilaritySearch


def load_data():
    """Load training and test data."""
    train_path = DATA_DIR / "synthetic" / "train_complaints.csv"
    test_path = DATA_DIR / "synthetic" / "test_complaints.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Please run 'python scripts/generate_data.py' first."
        )
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else None
    
    return train_df, test_df


def train_embedder(texts):
    """Initialize and save embedder."""
    print("\n" + "="*60)
    print("STEP 1: Initializing Embedder")
    print("="*60)
    
    embedder = ComplaintEmbedder()
    
    # Generate sample embeddings to verify
    sample_embeddings = embedder.embed(texts[:5])
    print(f"Sample embeddings shape: {sample_embeddings.shape}")
    
    # Save embedder
    embedder.save(str(EMBEDDER_PATH))
    
    return embedder


def train_officer_router(embedder, train_df, test_df):
    """Train officer routing model."""
    print("\n" + "="*60)
    print("STEP 2: Training Officer Router")
    print("="*60)
    
    # Generate embeddings
    print("Generating embeddings for training data...")
    X_train = embedder.embed(train_df['text'].tolist())
    y_train = train_df['officer_id'].values
    
    # Train model
    router = OfficerRouter(n_officers=len(OFFICERS))
    train_metrics = router.train(X_train, y_train)
    print(f"Training accuracy: {train_metrics['train_accuracy']:.3f}")
    
    # Evaluate on test set
    if test_df is not None:
        print("\nEvaluating on test set...")
        X_test = embedder.embed(test_df['text'].tolist())
        y_test = test_df['officer_id'].values
        
        y_pred = router.predict(X_test)
        test_acc = np.mean(y_pred == y_test)
        print(f"Test accuracy: {test_acc:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=[f"Officer {i}" for i in range(len(OFFICERS))]
        ))
    
    # Save model
    router.save(str(OFFICER_ROUTER_PATH))
    
    return router


def train_priority_classifier(embedder, train_df, test_df):
    """Train priority classification model."""
    print("\n" + "="*60)
    print("STEP 3: Training Priority Classifier")
    print("="*60)
    
    # Generate embeddings
    print("Generating embeddings for training data...")
    X_train = embedder.embed(train_df['text'].tolist())
    y_train = train_df['priority'].values
    
    # Train model
    classifier = PriorityClassifier()
    train_metrics = classifier.train(X_train, y_train)
    print(f"Training accuracy: {train_metrics['train_accuracy']:.3f}")
    
    # Evaluate on test set
    if test_df is not None:
        print("\nEvaluating on test set...")
        X_test = embedder.embed(test_df['text'].tolist())
        y_test = test_df['priority'].values
        
        y_pred = classifier.predict(X_test)
        test_acc = np.mean(y_pred == y_test)
        print(f"Test accuracy: {test_acc:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=["Low", "Medium", "High"]
        ))
    
    # Save model
    classifier.save(str(PRIORITY_CLASSIFIER_PATH))
    
    return classifier


def train_eta_predictor(embedder, train_df, test_df):
    """Train ETA prediction model."""
    print("\n" + "="*60)
    print("STEP 4: Training ETA Predictor")
    print("="*60)
    
    # Generate embeddings
    print("Generating embeddings for training data...")
    X_train = embedder.embed(train_df['text'].tolist())
    y_train = train_df['eta_days'].values
    
    # Train model
    predictor = ETAPredictor()
    train_metrics = predictor.train(X_train, y_train)
    print(f"Training MAE: {train_metrics['train_mae']:.2f} days")
    print(f"Training RMSE: {train_metrics['train_rmse']:.2f} days")
    
    # Evaluate on test set
    if test_df is not None:
        print("\nEvaluating on test set...")
        X_test = embedder.embed(test_df['text'].tolist())
        y_test = test_df['eta_days'].values
        
        y_pred = predictor.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Test MAE: {test_mae:.2f} days")
        print(f"Test RMSE: {test_rmse:.2f} days")
    
    # Save model
    predictor.save(str(ETA_REGRESSOR_PATH))
    
    return predictor


def build_similarity_index(embedder, train_df):
    """Build FAISS similarity search index."""
    print("\n" + "="*60)
    print("STEP 5: Building Similarity Search Index")
    print("="*60)
    
    # Generate embeddings
    print("Generating embeddings for all complaints...")
    embeddings = embedder.embed(train_df['text'].tolist())
    
    # Prepare metadata
    metadata = []
    for _, row in train_df.iterrows():
        metadata.append({
            'complaint_id': int(row['complaint_id']),
            'text': row['text'],
            'officer_id': int(row['officer_id']),
            'priority': int(row['priority']),
            'eta_days': float(row['eta_days']),
            'status': row['status']
        })
    
    # Build index
    search = SimilaritySearch(embedding_dim=embeddings.shape[1])
    search.build_index(embeddings, metadata)
    
    # Save index
    search.save(str(FAISS_INDEX_PATH), str(COMPLAINT_METADATA_PATH))
    
    return search


def main():
    """Run complete training pipeline."""
    print("="*60)
    print("COMPLAINT ROUTING SYSTEM - TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = load_data()
    print(f"Training samples: {len(train_df)}")
    if test_df is not None:
        print(f"Test samples: {len(test_df)}")
    
    # Train embedder
    embedder = train_embedder(train_df['text'].tolist())
    
    # Train officer router
    router = train_officer_router(embedder, train_df, test_df)
    
    # Train priority classifier
    classifier = train_priority_classifier(embedder, train_df, test_df)
    
    # Train ETA predictor
    predictor = train_eta_predictor(embedder, train_df, test_df)
    
    # Build similarity index
    search = build_similarity_index(embedder, train_df)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nAll models saved successfully:")
    print(f"  - Embedder: {EMBEDDER_PATH}")
    print(f"  - Officer Router: {OFFICER_ROUTER_PATH}")
    print(f"  - Priority Classifier: {PRIORITY_CLASSIFIER_PATH}")
    print(f"  - ETA Predictor: {ETA_REGRESSOR_PATH}")
    print(f"  - FAISS Index: {FAISS_INDEX_PATH}")
    print(f"  - Complaint Metadata: {COMPLAINT_METADATA_PATH}")
    
    print("\nYou can now run inference using:")
    print("  python app/cli.py --text 'Your complaint text'")


if __name__ == "__main__":
    main()
