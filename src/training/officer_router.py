"""Officer routing classifier."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from typing import List, Tuple


class OfficerRouter:
    """Classify complaints to appropriate officers."""
    
    def __init__(self, n_officers: int = 8):
        """
        Initialize officer router.
        
        Args:
            n_officers: Number of officer categories
        """
        self.n_officers = n_officers
        # Use LogisticRegression for multi-class classification
        self.model = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
            class_weight='balanced'
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the officer routing model.
        
        Args:
            X: Feature embeddings (n_samples, embedding_dim)
            y: Officer labels (n_samples,)
        
        Returns:
            Training metrics
        """
        print(f"Training officer router on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training accuracy
        train_acc = self.model.score(X, y)
        
        return {
            "train_accuracy": train_acc,
            "n_samples": len(X),
            "n_officers": self.n_officers
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict officer assignments.
        
        Args:
            X: Feature embeddings (n_samples, embedding_dim)
        
        Returns:
            Predicted officer IDs (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for each officer.
        
        Args:
            X: Feature embeddings (n_samples, embedding_dim)
        
        Returns:
            Probability matrix (n_samples, n_officers)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        """Save the model."""
        joblib.dump(self.model, path)
        print(f"Officer router saved to {path}")
    
    @classmethod
    def load(cls, path: str, n_officers: int = 8):
        """Load a saved model."""
        instance = cls(n_officers=n_officers)
        instance.model = joblib.load(path)
        instance.is_trained = True
        print(f"Officer router loaded from {path}")
        return instance


if __name__ == "__main__":
    # Test officer router
    router = OfficerRouter(n_officers=8)
    
    # Dummy data
    X_train = np.random.randn(100, 384)
    y_train = np.random.randint(0, 8, 100)
    
    metrics = router.train(X_train, y_train)
    print(f"Training metrics: {metrics}")
    
    # Test prediction
    X_test = np.random.randn(10, 384)
    predictions = router.predict(X_test)
    print(f"Predictions: {predictions}")
