"""Priority classification (High/Medium/Low)."""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib


class PriorityClassifier:
    """Classify complaint priority levels."""
    
    def __init__(self):
        """Initialize priority classifier."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.is_trained = False
        self.priority_map = {0: "Low", 1: "Medium", 2: "High"}
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the priority classifier.
        
        Args:
            X: Feature embeddings (n_samples, embedding_dim)
            y: Priority labels (n_samples,) - 0: Low, 1: Medium, 2: High
        
        Returns:
            Training metrics
        """
        print(f"Training priority classifier on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training accuracy
        train_acc = self.model.score(X, y)
        
        return {
            "train_accuracy": train_acc,
            "n_samples": len(X)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict priority levels.
        
        Args:
            X: Feature embeddings (n_samples, embedding_dim)
        
        Returns:
            Predicted priority labels (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for each priority level.
        
        Args:
            X: Feature embeddings (n_samples, embedding_dim)
        
        Returns:
            Probability matrix (n_samples, 3)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def predict_labels(self, X: np.ndarray) -> list:
        """
        Predict priority labels as strings.
        
        Args:
            X: Feature embeddings
        
        Returns:
            List of priority labels (e.g., ["High", "Medium", "Low"])
        """
        predictions = self.predict(X)
        return [self.priority_map[p] for p in predictions]
    
    def save(self, path: str):
        """Save the model."""
        joblib.dump(self.model, path)
        print(f"Priority classifier saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load a saved model."""
        instance = cls()
        instance.model = joblib.load(path)
        instance.is_trained = True
        print(f"Priority classifier loaded from {path}")
        return instance


if __name__ == "__main__":
    # Test priority classifier
    classifier = PriorityClassifier()
    
    # Dummy data
    X_train = np.random.randn(100, 384)
    y_train = np.random.randint(0, 3, 100)
    
    metrics = classifier.train(X_train, y_train)
    print(f"Training metrics: {metrics}")
    
    # Test prediction
    X_test = np.random.randn(10, 384)
    predictions = classifier.predict_labels(X_test)
    print(f"Predictions: {predictions}")
