"""ETA (resolution time) prediction."""

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import joblib


class ETAPredictor:
    """Predict expected resolution time in days."""
    
    def __init__(self):
        """Initialize ETA predictor."""
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            loss='huber'
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the ETA predictor.
        
        Args:
            X: Feature embeddings (n_samples, embedding_dim)
            y: Resolution times in days (n_samples,)
        
        Returns:
            Training metrics
        """
        print(f"Training ETA predictor on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(X)
        mae = np.mean(np.abs(train_predictions - y))
        rmse = np.sqrt(np.mean((train_predictions - y) ** 2))
        
        return {
            "train_mae": mae,
            "train_rmse": rmse,
            "n_samples": len(X)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict resolution times.
        
        Args:
            X: Feature embeddings (n_samples, embedding_dim)
        
        Returns:
            Predicted resolution times in days (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0.5)
        return predictions
    
    def predict_single(self, X: np.ndarray) -> float:
        """
        Predict resolution time for a single complaint.
        
        Args:
            X: Feature embedding (1, embedding_dim)
        
        Returns:
            Predicted resolution time in days
        """
        prediction = self.predict(X)[0]
        return round(prediction, 1)
    
    def save(self, path: str):
        """Save the model."""
        joblib.dump(self.model, path)
        print(f"ETA predictor saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load a saved model."""
        instance = cls()
        instance.model = joblib.load(path)
        instance.is_trained = True
        print(f"ETA predictor loaded from {path}")
        return instance


if __name__ == "__main__":
    # Test ETA predictor
    predictor = ETAPredictor()
    
    # Dummy data
    X_train = np.random.randn(100, 384)
    y_train = np.random.uniform(1, 30, 100)  # 1-30 days
    
    metrics = predictor.train(X_train, y_train)
    print(f"Training metrics: {metrics}")
    
    # Test prediction
    X_test = np.random.randn(10, 384)
    predictions = predictor.predict(X_test)
    print(f"Predictions (days): {predictions}")
