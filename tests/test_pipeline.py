"""Tests for the complaint routing pipeline."""

import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.training.embedder import ComplaintEmbedder
from src.training.officer_router import OfficerRouter
from src.training.priority_classifier import PriorityClassifier
from src.training.eta_predictor import ETAPredictor
from src.training.similarity_search import SimilaritySearch
from src.preprocessing.text_processor import TextProcessor


class TestTextProcessor(unittest.TestCase):
    """Test text preprocessing."""
    
    def setUp(self):
        self.processor = TextProcessor()
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "Water supply   disrupted!!!  "
        cleaned = self.processor.clean_text(text)
        self.assertEqual(cleaned, "Water supply disrupted")
    
    def test_process(self):
        """Test text processing."""
        text = "Test complaint"
        result = self.processor.process(text)
        self.assertIn('text', result)
        self.assertIn('language', result)


class TestEmbedder(unittest.TestCase):
    """Test embedding generation."""
    
    def setUp(self):
        self.embedder = ComplaintEmbedder()
    
    def test_embed_single(self):
        """Test single text embedding."""
        text = "Water supply issue"
        embedding = self.embedder.embed(text)
        self.assertEqual(embedding.shape[0], 1)
        self.assertEqual(embedding.shape[1], self.embedder.get_embedding_dim())
    
    def test_embed_batch(self):
        """Test batch embedding."""
        texts = ["Water issue", "Power outage", "Road problem"]
        embeddings = self.embedder.embed(texts)
        self.assertEqual(embeddings.shape[0], 3)


class TestOfficerRouter(unittest.TestCase):
    """Test officer routing."""
    
    def test_train_predict(self):
        """Test training and prediction."""
        router = OfficerRouter(n_officers=8)
        
        # Dummy data
        X = np.random.randn(100, 384)
        y = np.random.randint(0, 8, 100)
        
        # Train
        metrics = router.train(X, y)
        self.assertIn('train_accuracy', metrics)
        
        # Predict
        predictions = router.predict(X[:10])
        self.assertEqual(len(predictions), 10)


class TestPriorityClassifier(unittest.TestCase):
    """Test priority classification."""
    
    def test_train_predict(self):
        """Test training and prediction."""
        classifier = PriorityClassifier()
        
        # Dummy data
        X = np.random.randn(100, 384)
        y = np.random.randint(0, 3, 100)
        
        # Train
        metrics = classifier.train(X, y)
        self.assertIn('train_accuracy', metrics)
        
        # Predict
        predictions = classifier.predict_labels(X[:10])
        self.assertEqual(len(predictions), 10)
        self.assertIn(predictions[0], ["Low", "Medium", "High"])


class TestETAPredictor(unittest.TestCase):
    """Test ETA prediction."""
    
    def test_train_predict(self):
        """Test training and prediction."""
        predictor = ETAPredictor()
        
        # Dummy data
        X = np.random.randn(100, 384)
        y = np.random.uniform(1, 30, 100)
        
        # Train
        metrics = predictor.train(X, y)
        self.assertIn('train_mae', metrics)
        
        # Predict
        predictions = predictor.predict(X[:10])
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(p >= 0 for p in predictions))


class TestSimilaritySearch(unittest.TestCase):
    """Test similarity search."""
    
    def test_build_search(self):
        """Test index building and search."""
        search = SimilaritySearch(embedding_dim=384)
        
        # Dummy data
        embeddings = np.random.randn(100, 384).astype('float32')
        metadata = [{"id": i, "text": f"Complaint {i}"} for i in range(100)]
        
        # Build index
        search.build_index(embeddings, metadata)
        self.assertTrue(search.is_built)
        
        # Search
        query = np.random.randn(384)
        results = search.search(query, k=5)
        self.assertEqual(len(results), 5)
        self.assertIn('similarity_score', results[0])


if __name__ == '__main__':
    unittest.main()
