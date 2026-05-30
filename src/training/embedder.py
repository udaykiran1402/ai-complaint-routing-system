"""Sentence embedding model for semantic representation."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List
import torch


class ComplaintEmbedder:
    """Generate embeddings for complaint text using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize sentence transformer model.
        
        Args:
            model_name: HuggingFace model name
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Embedding model loaded on {self.device}")
    
    def embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
        
        Returns:
            Numpy array of embeddings (n_samples, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()
    
    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
        print(f"Embedder saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load a saved model."""
        instance = cls.__new__(cls)
        instance.model = SentenceTransformer(path)
        instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        instance.model.to(instance.device)
        print(f"Embedder loaded from {path}")
        return instance


if __name__ == "__main__":
    # Test embedder
    embedder = ComplaintEmbedder()
    sample_texts = [
        "Water supply is disrupted in my area",
        "No electricity for 2 days",
        "Road has big potholes"
    ]
    embeddings = embedder.embed(sample_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.get_embedding_dim()}")
