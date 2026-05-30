"""Similarity search using FAISS."""

import faiss
import numpy as np
import pickle
from typing import List, Tuple, Dict


class SimilaritySearch:
    """Find similar past complaints using FAISS vector search."""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.complaint_metadata = []
        self.is_built = False
    
    def build_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Complaint embeddings (n_samples, embedding_dim)
            metadata: List of complaint metadata dictionaries
        """
        print(f"Building FAISS index with {len(embeddings)} complaints...")
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Create flat L2 index (exact search)
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        self.complaint_metadata = metadata
        self.is_built = True
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar complaints.
        
        Args:
            query_embedding: Query embedding (1, embedding_dim) or (embedding_dim,)
            k: Number of similar complaints to retrieve
        
        Returns:
            List of similar complaints with metadata and similarity scores
        """
        if not self.is_built:
            raise ValueError("Index not built yet")
        
        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
            if idx < len(self.complaint_metadata):
                result = self.complaint_metadata[idx].copy()
                result['similarity_score'] = float(sim)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def add_complaint(self, embedding: np.ndarray, metadata: Dict):
        """
        Add a new complaint to the index.
        
        Args:
            embedding: Complaint embedding (embedding_dim,)
            metadata: Complaint metadata
        """
        if not self.is_built:
            raise ValueError("Index not built yet")
        
        # Ensure correct shape and type
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype('float32')
        faiss.normalize_L2(embedding)
        
        # Add to index
        self.index.add(embedding)
        self.complaint_metadata.append(metadata)
    
    def save(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata."""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.complaint_metadata, f)
        print(f"FAISS index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str):
        """Load FAISS index and metadata."""
        instance = cls()
        instance.index = faiss.read_index(index_path)
        instance.embedding_dim = instance.index.d
        
        with open(metadata_path, 'rb') as f:
            instance.complaint_metadata = pickle.load(f)
        
        instance.is_built = True
        print(f"FAISS index loaded from {index_path}")
        print(f"Loaded {instance.index.ntotal} vectors")
        return instance


if __name__ == "__main__":
    # Test similarity search
    search = SimilaritySearch(embedding_dim=384)
    
    # Dummy data
    embeddings = np.random.randn(100, 384).astype('float32')
    metadata = [{"id": i, "text": f"Complaint {i}"} for i in range(100)]
    
    search.build_index(embeddings, metadata)
    
    # Test search
    query = np.random.randn(384)
    results = search.search(query, k=5)
    print(f"Found {len(results)} similar complaints")
    for r in results:
        print(f"  Rank {r['rank']}: ID {r['id']}, Similarity: {r['similarity_score']:.3f}")
