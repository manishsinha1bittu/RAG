"""
Vector Store Implementation for RAG

This module provides vector storage and similarity search functionality
using sentence transformers and FAISS for efficient retrieval.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import pickle
import os


class VectorStore:
    """Vector store for similarity search using embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.embeddings = None
        self.documents = []
        self.document_ids = []
        
        # Try to import sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            print("Warning: sentence_transformers not installed. Using dummy embeddings.")
            self.model = None
    
    def add_documents(self, documents: List[str], doc_ids: List[str] = None) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document strings
            doc_ids: Optional list of document IDs
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        if self.model is not None:
            embeddings = self.model.encode(documents, show_progress_bar=True)
        else:
            # Dummy embeddings for demonstration
            embeddings = np.random.rand(len(documents), 384)
        
        # Store documents and embeddings
        self.documents.extend(documents)
        self.document_ids.extend(doc_ids)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Generate query embedding
        if self.model is not None:
            query_embedding = self.model.encode([query])
        else:
            query_embedding = np.random.rand(1, 384)
        
        # Calculate cosine similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # Get top-k results
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': self.document_ids[idx],
                'text': self.documents[idx],
                'score': float(similarities[0][idx])
            })
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between two arrays.
        
        Args:
            a: First array
            b: Second array
            
        Returns:
            Cosine similarity matrix
        """
        # Normalize vectors
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        return np.dot(a_norm, b_norm.T)
    
    def save(self, filepath: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            filepath: Path to save the vector store
        """
        data = {
            'embeddings': self.embeddings,
            'documents': self.documents,
            'document_ids': self.document_ids,
            'model_name': self.model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            filepath: Path to load the vector store from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vector store file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.documents = data['documents']
        self.document_ids = data['document_ids']
        self.model_name = data['model_name']
        
        # Reinitialize model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            self.model = None


class FAISSVectorStore(VectorStore):
    """Vector store using FAISS for efficient similarity search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """
        Initialize FAISS vector store.
        
        Args:
            model_name: Name of the sentence transformer model
            dimension: Dimension of embeddings
        """
        super().__init__(model_name)
        self.dimension = dimension
        self.index = None
        
        # Try to import FAISS
        try:
            import faiss
            self.faiss = faiss
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        except ImportError:
            print("Warning: FAISS not installed. Using numpy-based search.")
            self.faiss = None
    
    def add_documents(self, documents: List[str], doc_ids: List[str] = None) -> None:
        """
        Add documents to the FAISS index.
        
        Args:
            documents: List of document strings
            doc_ids: Optional list of document IDs
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        if self.model is not None:
            embeddings = self.model.encode(documents, show_progress_bar=True)
        else:
            embeddings = np.random.rand(len(documents), self.dimension)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        if self.faiss is not None and self.index is not None:
            self.index.add(embeddings.astype('float32'))
        
        # Store documents
        self.documents.extend(documents)
        self.document_ids.extend(doc_ids)
        
        # Store embeddings for fallback
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using FAISS index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if len(self.documents) == 0:
            return []
        
        # Generate query embedding
        if self.model is not None:
            query_embedding = self.model.encode([query])
        else:
            query_embedding = np.random.rand(1, self.dimension)
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search using FAISS
        if self.faiss is not None and self.index is not None:
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback to numpy-based search
            similarities = self._cosine_similarity(query_embedding, self.embeddings)
            top_indices = np.argsort(similarities[0])[::-1][:top_k]
            scores = similarities[0][top_indices]
            indices = top_indices
        
        results = []
        for score, idx in zip(scores, indices):
            if idx < len(self.documents):  # Valid index
                results.append({
                    'id': self.document_ids[idx],
                    'text': self.documents[idx],
                    'score': float(score)
                })
        
        return results


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Machine learning enables computers to learn from data.",
        "Natural language processing helps computers understand text.",
        "Deep learning uses neural networks with multiple layers.",
        "Computer vision allows machines to interpret images.",
        "Robotics combines AI with mechanical engineering."
    ]
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Add documents
    vector_store.add_documents(documents)
    
    # Search
    query = "What is machine learning?"
    results = vector_store.similarity_search(query, top_k=3)
    
    print(f"Query: {query}")
    print("\nTop results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Text: {result['text']}")
        print() 