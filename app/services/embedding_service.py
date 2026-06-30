"""
Embedding Service using FastEmbed

Provides efficient embeddings for job descriptions and candidate profiles
using FastEmbed's highest-dimension BGE (BAAI General Embedding) models.
"""

from typing import List
import numpy as np
from fastembed import TextEmbedding


# FastEmbed model configuration
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024


class EmbeddingService:
    """
    Embedding service using FastEmbed for efficient, local embeddings.
    
    Model: BAAI/bge-large-en-v1.5 (1024-dimensional)
    - Highest quality semantic embeddings
    - Optimized for semantic search and relevance matching
    - Runs locally without API calls
    """

    def __init__(self):
        """Initialize embedding model with hardcoded high-dimension model."""
        self.model_name = DEFAULT_MODEL
        self._dimension = EMBEDDING_DIMENSION
        self._embedding_model = None

    @property
    def embedding_model(self) -> TextEmbedding:
        """Lazy-load embedding model on first access."""
        if self._embedding_model is None:
            self._embedding_model = TextEmbedding(model_name=self.model_name)
        return self._embedding_model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed (full text, no truncation)
        
        Returns:
            Embedding vector as numpy array (1024-dimensional by default)
        """
        if not text or not isinstance(text, str):
            text = ""
        
        # Clean and normalize text
        text = text.strip()
        
        if not text:
            # Return zero embedding for empty text
            return np.zeros(self._dimension, dtype=np.float32)
        
        # FastEmbed returns generator, get first result
        embeddings = list(self.embedding_model.embed([text]))
        embedding = np.array(embeddings[0], dtype=np.float32)
        
        return embedding

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batch mode).
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors (1024-dimensional by default)
        """
        if not texts:
            return []
        
        # Clean texts without truncation
        cleaned_texts = []
        for text in texts:
            if text and isinstance(text, str):
                cleaned_texts.append(text.strip())
            else:
                cleaned_texts.append("")
        
        # FastEmbed batch embedding
        embeddings = list(self.embedding_model.embed(cleaned_texts))
        
        # Convert to numpy arrays
        result = []
        for i, embedding in enumerate(embeddings):
            if cleaned_texts[i]:  # Non-empty text
                result.append(np.array(embedding, dtype=np.float32))
            else:  # Empty text
                result.append(np.zeros(self._dimension, dtype=np.float32))
        
        return result

    def embed_jd_keywords(self, jd_keywords: List[str]) -> np.ndarray:
        """
        Embed job description keywords and return mean pooled embedding.
        
        Args:
            jd_keywords: List of keywords extracted from JD
        
        Returns:
            Mean-pooled embedding (1024-dimensional by default)
        """
        if not jd_keywords:
            return np.zeros(self._dimension, dtype=np.float32)
        
        embeddings = self.embed_texts(jd_keywords)
        embeddings = [e for e in embeddings if e is not None]
        
        if not embeddings:
            return np.zeros(self._dimension, dtype=np.float32)
        
        # Mean pooling of keyword embeddings
        return np.mean(embeddings, axis=0).astype(np.float32)

    def embed_candidate_skills(self, skills: List[str]) -> np.ndarray:
        """
        Embed candidate skills and return mean pooled embedding.
        
        Args:
            skills: List of candidate's skills
        
        Returns:
            Mean-pooled embedding (1024-dimensional by default)
        """
        if not skills:
            return np.zeros(self._dimension, dtype=np.float32)
        
        embeddings = self.embed_texts(skills)
        embeddings = [e for e in embeddings if e is not None]
        
        if not embeddings:
            return np.zeros(self._dimension, dtype=np.float32)
        
        # Mean pooling of skill embeddings
        return np.mean(embeddings, axis=0).astype(np.float32)

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        FastEmbed doesn't provide a built-in similarity function, so we use
        numpy for efficient computation.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score (0.0-1.0)
        """
        # Normalize embeddings using numpy's linalg
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Use numpy's efficient dot product on normalized vectors
        similarity = float(np.dot(embedding1 / norm1, embedding2 / norm2))
        
        # Clamp to [0, 1] for cosine similarity
        return max(0.0, min(1.0, similarity))

    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings produced by this model."""
        return self._dimension


# Global instance for reuse
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create global embedding service instance.
    
    Uses BAAI/bge-large-en-v1.5 (1024-dim) for best quality.
    
    Returns:
        EmbeddingService instance with 1024-dimensional embeddings
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
