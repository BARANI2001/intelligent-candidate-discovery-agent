"""
Embedding Service using FastEmbed

Provides efficient embeddings for job descriptions and candidate profiles
using the BAAI/bge-small-en-v1.5 model via FastEmbed.
"""

from typing import List
import numpy as np
from fastembed import TextEmbedding


class EmbeddingService:
    """
    Embedding service using FastEmbed for efficient, local embeddings.
    
    Model: BAAI/bge-small-en-v1.5 (384-dimensional)
    - Optimized for semantic search
    - Fast inference (~100ms for batch)
    - Runs locally without API calls
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize embedding model.
        
        Args:
            model_name: FastEmbed model identifier
                Default: BAAI/bge-small-en-v1.5 (recommended for semantic search)
                Alternatives: BAAI/bge-base-en-v1.5, gte-small, etc.
        """
        self.model_name = model_name
        self._embedding_model = None

    @property
    def embedding_model(self) -> TextEmbedding:
        """Lazy-load embedding model on first access."""
        if self._embedding_model is None:
            self._embedding_model = TextEmbedding(model=self.model_name)
        return self._embedding_model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed (truncated to 8192 chars for safety)
        
        Returns:
            Embedding vector as numpy array (384-dimensional)
        """
        if not text or not isinstance(text, str):
            text = ""
        
        # Truncate to avoid issues
        text = text[:8192].strip()
        
        if not text:
            # Return zero embedding for empty text
            return np.zeros(384, dtype=np.float32)
        
        # FastEmbed returns generator, get first result
        embeddings = list(self.embedding_model.embed([text]))
        embedding = np.array(embeddings[0], dtype=np.float32)
        print(embeddings, embedding)
        
        return embedding

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batch mode).
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors (384-dimensional each)
        """
        if not texts:
            return []
        
        # Truncate and clean texts
        cleaned_texts = []
        for text in texts:
            if text and isinstance(text, str):
                cleaned_texts.append(text[:8192].strip())
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
                result.append(np.zeros(384, dtype=np.float32))
        
        return result

    def embed_jd_keywords(self, jd_keywords: List[str]) -> np.ndarray:
        """
        Embed job description keywords and return mean pooled embedding.
        
        Args:
            jd_keywords: List of keywords extracted from JD
        
        Returns:
            Mean-pooled embedding (384-dimensional)
        """
        if not jd_keywords:
            return np.zeros(384, dtype=np.float32)
        
        embeddings = self.embed_texts(jd_keywords)
        embeddings = [e for e in embeddings if e is not None]
        
        if not embeddings:
            return np.zeros(384, dtype=np.float32)
        
        # Mean pooling of keyword embeddings
        return np.mean(embeddings, axis=0).astype(np.float32)

    def embed_candidate_skills(self, skills: List[str]) -> np.ndarray:
        """
        Embed candidate skills and return mean pooled embedding.
        
        Args:
            skills: List of candidate's skills
        
        Returns:
            Mean-pooled embedding (384-dimensional)
        """
        if not skills:
            return np.zeros(384, dtype=np.float32)
        
        embeddings = self.embed_texts(skills)
        embeddings = [e for e in embeddings if e is not None]
        
        if not embeddings:
            return np.zeros(384, dtype=np.float32)
        
        # Mean pooling of skill embeddings
        return np.mean(embeddings, axis=0).astype(np.float32)

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score (0.0-1.0)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding1_norm = embedding1 / norm1
        embedding2_norm = embedding2 / norm2
        
        # Cosine similarity
        similarity = float(np.dot(embedding1_norm, embedding2_norm))
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, similarity))

    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings produced by this model."""
        return 384  # BAAI/bge-small-en-v1.5 uses 384 dimensions


# Global instance for reuse
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
