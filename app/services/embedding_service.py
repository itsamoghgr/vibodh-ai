# -*- coding: utf-8 -*-
"""
Embedding Service
Handles text chunking and embedding generation using OpenAI or Groq
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from app.core.config import settings


class EmbeddingService:
    def __init__(self, provider: str = "openai"):
        """
        Initialize embedding service

        Note: For now, we use OpenAI for embeddings as Groq doesn't have
        an embeddings API yet. Groq will be used for LLM chat completions.

        Args:
            provider: 'openai' (embeddings always use OpenAI for now)
        """
        self.provider = "openai"  # Force OpenAI for embeddings
        self.chunk_size = 1500
        self.chunk_overlap = 200

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Initialize OpenAI client for embeddings
        from openai import OpenAI
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set for embeddings")

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "text-embedding-3-small"  # 1536 dimensions, cost-effective
        self.dimensions = 1536

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text.strip()) == 0:
            return []

        chunks = self.text_splitter.split_text(text)

        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "metadata": metadata or {}
            }
            result.append(chunk_data)

        return result

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single piece of text

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Cannot generate embedding for empty text")

        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single API call

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and len(t.strip()) > 0]
        if not valid_texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=valid_texts
        )

        return [item.embedding for item in response.data]

    def embed_document(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk a document and generate embeddings for all chunks

        Args:
            text: Document text
            metadata: Metadata to attach to chunks

        Returns:
            List of dictionaries containing chunk text, embedding, and metadata
        """
        # Chunk the text
        chunks = self.chunk_text(text, metadata)

        if not chunks:
            return []

        # Extract chunk texts
        chunk_texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings in batch
        embeddings = self.generate_embeddings_batch(chunk_texts)

        # Combine chunks with their embeddings
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            result.append({
                "content": chunk["text"],
                "embedding": embedding,
                "metadata": {
                    **chunk["metadata"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"]
                }
            })

        return result

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (useful for staying under API limits)

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except Exception:
            # Rough estimate if tiktoken fails
            return len(text) // 4


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton"""
    global _embedding_service
    if _embedding_service is None:
        provider = getattr(settings, "EMBEDDING_PROVIDER", "openai")
        _embedding_service = EmbeddingService(provider=provider)
    return _embedding_service
