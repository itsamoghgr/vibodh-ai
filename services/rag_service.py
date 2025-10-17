# -*- coding: utf-8 -*-
"""
RAG Service
Handles Retrieval-Augmented Generation with streaming support
"""

from typing import List, Dict, Any, AsyncGenerator, Optional
from supabase import Client
from services.embedding_service import get_embedding_service
from services.llm_service import get_llm_service


class RAGService:
    def __init__(self, supabase_client: Client):
        """Initialize RAG service"""
        self.supabase = supabase_client
        self.embedding_service = get_embedding_service()
        self.llm_service = get_llm_service()

    def retrieve_context(
        self,
        query: str,
        org_id: str,
        limit: int = 5,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using semantic search

        Args:
            query: User's question
            org_id: Organization ID
            limit: Max number of context items
            threshold: Similarity threshold

        Returns:
            List of context items with document info
        """
        import numpy as np

        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embedding(query)
        query_vector = np.array(query_embedding)

        # Fetch all embeddings for this organization
        result = self.supabase.table("embeddings")\
            .select("id, document_id, content, embedding, metadata")\
            .eq("org_id", org_id)\
            .execute()

        if not result.data:
            return []

        # Calculate cosine similarity in Python
        similarities = []
        for item in result.data:
            # Convert embedding from database format to numpy array
            # Database may return it as string or list, handle both
            embedding = item["embedding"]
            if isinstance(embedding, str):
                import json
                embedding = json.loads(embedding)
            doc_vector = np.array(embedding, dtype=np.float32)

            # Calculate cosine similarity
            # similarity = (A · B) / (||A|| * ||B||)
            dot_product = np.dot(query_vector, doc_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc_vector)
            similarity = dot_product / (norm_query * norm_doc)

            # Only include if above threshold
            if similarity > threshold:
                similarities.append({
                    "document_id": item["document_id"],
                    "content": item["content"],
                    "similarity": float(similarity),
                    "metadata": item.get("metadata", {})
                })

        # Sort by similarity (highest first) and limit results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:limit]

    async def generate_answer_stream(
        self,
        query: str,
        org_id: str,
        max_context_items: int = 5,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming answer using RAG

        Args:
            query: User's question
            org_id: Organization ID
            max_context_items: Max context items to retrieve
            system_prompt: Optional custom system prompt

        Yields:
            Answer chunks as they're generated
        """
        # Retrieve context
        context_items = self.retrieve_context(
            query=query,
            org_id=org_id,
            limit=max_context_items,
            threshold=0.5  # Lowered threshold for better recall
        )

        if not context_items:
            yield "I couldn't find any relevant information in your company's knowledge base to answer this question. Please try rephrasing your query or connect more data sources."
            return

        # Format context
        context_text = "\n\n".join([
            f"[Source {i+1}] {item['content']}"
            for i, item in enumerate(context_items)
        ])

        # Default system prompt
        if not system_prompt:
            system_prompt = """You are Vibodh, an AI assistant that helps answer questions about a company's internal knowledge.

Your task is to answer questions based on the provided context from the company's Slack messages, documents, and other sources.

Guidelines:
- Always base your answer on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite sources when possible (e.g., "According to Source 1..." or "As mentioned in the Slack message...")
- Be concise but thorough
- If you're uncertain, express appropriate uncertainty
- Use a professional and helpful tone
- Format your response with proper paragraphs and bullet points when appropriate"""

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Context from company knowledge base:

{context_text}

---

Question: {query}

Please provide a helpful answer based on the context above."""}
        ]

        # Stream response from LLM
        async for chunk in self.llm_service.generate_streaming_response(messages):
            yield chunk

    def generate_answer(
        self,
        query: str,
        org_id: str,
        max_context_items: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a non-streaming answer using RAG

        Args:
            query: User's question
            org_id: Organization ID
            max_context_items: Max context items to retrieve

        Returns:
            Dictionary with answer and context
        """
        # Retrieve context
        context_items = self.retrieve_context(
            query=query,
            org_id=org_id,
            limit=max_context_items,
            threshold=0.5  # Lowered threshold for better recall
        )

        if not context_items:
            return {
                "query": query,
                "context": [],
                "answer": "I couldn't find any relevant information in your company's knowledge base to answer this question."
            }

        # Generate answer
        answer = self.llm_service.generate_answer_from_context(
            query=query,
            context_items=context_items
        )

        return {
            "query": query,
            "context": context_items,
            "answer": answer
        }


# Singleton instance
_rag_service = None


def get_rag_service(supabase_client: Client) -> RAGService:
    """Get or create RAG service instance"""
    return RAGService(supabase_client)
