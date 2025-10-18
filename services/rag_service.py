# -*- coding: utf-8 -*-
"""
RAG Service
Handles Retrieval-Augmented Generation with streaming support
Now with Memory Layer, Caching, and Context Optimization
"""

from typing import List, Dict, Any, AsyncGenerator, Optional
from supabase import Client
from services.embedding_service import get_embedding_service
from services.llm_service import get_llm_service
from datetime import datetime, timedelta
import os

# Debug flag
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


class EmbeddingsCache:
    """Simple in-memory cache for embeddings"""

    def __init__(self, ttl_minutes: int = 15):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)

    def get(self, org_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached embeddings if still valid"""
        if org_id in self.cache:
            cached_data = self.cache[org_id]
            if datetime.now() - cached_data["timestamp"] < self.ttl:
                if DEBUG:
                    print(f"[CACHE HIT] Returning cached embeddings for org {org_id}")
                return cached_data["embeddings"]
            else:
                if DEBUG:
                    print(f"[CACHE EXPIRED] Removing stale cache for org {org_id}")
                del self.cache[org_id]
        return None

    def set(self, org_id: str, embeddings: List[Dict[str, Any]]):
        """Cache embeddings with timestamp"""
        self.cache[org_id] = {"embeddings": embeddings, "timestamp": datetime.now()}
        if DEBUG:
            print(f"[CACHE SET] Cached {len(embeddings)} embeddings for org {org_id}")

    def invalidate(self, org_id: str):
        """Invalidate cache for an organization"""
        if org_id in self.cache:
            del self.cache[org_id]
            if DEBUG:
                print(f"[CACHE INVALIDATE] Cleared cache for org {org_id}")


# Global cache instance
_embeddings_cache = EmbeddingsCache(ttl_minutes=15)


class RAGService:
    def __init__(self, supabase_client: Client):
        """Initialize RAG service"""
        self.supabase = supabase_client
        self.embedding_service = get_embedding_service()
        self.llm_service = get_llm_service()

    def retrieve_memory(
        self, org_id: str, user_id: Optional[str] = None, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent important memories before RAG retrieval

        Args:
            org_id: Organization ID
            user_id: Optional user ID for personalized memories
            limit: Max memories to retrieve

        Returns:
            List of memory entries
        """
        try:
            query = (
                self.supabase.table("ai_memory")
                .select("*")
                .eq("org_id", org_id)
                .order("importance", desc=True)
                .order("created_at", desc=True)
                .limit(limit)
            )

            # Filter by user if provided
            if user_id:
                query = query.eq("user_id", user_id)

            result = query.execute()

            if DEBUG:
                print(
                    f"[MEMORY] Retrieved {len(result.data) if result.data else 0} memory entries"
                )

            return result.data if result.data else []

        except Exception as e:
            if DEBUG:
                print(f"[MEMORY ERROR] Failed to retrieve memories: {e}")
            return []

    def retrieve_context(
        self, query: str, org_id: str, limit: int = 5, threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using semantic search

        Args:
            query: User's question
            org_id: Organization ID
            limit: Max number of context items
            threshold: Similarity threshold (default 0.3 for better recall)

        Returns:
            List of context items with document info
        """
        import numpy as np

        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embedding(query)
        query_vector = np.array(query_embedding)

        # Try to get from cache first
        cached_embeddings = _embeddings_cache.get(org_id)

        if cached_embeddings is not None:
            embeddings_data = cached_embeddings
        else:
            # Fetch all embeddings for this organization
            result = (
                self.supabase.table("embeddings")
                .select("id, document_id, content, embedding, metadata")
                .eq("org_id", org_id)
                .execute()
            )

            if DEBUG:
                print(
                    f"[EMBEDDINGS] Fetched {len(result.data) if result.data else 0} embeddings from DB"
                )

            if not result.data:
                if DEBUG:
                    print("[EMBEDDINGS] No embeddings found, returning empty list")
                return []

            embeddings_data = result.data
            # Cache for future use
            _embeddings_cache.set(org_id, embeddings_data)

        # Calculate cosine similarity in Python
        similarities = []
        for item in embeddings_data:
            # Convert embedding from database format to numpy array
            # Database may return it as string or list, handle both
            embedding = item["embedding"]
            if isinstance(embedding, str):
                import json

                embedding = json.loads(embedding)
            doc_vector = np.array(embedding, dtype=np.float32)

            # Calculate cosine similarity with zero-division protection
            dot_product = np.dot(query_vector, doc_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc_vector)
            similarity = dot_product / max(norm_query * norm_doc, 1e-8)

            # Only include if above threshold
            if similarity > threshold:
                similarities.append(
                    {
                        "document_id": item["document_id"],
                        "content": item["content"],
                        "similarity": float(similarity),
                        "metadata": item.get("metadata", {}),
                    }
                )

        if DEBUG:
            print(
                f"[SIMILARITY] Found {len(similarities)} documents above threshold {threshold}"
            )
            if similarities:
                print(
                    f"[SIMILARITY] Top score: {max(s['similarity'] for s in similarities):.4f}"
                )

        # Sort by similarity (highest first) and limit results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = similarities[:limit]

        if DEBUG:
            print(f"[CONTEXT] Returning top {len(top_results)} results")

        # Enrich with document metadata (author, channel, etc.)
        enriched_results = []
        for result in top_results:
            doc = (
                self.supabase.table("documents")
                .select(
                    "author, author_id, channel_name, channel_id, title, url, created_at"
                )
                .eq("id", result["document_id"])
                .single()
                .execute()
            )

            if doc.data:
                result["author"] = doc.data.get("author")
                result["author_id"] = doc.data.get("author_id")
                result["channel_name"] = doc.data.get("channel_name")
                result["channel_id"] = doc.data.get("channel_id")
                result["title"] = doc.data.get("title")
                result["url"] = doc.data.get("url")
                result["created_at"] = doc.data.get("created_at")

            enriched_results.append(result)

        return enriched_results

    async def store_memory(
        self,
        org_id: str,
        title: str,
        content: str,
        memory_type: str = "conversation",
        importance: float = 0.5,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Store a memory entry after a conversation

        Args:
            org_id: Organization ID
            title: Short title/summary
            content: Full content
            memory_type: Type of memory (conversation, insight, decision, update)
            importance: Importance score 0-1
            user_id: Optional user ID
            metadata: Optional metadata dict
        """
        try:
            memory_data = {
                "org_id": org_id,
                "user_id": user_id,
                "memory_type": memory_type,
                "title": title,
                "content": content,
                "importance": max(0.0, min(1.0, importance)),  # Clamp to [0, 1]
                "metadata": metadata or {},
            }

            result = self.supabase.table("ai_memory").insert(memory_data).execute()

            if DEBUG:
                print(
                    f"[MEMORY STORED] Type: {memory_type}, Importance: {importance:.2f}, Title: {title}"
                )

            return result.data[0] if result.data else None

        except Exception as e:
            if DEBUG:
                print(f"[MEMORY ERROR] Failed to store memory: {e}")
            return None

    async def generate_answer_stream(
        self,
        query: str,
        org_id: str,
        max_context_items: int = 5,
        system_prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming answer using RAG with memory layer and conversation history

        Args:
            query: User's question
            org_id: Organization ID
            max_context_items: Max context items to retrieve
            system_prompt: Optional custom system prompt
            user_id: Optional user ID for personalized memories
            conversation_history: Optional list of previous messages [{"role": "user", "content": "..."}, ...]

        Yields:
            Answer chunks as they're generated
        """
        # Check if this is a simple greeting/small talk (don't need context for these)
        simple_queries = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "ok", "okay"]
        is_simple_query = query.lower().strip() in simple_queries or len(query.strip()) < 10

        # Step 1: Retrieve important memories first
        memories = self.retrieve_memory(org_id, user_id, limit=3)

        # Step 2: Retrieve context (skip for simple queries)
        context_items = []
        if not is_simple_query:
            context_items = self.retrieve_context(
                query=query,
                org_id=org_id,
                limit=max_context_items,
                threshold=0.3,  # Lower threshold for better recall
            )

        # For simple queries, just respond naturally without needing context
        if is_simple_query:
            pass  # Will use default prompt below
        elif not context_items and not memories:
            yield "I don't have information about that yet. Could you provide more context or rephrase your question?"
            return

        # Format memory context
        memory_text = ""
        if memories:
            memory_parts = []
            for mem in memories:
                memory_parts.append(
                    f"[Recent Memory] {mem['title']}: {mem['content']}"
                )
            memory_text = "\n\n".join(memory_parts)

        # Format document context with metadata
        context_parts = []
        for i, item in enumerate(context_items):
            source_text = f"[Source {i+1}]"

            # Add author if available
            if item.get("author"):
                source_text += f" Author: {item['author']}"

            # Add channel if available
            if item.get("channel_name"):
                source_text += f" | Channel: #{item['channel_name']}"

            # Add content
            source_text += f"\n{item['content']}"

            context_parts.append(source_text)

        context_text = "\n\n".join(context_parts)

        # Combine memory and context
        full_context = ""
        if memory_text:
            full_context += f"=== Important Recent Context ===\n{memory_text}\n\n"
        if context_text:
            full_context += f"=== Retrieved Information ===\n{context_text}"

        # Default system prompt
        if not system_prompt:
            system_prompt = """You are Vibodh, an intelligent AI team member who has complete knowledge of this company's conversations, projects, and work.

You have direct access to all company Slack channels, discussions, and documents. You are NOT a search assistant - you are a knowledgeable colleague who remembers everything that's been said and done.

Guidelines for interaction:
- Be BRIEF and CONVERSATIONAL - respond like a real person would in Slack
- For greetings like "Hi" or "Hello", keep it SHORT (1-2 sentences max): "Hey! How can I help you?" or "Hi there! What's up?"
- Don't info-dump or list everything you know unless specifically asked
- Speak naturally and confidently, as if you were in the room when these conversations happened
- NEVER say "according to the knowledge base" or "I found in the context" - you simply KNOW these things
- When someone asks "who said X?", just answer with the name: "Mike T" or "That was Yusuf"
- When asked about specific topics, be helpful but concise - don't ramble
- You can have opinions and make suggestions based on what you know about the team's goals and challenges
- If you truly don't have information about something, say "I don't have information about that"
- Be conversational, friendly, and act like you're part of the team
- Match the user's energy level - if they're brief, be brief back

Remember: You're not searching files - you're a smart team member. Act natural and keep responses concise unless asked for details."""

        # Build messages with conversation history
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history if available (exclude the current query if it's in history)
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # For simple queries, don't include context in the prompt
        if is_simple_query:
            messages.append({
                "role": "user",
                "content": query,
            })
        elif full_context:
            messages.append({
                "role": "user",
                "content": f"""Context from company knowledge base:

{full_context}

Question: {query}

Please answer based on the above context.""",
            })
        else:
            messages.append({
                "role": "user",
                "content": query,
            })

        # Stream the response
        full_response = ""
        async for chunk in self.llm_service.generate_streaming_response(messages):
            full_response += chunk
            yield chunk

        # Step 3: Post-answer summarization - store important insights
        # Only store if the response is substantial and not a simple greeting
        if not is_simple_query and len(full_response) > 50 and len(query) > 10:
            # Generate a short summary title (first 100 chars of response)
            summary_title = full_response[:100].strip()
            if len(summary_title) > 97:
                summary_title = summary_title[:97] + "..."

            # Determine importance based on response length and query complexity
            importance = min(
                0.5 + (len(full_response) / 1000) * 0.3, 0.9
            )  # Max 0.9

            await self.store_memory(
                org_id=org_id,
                user_id=user_id,
                title=summary_title,
                content=f"Q: {query}\nA: {full_response}",
                memory_type="conversation",
                importance=importance,
                metadata={
                    "query_length": len(query),
                    "response_length": len(full_response),
                    "context_items_used": len(context_items),
                    "memories_used": len(memories),
                },
            )


def get_rag_service(supabase_client: Client) -> RAGService:
    """Get RAG service instance"""
    return RAGService(supabase_client)


def invalidate_embeddings_cache(org_id: str):
    """Invalidate embeddings cache for an organization (call after new ingestion)"""
    _embeddings_cache.invalidate(org_id)
