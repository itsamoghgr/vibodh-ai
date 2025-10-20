"""
Memory Service - Phase 3, Step 2
Manages AI memory layer for long-term reasoning and context retention
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from supabase import Client
import numpy as np

from app.services.embedding_service import get_embedding_service
from app.services.llm_service import get_llm_service
from app.core.logging import logger


class MemoryService:
    """
    Memory Service for storing and retrieving organizational memories.

    Supports:
    - Semantic memory search via embeddings
    - Importance scoring and decay
    - Memory consolidation
    - Deduplication
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.embedding_service = get_embedding_service()
        self.llm_service = get_llm_service()

    async def store_memory(
        self,
        org_id: str,
        title: str,
        content: str,
        memory_type: str = "conversation",
        importance: float = 0.5,
        user_id: Optional[str] = None,
        source_refs: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        expires_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Store a new memory with embedding.

        Args:
            org_id: Organization ID
            title: Memory title
            content: Memory content
            memory_type: conversation, insight, decision, update
            importance: 0.0-1.0 importance score
            user_id: Optional user ID
            source_refs: Optional source references
            metadata: Optional metadata
            expires_at: Optional expiration date

        Returns:
            Created memory record
        """
        try:
            logger.info(f"Storing memory for org {org_id}: {title}")

            # Generate embedding for content
            embeddings = await self.embedding_service.generate_embeddings([content])
            embedding = embeddings[0] if embeddings else None

            # Prepare memory data
            memory_data = {
                "org_id": org_id,
                "user_id": user_id,
                "memory_type": memory_type,
                "title": title,
                "content": content,
                "importance": max(0.0, min(1.0, importance)),  # Clamp to 0-1
                "embedding": embedding,
                "source_refs": source_refs or [],
                "metadata": metadata or {},
                "expires_at": expires_at.isoformat() if expires_at else None,
                "access_count": 0,
                "last_accessed_at": datetime.utcnow().isoformat()
            }

            # Insert into database
            result = self.supabase.table("ai_memory").insert(memory_data).execute()

            if result.data:
                logger.info(f"Memory stored successfully: {result.data[0]['id']}")
                return result.data[0]
            else:
                raise Exception("Failed to store memory")

        except Exception as e:
            logger.error(f"Error storing memory: {e}", exc_info=True)
            raise

    async def retrieve_relevant_memories(
        self,
        org_id: str,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.3,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to query using semantic search.

        Args:
            org_id: Organization ID
            query: Search query
            limit: Maximum number of memories to return
            memory_types: Optional filter by memory types
            min_importance: Minimum importance threshold
            user_id: Optional filter by user_id

        Returns:
            List of relevant memories
        """
        try:
            logger.info(f"Retrieving memories for query: {query[:50]}...")

            # Generate query embedding
            query_embeddings = await self.embedding_service.generate_embeddings([query])
            query_embedding = query_embeddings[0] if query_embeddings else None

            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []

            # Build query
            query_builder = self.supabase.table("ai_memory")\
                .select("*")\
                .eq("org_id", org_id)\
                .gte("importance", min_importance)\
                .is_("expires_at", "null")  # Only non-expired memories

            if memory_types:
                query_builder = query_builder.in_("memory_type", memory_types)

            if user_id:
                query_builder = query_builder.eq("user_id", user_id)

            # Get all memories (we'll do cosine similarity in Python)
            result = query_builder.execute()

            if not result.data:
                return []

            # Calculate cosine similarity
            memories_with_similarity = []
            for memory in result.data:
                if memory.get("embedding"):
                    similarity = self._cosine_similarity(
                        query_embedding,
                        memory["embedding"]
                    )
                    memory["similarity"] = similarity
                    memories_with_similarity.append(memory)

            # Sort by similarity and get top results
            memories_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)
            top_memories = memories_with_similarity[:limit]

            # Update access counts
            for memory in top_memories:
                self._update_memory_access(memory["id"])

            logger.info(f"Retrieved {len(top_memories)} relevant memories")
            return top_memories

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a = np.array(vec1)
            b = np.array(vec2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except:
            return 0.0

    def _update_memory_access(self, memory_id: str):
        """Update access count and last_accessed_at for a memory."""
        try:
            self.supabase.rpc(
                "increment_memory_access",
                {"memory_id": memory_id}
            ).execute()
        except:
            # Fallback to manual update
            try:
                self.supabase.table("ai_memory")\
                    .update({
                        "access_count": self.supabase.table("ai_memory")
                            .select("access_count")
                            .eq("id", memory_id)
                            .single()
                            .execute()
                            .data["access_count"] + 1,
                        "last_accessed_at": datetime.utcnow().isoformat()
                    })\
                    .eq("id", memory_id)\
                    .execute()
            except Exception as e:
                logger.warning(f"Failed to update memory access: {e}")

    def calculate_decayed_importance(
        self,
        original_importance: float,
        created_at: datetime,
        access_count: int,
        last_accessed_at: datetime
    ) -> float:
        """
        Calculate importance with time decay and access boost.

        Algorithm:
        - Base decay: 10% per week if not accessed
        - Access boost: +5% per access, max 50%
        - Age penalty: -20% after 90 days
        """
        try:
            now = datetime.utcnow()
            days_old = (now - created_at).days
            days_since_access = (now - last_accessed_at).days

            # Time decay: 10% per week
            time_decay = 0.9 ** (days_since_access / 7)

            # Access boost: +5% per access, max 50%
            access_boost = min(0.5, access_count * 0.05)

            # Age penalty: -20% after 90 days
            age_penalty = 0.8 if days_old > 90 else 1.0

            new_importance = original_importance * time_decay * age_penalty + access_boost

            return max(0.0, min(1.0, new_importance))

        except Exception as e:
            logger.error(f"Error calculating decayed importance: {e}")
            return original_importance

    async def update_importance_scores(self, org_id: str) -> int:
        """
        Update importance scores for all memories using decay algorithm.

        Args:
            org_id: Organization ID

        Returns:
            Number of memories updated
        """
        try:
            logger.info(f"Updating importance scores for org {org_id}")

            # Get all non-expired memories
            result = self.supabase.table("ai_memory")\
                .select("id, importance, created_at, access_count, last_accessed_at")\
                .eq("org_id", org_id)\
                .is_("expires_at", "null")\
                .execute()

            if not result.data:
                return 0

            updated_count = 0
            for memory in result.data:
                # Calculate new importance
                new_importance = self.calculate_decayed_importance(
                    memory["importance"],
                    datetime.fromisoformat(memory["created_at"].replace('Z', '+00:00')),
                    memory.get("access_count", 0),
                    datetime.fromisoformat(memory.get("last_accessed_at", memory["created_at"]).replace('Z', '+00:00'))
                )

                # Update if changed significantly (>5%)
                if abs(new_importance - memory["importance"]) > 0.05:
                    self.supabase.table("ai_memory")\
                        .update({"importance": new_importance})\
                        .eq("id", memory["id"])\
                        .execute()
                    updated_count += 1

            logger.info(f"Updated {updated_count} memory importance scores")
            return updated_count

        except Exception as e:
            logger.error(f"Error updating importance scores: {e}", exc_info=True)
            return 0

    async def consolidate_memories(
        self,
        org_id: str,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        Consolidate short-term memories into long-term summaries.

        Args:
            org_id: Organization ID
            lookback_days: Days to look back for consolidation

        Returns:
            Consolidation statistics
        """
        try:
            logger.info(f"Consolidating memories for org {org_id}")

            # Get recent conversation/update memories
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
            result = self.supabase.table("ai_memory")\
                .select("*")\
                .eq("org_id", org_id)\
                .in_("memory_type", ["conversation", "update"])\
                .gte("created_at", cutoff_date.isoformat())\
                .execute()

            if not result.data or len(result.data) < 3:
                logger.info("Not enough memories to consolidate")
                return {"consolidated": 0, "created": 0}

            # Group by topic using LLM
            memories_text = "\n\n".join([
                f"Title: {m['title']}\nContent: {m['content']}"
                for m in result.data
            ])

            # Generate consolidated summary
            prompt = f"""Analyze these recent memories and create a consolidated summary:

{memories_text}

Create a concise summary that:
1. Groups related topics together
2. Identifies key decisions and outcomes
3. Highlights important patterns or trends
4. Is 3-5 sentences long

Summary:"""

            summary = await self.llm_service.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=500
            )

            # Store as long-term memory (insight type)
            consolidated_memory = await self.store_memory(
                org_id=org_id,
                title=f"Weekly Summary - {datetime.utcnow().strftime('%Y-%m-%d')}",
                content=summary,
                memory_type="insight",
                importance=0.8,
                source_refs=[{"memory_id": m["id"]} for m in result.data],
                metadata={"consolidated_from": len(result.data), "period_days": lookback_days}
            )

            logger.info(f"Created consolidated memory: {consolidated_memory['id']}")

            return {
                "consolidated": len(result.data),
                "created": 1,
                "memory_id": consolidated_memory["id"]
            }

        except Exception as e:
            logger.error(f"Error consolidating memories: {e}", exc_info=True)
            return {"consolidated": 0, "created": 0, "error": str(e)}

    def get_memory_stats(self, org_id: str) -> Dict[str, Any]:
        """
        Get memory statistics for organization.

        Args:
            org_id: Organization ID

        Returns:
            Memory statistics
        """
        try:
            # Total memories
            total = self.supabase.table("ai_memory")\
                .select("id", count="exact")\
                .eq("org_id", org_id)\
                .execute()

            # By type
            by_type = self.supabase.table("ai_memory")\
                .select("memory_type", count="exact")\
                .eq("org_id", org_id)\
                .execute()

            # Average importance
            avg_importance = self.supabase.table("ai_memory")\
                .select("importance")\
                .eq("org_id", org_id)\
                .execute()

            avg_imp = np.mean([m["importance"] for m in avg_importance.data]) if avg_importance.data else 0

            # This week
            week_ago = datetime.utcnow() - timedelta(days=7)
            this_week = self.supabase.table("ai_memory")\
                .select("id", count="exact")\
                .eq("org_id", org_id)\
                .gte("created_at", week_ago.isoformat())\
                .execute()

            return {
                "total_memories": total.count if hasattr(total, 'count') else len(total.data),
                "by_type": by_type.data if by_type.data else [],
                "average_importance": round(avg_imp, 2),
                "created_this_week": this_week.count if hasattr(this_week, 'count') else len(this_week.data)
            }

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}", exc_info=True)
            return {
                "total_memories": 0,
                "by_type": [],
                "average_importance": 0,
                "created_this_week": 0
            }

    async def delete_expired_memories(self, org_id: str) -> int:
        """
        Delete memories that have passed their expiration date.

        Args:
            org_id: Organization ID

        Returns:
            Number of memories deleted
        """
        try:
            result = self.supabase.table("ai_memory")\
                .delete()\
                .eq("org_id", org_id)\
                .lt("expires_at", datetime.utcnow().isoformat())\
                .execute()

            deleted_count = len(result.data) if result.data else 0
            logger.info(f"Deleted {deleted_count} expired memories")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting expired memories: {e}", exc_info=True)
            return 0


# Factory function
def get_memory_service(supabase: Client) -> MemoryService:
    """Get or create MemoryService instance."""
    return MemoryService(supabase)
