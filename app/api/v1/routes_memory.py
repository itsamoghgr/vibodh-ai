"""
Memory API Routes
Phase 3, Step 2: Memory Layer endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

from app.db import supabase
from app.services.memory_service import get_memory_service
from app.core.logging import logger

router = APIRouter(prefix="/memory", tags=["Memory"])


# Request/Response Models
class MemoryCreateRequest(BaseModel):
    """Request to create a new memory"""
    org_id: str
    title: str
    content: str
    memory_type: str = Field(default="conversation", description="conversation, insight, decision, update")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    user_id: Optional[str] = None
    source_refs: Optional[List[dict]] = None
    metadata: Optional[dict] = None
    expires_at: Optional[str] = None


class MemorySearchRequest(BaseModel):
    """Request to search memories"""
    org_id: str
    query: str
    limit: int = Field(default=5, ge=1, le=20)
    memory_types: Optional[List[str]] = None
    min_importance: float = Field(default=0.3, ge=0.0, le=1.0)
    user_id: Optional[str] = None


@router.post("/create")
async def create_memory(request: MemoryCreateRequest):
    """
    Create a new memory with embedding generation
    """
    try:
        memory_service = get_memory_service(supabase)

        # Parse expires_at if provided
        expires_at = None
        if request.expires_at:
            try:
                expires_at = datetime.fromisoformat(request.expires_at.replace('Z', '+00:00'))
            except:
                pass

        memory = await memory_service.store_memory(
            org_id=request.org_id,
            title=request.title,
            content=request.content,
            memory_type=request.memory_type,
            importance=request.importance,
            user_id=request.user_id,
            source_refs=request.source_refs,
            metadata=request.metadata,
            expires_at=expires_at
        )

        return {"success": True, "memory": memory}

    except Exception as e:
        logger.error(f"Create memory error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_memories(request: MemorySearchRequest):
    """
    Search memories using semantic similarity
    """
    try:
        memory_service = get_memory_service(supabase)

        memories = await memory_service.retrieve_relevant_memories(
            org_id=request.org_id,
            query=request.query,
            limit=request.limit,
            memory_types=request.memory_types,
            min_importance=request.min_importance,
            user_id=request.user_id
        )

        return {
            "success": True,
            "count": len(memories),
            "memories": memories
        }

    except Exception as e:
        logger.error(f"Search memories error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list/{org_id}")
async def list_memories(
    org_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    memory_type: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    List memories for an organization (paginated)
    """
    try:
        query = supabase.table("ai_memory")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .range(offset, offset + limit - 1)

        if memory_type:
            query = query.eq("memory_type", memory_type)

        if user_id:
            query = query.eq("user_id", user_id)

        result = query.execute()

        return {
            "success": True,
            "count": len(result.data) if result.data else 0,
            "memories": result.data or []
        }

    except Exception as e:
        logger.error(f"List memories error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{org_id}")
async def get_memory_stats(org_id: str):
    """
    Get memory statistics for an organization
    """
    try:
        memory_service = get_memory_service(supabase)
        stats = memory_service.get_memory_stats(org_id)

        return {
            "success": True,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Get memory stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-importance/{org_id}")
async def update_importance_scores(org_id: str):
    """
    Update importance scores using decay algorithm for all memories
    """
    try:
        memory_service = get_memory_service(supabase)
        updated_count = await memory_service.update_importance_scores(org_id)

        return {
            "success": True,
            "updated_count": updated_count
        }

    except Exception as e:
        logger.error(f"Update importance error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate/{org_id}")
async def consolidate_memories(
    org_id: str,
    lookback_days: int = Query(default=7, ge=1, le=30)
):
    """
    Consolidate recent memories into a long-term summary
    """
    try:
        memory_service = get_memory_service(supabase)
        result = await memory_service.consolidate_memories(org_id, lookback_days)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        logger.error(f"Consolidate memories error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/expired/{org_id}")
async def delete_expired_memories(org_id: str):
    """
    Delete memories that have passed their expiration date
    """
    try:
        memory_service = get_memory_service(supabase)
        deleted_count = await memory_service.delete_expired_memories(org_id)

        return {
            "success": True,
            "deleted_count": deleted_count
        }

    except Exception as e:
        logger.error(f"Delete expired memories error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{memory_id}")
async def get_memory(memory_id: str):
    """
    Get a specific memory by ID
    """
    try:
        result = supabase.table("ai_memory")\
            .select("*")\
            .eq("id", memory_id)\
            .single()\
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Memory not found")

        return {
            "success": True,
            "memory": result.data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get memory error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """
    Delete a specific memory by ID
    """
    try:
        result = supabase.table("ai_memory")\
            .delete()\
            .eq("id", memory_id)\
            .execute()

        return {
            "success": True,
            "message": "Memory deleted successfully"
        }

    except Exception as e:
        logger.error(f"Delete memory error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
