"""
Knowledge Graph API Routes
"""

from fastapi import APIRouter, Query, HTTPException
from uuid import UUID

from app.db import supabase
from app.core.logging import logger, log_error
from app.services.kg_service import get_kg_service

router = APIRouter(prefix="/kg", tags=["Knowledge Graph"])


@router.get("/stats/{org_id}")
async def get_kg_stats(org_id: str):
    """
    Get knowledge graph statistics for an organization
    """
    try:
        kg_service = get_kg_service(supabase)
        stats = kg_service.get_kg_stats(org_id)
        return stats

    except Exception as e:
        log_error(e, context="Get KG stats")
        raise HTTPException(status_code=500, detail=f"Failed to get KG stats: {str(e)}")


@router.get("/entities/{org_id}")
async def get_entities(
    org_id: str,
    entity_type: str = Query(None, description="Filter by entity type"),
    limit: int = Query(default=50, ge=1, le=200)
):
    """
    Get entities from the knowledge graph
    """
    try:
        query = supabase.table("kg_entities")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)

        if entity_type:
            query = query.eq("type", entity_type)

        result = query.execute()

        return {
            "entities": result.data if result.data else [],
            "count": len(result.data) if result.data else 0
        }

    except Exception as e:
        log_error(e, context="Get entities")
        raise HTTPException(status_code=500, detail=f"Failed to get entities: {str(e)}")


@router.get("/edges/{org_id}")
async def get_edges(
    org_id: str,
    relation: str = Query(None, description="Filter by relation type"),
    limit: int = Query(default=50, ge=1, le=200)
):
    """
    Get relationship edges from the knowledge graph
    """
    try:
        query = supabase.table("kg_edges")\
            .select("*, source:kg_entities!kg_edges_source_id_fkey(name, type), target:kg_entities!kg_edges_target_id_fkey(name, type)")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)

        if relation:
            query = query.eq("relation", relation)

        result = query.execute()

        return {
            "edges": result.data if result.data else [],
            "count": len(result.data) if result.data else 0
        }

    except Exception as e:
        log_error(e, context="Get edges")
        raise HTTPException(status_code=500, detail=f"Failed to get edges: {str(e)}")
