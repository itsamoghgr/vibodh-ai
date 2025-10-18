"""
Orchestrator API Routes
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from uuid import UUID

from app.services import get_orchestrator_service
from app.db import supabase
from app.models.query import QueryRequest, QueryResponse
from app.core.logging import logger, log_error

router = APIRouter(prefix="/orchestrate", tags=["Orchestrator"])


@router.post("/query", response_model=QueryResponse)
async def orchestrate_query(request: QueryRequest):
    """
    Orchestrate a query through the cognitive core.

    Routes the query to appropriate modules (RAG, KG, Insights)
    and generates a synthesized response.
    """
    try:
        logger.info(f"Orchestrating query: {request.query[:50]}...")

        orchestrator = get_orchestrator_service(supabase)
        result = await orchestrator.orchestrate_query(
            query=request.query,
            org_id=str(request.org_id),
            user_id=str(request.user_id) if request.user_id else None
        )

        return QueryResponse(
            query=request.query,
            intent=result.get("intent"),
            modules_used=result.get("modules_used"),
            final_answer=result.get("final_answer"),
            context_sources=result.get("context_sources"),
            execution_time_ms=result.get("execution_time_ms")
        )

    except Exception as e:
        log_error(e, context="Orchestrator query")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_reasoning_logs(
    org_id: UUID = Query(..., description="Organization ID"),
    limit: int = Query(10, ge=1, le=100, description="Number of logs to retrieve")
):
    """
    Get reasoning logs for an organization.

    Returns the AI's reasoning steps for recent queries.
    """
    try:
        logger.info(f"Fetching {limit} reasoning logs for org {org_id}")

        logs = supabase.table("reasoning_logs")\
            .select("*")\
            .eq("org_id", str(org_id))\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()

        return {
            "logs": logs.data,
            "count": len(logs.data)
        }

    except Exception as e:
        log_error(e, context="Get reasoning logs")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/{log_id}")
async def get_reasoning_log_detail(log_id: UUID):
    """
    Get detailed reasoning log by ID.
    """
    try:
        logger.info(f"Fetching reasoning log {log_id}")

        log = supabase.table("reasoning_logs")\
            .select("*")\
            .eq("id", str(log_id))\
            .single()\
            .execute()

        if not log.data:
            raise HTTPException(status_code=404, detail="Log not found")

        return log.data

    except Exception as e:
        log_error(e, context="Get reasoning log detail")
        raise HTTPException(status_code=500, detail=str(e))
