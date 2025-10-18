"""
Insights API Routes
"""

from fastapi import APIRouter, Query, HTTPException
from uuid import UUID
from typing import Optional

from app.db import supabase
from app.core.logging import logger, log_error
from app.services.insight_service import get_insight_service

router = APIRouter(prefix="/insights", tags=["Insights"])


@router.post("/run/{org_id}")
async def generate_insights(
    org_id: str,
    days: int = Query(default=7, ge=1, le=30, description="Number of days to analyze")
):
    """
    Generate AI insights for an organization

    Analyzes recent KG activity, memory summaries, and documents to generate
    actionable insights about projects, teams, trends, and risks.
    """
    try:
        insight_service = get_insight_service(supabase)
        result = insight_service.generate_insights(org_id, days)
        return result

    except Exception as e:
        log_error(e, context="Generate insights")
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@router.get("/list/{org_id}")
async def list_insights(
    org_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of insights to return"),
    category: Optional[str] = Query(default=None, description="Filter by category: project, team, trend, risk, general")
):
    """
    List recent AI insights for an organization

    Returns insights sorted by creation date (most recent first).
    Optionally filter by category.
    """
    try:
        insight_service = get_insight_service(supabase)
        insights = insight_service.list_insights(org_id, limit, category)

        return {
            "insights": insights,
            "count": len(insights)
        }

    except Exception as e:
        log_error(e, context="List insights")
        raise HTTPException(status_code=500, detail=f"Failed to list insights: {str(e)}")


@router.get("/stats/{org_id}")
async def get_insight_stats(org_id: str):
    """
    Get insight statistics for an organization

    Returns counts by category, average confidence, and last generation time.
    """
    try:
        insight_service = get_insight_service(supabase)
        stats = insight_service.get_insight_stats(org_id)
        return stats

    except Exception as e:
        log_error(e, context="Get insight stats")
        raise HTTPException(status_code=500, detail=f"Failed to get insight stats: {str(e)}")
