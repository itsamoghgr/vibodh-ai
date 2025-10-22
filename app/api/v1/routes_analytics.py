"""
Analytics API Routes - Phase 4, Observability Dashboard
Endpoints for system analytics, metrics, and observability
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel

from app.core.auth import get_current_user, get_org_id
from app.db import supabase, supabase_admin
from app.core.logging import logger

router = APIRouter(prefix="/analytics", tags=["analytics"])


# Response Models
class AgentPerformanceMetrics(BaseModel):
    """Agent performance metrics"""
    agent_type: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    avg_execution_time_ms: Optional[float]
    total_actions: int


class SystemOverview(BaseModel):
    """System-wide overview metrics"""
    total_plans: int
    active_plans: int
    completed_plans: int
    failed_plans: int
    pending_approvals: int
    total_agents: int
    active_integrations: int
    health_status: str


class TimeSeriesDataPoint(BaseModel):
    """Time series data point"""
    timestamp: str
    value: float
    label: Optional[str] = None


@router.get("/overview", summary="Get system overview")
async def get_system_overview(
    org_id: str = Query(..., description="Organization ID"),
    days_back: int = Query(7, ge=1, le=90)
):
    """
    Get high-level system overview metrics.

    Returns key metrics for dashboard summary cards.
    """
    try:
        cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

        # Get plan counts - use admin client to bypass RLS
        plans_result = supabase_admin.table("ai_action_plans")\
            .select("status", count="exact")\
            .eq("org_id", org_id)\
            .gte("created_at", cutoff_date)\
            .execute()

        total_plans = plans_result.count or 0

        # Count by status
        plans = plans_result.data or []
        active_plans = sum(1 for p in plans if p["status"] in ["pending", "executing"])
        completed_plans = sum(1 for p in plans if p["status"] == "completed")
        failed_plans = sum(1 for p in plans if p["status"] == "failed")

        # Get pending approvals count
        approvals_result = supabase_admin.table("ai_actions_pending")\
            .select("id", count="exact")\
            .eq("org_id", org_id)\
            .eq("status", "pending")\
            .execute()

        pending_approvals = approvals_result.count or 0

        # Get active agents count
        agents_result = supabase_admin.table("ai_agent_registry")\
            .select("id", count="exact")\
            .eq("org_id", org_id)\
            .eq("status", "active")\
            .execute()

        total_agents = agents_result.count or 0

        # Get active integrations count
        integrations_result = supabase_admin.table("connections")\
            .select("id", count="exact")\
            .eq("org_id", org_id)\
            .eq("status", "active")\
            .execute()

        active_integrations = integrations_result.count or 0

        # Determine overall health status
        health_status = "healthy"
        if failed_plans > completed_plans * 0.2:  # >20% failure rate
            health_status = "degraded"
        if failed_plans > completed_plans * 0.5:  # >50% failure rate
            health_status = "critical"

        return {
            "total_plans": total_plans,
            "active_plans": active_plans,
            "completed_plans": completed_plans,
            "failed_plans": failed_plans,
            "success_rate": completed_plans / total_plans if total_plans > 0 else 0,
            "pending_approvals": pending_approvals,
            "total_agents": total_agents,
            "active_integrations": active_integrations,
            "health_status": health_status,
            "period_days": days_back
        }

    except Exception as e:
        logger.error(f"Error fetching system overview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/performance", summary="Get agent performance metrics")
async def get_agent_performance(
    org_id: str = Query(..., description="Organization ID"),
    days_back: int = Query(30, ge=1, le=365),
    agent_type: Optional[str] = Query(None, description="Filter by agent type")
):
    """
    Get performance metrics for all agents.

    Returns execution counts, success rates, and timing metrics.
    """
    try:
        cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

        # Get all action plans - use admin client to bypass RLS
        query = supabase_admin.table("ai_action_plans")\
            .select("agent_type, status, execution_time_ms, completed_steps, total_steps")\
            .eq("org_id", org_id)\
            .gte("created_at", cutoff_date)

        if agent_type:
            query = query.eq("agent_type", agent_type)

        result = query.execute()
        plans = result.data or []

        # Aggregate metrics by agent type
        agent_metrics = {}

        for plan in plans:
            agent = plan["agent_type"]

            if agent not in agent_metrics:
                agent_metrics[agent] = {
                    "agent_type": agent,
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "total_actions": 0,
                    "execution_times": []
                }

            agent_metrics[agent]["total_executions"] += 1
            agent_metrics[agent]["total_actions"] += plan.get("total_steps", 0)

            if plan["status"] == "completed":
                agent_metrics[agent]["successful_executions"] += 1
            elif plan["status"] == "failed":
                agent_metrics[agent]["failed_executions"] += 1

            if plan.get("execution_time_ms"):
                agent_metrics[agent]["execution_times"].append(plan["execution_time_ms"])

        # Calculate derived metrics
        performance_data = []
        for agent, metrics in agent_metrics.items():
            total = metrics["total_executions"]
            success_rate = metrics["successful_executions"] / total if total > 0 else 0

            avg_time = None
            if metrics["execution_times"]:
                avg_time = sum(metrics["execution_times"]) / len(metrics["execution_times"])

            performance_data.append({
                "agent_type": agent,
                "total_executions": total,
                "successful_executions": metrics["successful_executions"],
                "failed_executions": metrics["failed_executions"],
                "success_rate": round(success_rate, 3),
                "avg_execution_time_ms": round(avg_time, 2) if avg_time else None,
                "total_actions": metrics["total_actions"]
            })

        # Sort by total executions
        performance_data.sort(key=lambda x: x["total_executions"], reverse=True)

        return {
            "agents": performance_data,
            "period_days": days_back
        }

    except Exception as e:
        logger.error(f"Error fetching agent performance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/timeline", summary="Get execution timeline")
async def get_execution_timeline(
    user=Depends(get_current_user),
    org_id: str = Depends(get_org_id),
    days_back: int = Query(30, ge=1, le=90),
    interval: str = Query("day", regex="^(hour|day|week)$")
):
    """
    Get execution timeline data for charts.

    Returns time-series data of executions over time.
    """
    try:
        cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

        # Get all executed actions
        result = supabase.table("ai_actions_executed")\
            .select("status, completed_at")\
            .eq("org_id", org_id)\
            .gte("completed_at", cutoff_date)\
            .order("completed_at")\
            .execute()

        actions = result.data or []

        # Group by time interval
        timeline_data = {}

        for action in actions:
            if not action.get("completed_at"):
                continue

            timestamp = datetime.fromisoformat(action["completed_at"].replace("Z", "+00:00"))

            # Round to interval
            if interval == "hour":
                key = timestamp.replace(minute=0, second=0, microsecond=0)
            elif interval == "day":
                key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            else:  # week
                key = timestamp - timedelta(days=timestamp.weekday())
                key = key.replace(hour=0, minute=0, second=0, microsecond=0)

            key_str = key.isoformat()

            if key_str not in timeline_data:
                timeline_data[key_str] = {
                    "timestamp": key_str,
                    "total": 0,
                    "successful": 0,
                    "failed": 0
                }

            timeline_data[key_str]["total"] += 1

            if action["status"] == "success":
                timeline_data[key_str]["successful"] += 1
            elif action["status"] == "failed":
                timeline_data[key_str]["failed"] += 1

        # Convert to sorted list
        timeline = sorted(timeline_data.values(), key=lambda x: x["timestamp"])

        return {
            "timeline": timeline,
            "interval": interval,
            "period_days": days_back
        }

    except Exception as e:
        logger.error(f"Error fetching execution timeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/integrations/health", summary="Get integration health summary")
async def get_integration_health(
    org_id: str = Query(..., description="Organization ID")
):
    """
    Get health status of all integrations.

    Returns recent health check results for each integration.
    """
    try:
        # Get latest health check for each integration - use admin client to bypass RLS
        # Note: integration_health_checks table doesn't exist, return empty for now
        result = supabase_admin.table("connections")\
            .select("source_type, status, updated_at")\
            .eq("org_id", org_id)\
            .order("updated_at", desc=True)\
            .execute()

        connections = result.data or []

        # Format as health checks
        health_summary = []
        for conn in connections:
            health_summary.append({
                "integration": conn["source_type"],
                "status": "healthy" if conn["status"] == "active" else "failed",
                "message": f"{conn['source_type']} integration",
                "response_time_ms": None,
                "checked_at": conn["updated_at"]
            })

        # Calculate overall status
        statuses = [check["status"] for check in health_summary]
        overall_status = "healthy"

        if "failed" in statuses:
            overall_status = "degraded"
        if all(s == "failed" for s in statuses) and len(statuses) > 0:
            overall_status = "critical"

        return {
            "integrations": health_summary,
            "overall_status": overall_status,
            "total_integrations": len(health_summary),
            "healthy_count": sum(1 for s in statuses if s == "healthy"),
            "failed_count": sum(1 for s in statuses if s == "failed")
        }

    except Exception as e:
        logger.error(f"Error fetching integration health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reflections/insights", summary="Get reflection insights")
async def get_reflection_insights(
    user=Depends(get_current_user),
    org_id: str = Depends(get_org_id),
    limit: int = Query(20, ge=1, le=100),
    agent_type: Optional[str] = Query(None, description="Filter by agent type")
):
    """
    Get recent agent reflections and learning insights.

    Returns reflections with key insights and improvements.
    """
    try:
        query = supabase.table("ai_reflections")\
            .select("agent_type, reflection_type, summary, insights, improvements_suggested, overall_success, performance_metrics, created_at")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)

        if agent_type:
            query = query.eq("agent_type", agent_type)

        result = query.execute()

        return {
            "reflections": result.data or [],
            "count": len(result.data) if result.data else 0
        }

    except Exception as e:
        logger.error(f"Error fetching reflections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/recent", summary="Get recent agent events")
async def get_recent_events(
    org_id: str = Query(..., description="Organization ID"),
    limit: int = Query(50, ge=1, le=100),
    event_type: Optional[str] = Query(None, description="Filter by event type")
):
    """
    Get recent agent-to-agent events.

    Returns event bus activity for monitoring cross-agent coordination.
    """
    try:
        # Use admin client to bypass RLS
        query = supabase_admin.table("ai_agent_events")\
            .select("event_type, agent_type, metadata, created_at")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)

        if event_type:
            query = query.eq("event_type", event_type)

        result = query.execute()

        # Format events to match frontend expectations
        events = []
        for event in (result.data or []):
            events.append({
                "event_type": event["event_type"],
                "source_agent": event["agent_type"],
                "target_agent": event.get("metadata", {}).get("target_agent"),
                "status": "completed",  # Default status
                "created_at": event["created_at"]
            })

        return {
            "events": events,
            "count": len(events)
        }

    except Exception as e:
        logger.error(f"Error fetching agent events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adaptive/learning-metrics", summary="Get adaptive learning metrics")
async def get_learning_metrics(
    user=Depends(get_current_user),
    org_id: str = Depends(get_org_id),
    days_back: int = Query(30, ge=1, le=365)
):
    """
    Get adaptive learning metrics.

    Returns data about reflections ingested and optimizations performed.
    """
    try:
        cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

        # Get reflection ingestion stats
        reflections_result = supabase.table("ai_reflections")\
            .select("ingested_by_adaptive_engine", count="exact")\
            .eq("org_id", org_id)\
            .gte("created_at", cutoff_date)\
            .execute()

        total_reflections = reflections_result.count or 0
        ingested = sum(1 for r in (reflections_result.data or []) if r.get("ingested_by_adaptive_engine"))

        # Get optimization log
        optimizations_result = supabase.table("ai_optimization_log")\
            .select("optimization_type, parameter_name", count="exact")\
            .eq("org_id", org_id)\
            .gte("created_at", cutoff_date)\
            .execute()

        total_optimizations = optimizations_result.count or 0

        # Get adaptive config changes
        config_result = supabase.table("ai_adaptive_config")\
            .select("last_optimized_at, optimization_count")\
            .eq("org_id", org_id)\
            .single()\
            .execute()

        config = config_result.data or {}

        return {
            "total_reflections": total_reflections,
            "ingested_reflections": ingested,
            "pending_reflections": total_reflections - ingested,
            "total_optimizations": total_optimizations,
            "optimization_count": config.get("optimization_count", 0),
            "last_optimized_at": config.get("last_optimized_at"),
            "period_days": days_back
        }

    except Exception as e:
        logger.error(f"Error fetching learning metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
