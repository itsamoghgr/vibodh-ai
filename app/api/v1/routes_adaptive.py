"""
Adaptive Reasoning API Routes
Phase 3, Step 3: Self-optimization and performance monitoring endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel, Field

from app.db import supabase
from app.services.feedback_service import get_feedback_service
from app.services.adaptive_engine import get_adaptive_engine
from app.core.logging import logger

router = APIRouter(prefix="/adaptive", tags=["Adaptive Reasoning"])


# Request/Response Models
class FeedbackSubmission(BaseModel):
    """User feedback submission"""
    reasoning_log_id: Optional[str] = None
    user_feedback: str = Field(..., description="positive, negative, or neutral")
    feedback_comment: Optional[str] = None


class ManualOptimization(BaseModel):
    """Manual optimization request"""
    days_back: int = Field(default=7, ge=1, le=30)
    dry_run: bool = Field(default=False)


@router.get("/config/{org_id}")
async def get_adaptive_config(org_id: str):
    """
    Get current adaptive configuration for organization.
    """
    try:
        adaptive_engine = get_adaptive_engine(supabase)
        config = adaptive_engine.get_adaptive_config(org_id)

        return {
            "success": True,
            "config": config
        }

    except Exception as e:
        logger.error(f"Get adaptive config error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/{org_id}")
async def submit_feedback(org_id: str, feedback: FeedbackSubmission):
    """
    Submit user feedback for a query/response.
    Updates the ai_feedback_metrics table with user feedback.
    """
    try:
        feedback_service = get_feedback_service(supabase)

        # Get the existing feedback metric by reasoning_log_id
        if feedback.reasoning_log_id:
            result = supabase.table("ai_feedback_metrics")\
                .select("*")\
                .eq("reasoning_log_id", feedback.reasoning_log_id)\
                .eq("org_id", org_id)\
                .execute()

            if result.data and len(result.data) > 0:
                # Update existing feedback metric
                update_data = {
                    "user_feedback": feedback.user_feedback,
                    "feedback_comment": feedback.feedback_comment
                }

                # Recalculate accuracy estimate
                metric = result.data[0]
                feedback_weight = {
                    'positive': 1.0,
                    'negative': 0.0,
                    'neutral': 0.5
                }.get(feedback.user_feedback, 0.5)

                accuracy_estimate = (
                    0.4 * feedback_weight +
                    0.3 * metric.get('confidence_score', 0.5) +
                    0.3 * metric.get('context_relevance_score', 0.5)
                )
                update_data["accuracy_estimate"] = accuracy_estimate

                supabase.table("ai_feedback_metrics")\
                    .update(update_data)\
                    .eq("id", metric["id"])\
                    .execute()

                return {
                    "success": True,
                    "message": "Feedback recorded successfully",
                    "feedback_id": metric["id"]
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Feedback metric not found for this reasoning log"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="reasoning_log_id is required"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit feedback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/summary/{org_id}")
async def get_performance_summary(
    org_id: str,
    days_back: int = Query(default=7, ge=1, le=30)
):
    """
    Get performance summary and trends.
    """
    try:
        feedback_service = get_feedback_service(supabase)
        trends = feedback_service.analyze_feedback_trends(org_id, days_back)

        return {
            "success": True,
            "days_analyzed": days_back,
            "summary": trends.get("summary", {}),
            "module_stats": trends.get("module_stats", []),
            "response_times": trends.get("response_times", []),
            "underperforming_modules": trends.get("underperforming_modules", []),
            "slow_intents": trends.get("slow_intents", []),
            "recommendations": trends.get("recommendations", [])
        }

    except Exception as e:
        logger.error(f"Get performance summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/metrics/{org_id}")
async def get_performance_metrics(
    org_id: str,
    days_back: int = Query(default=7, ge=1, le=30),
    limit: int = Query(default=100, ge=1, le=500)
):
    """
    Get detailed performance metrics over time.
    """
    try:
        # Get feedback metrics ordered by time
        result = supabase.table("ai_feedback_metrics")\
            .select("created_at, user_feedback, response_time_ms, confidence_score, accuracy_estimate, intent")\
            .eq("org_id", org_id)\
            .order("created_at", desc=False)\
            .limit(limit)\
            .execute()

        return {
            "success": True,
            "count": len(result.data) if result.data else 0,
            "metrics": result.data or []
        }

    except Exception as e:
        logger.error(f"Get performance metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/module-stats/{org_id}")
async def get_module_statistics(
    org_id: str,
    days_back: int = Query(default=7, ge=1, le=30)
):
    """
    Get module-specific performance statistics.
    """
    try:
        result = supabase.rpc(
            "get_module_success_rate",
            {"org_uuid": org_id, "days_back": days_back}
        ).execute()

        return {
            "success": True,
            "module_stats": result.data or []
        }

    except Exception as e:
        logger.error(f"Get module stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/{org_id}")
async def run_optimization(org_id: str, request: ManualOptimization):
    """
    Manually trigger optimization for an organization.
    """
    try:
        adaptive_engine = get_adaptive_engine(supabase)

        result = await adaptive_engine.run_full_optimization(
            org_id=org_id,
            days_back=request.days_back,
            dry_run=request.dry_run
        )

        return {
            "success": True,
            "dry_run": request.dry_run,
            "optimization_result": result
        }

    except Exception as e:
        logger.error(f"Run optimization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/weights/{org_id}")
async def optimize_module_weights(org_id: str, request: ManualOptimization):
    """
    Optimize module weights based on performance data.
    """
    try:
        adaptive_engine = get_adaptive_engine(supabase)

        result = await adaptive_engine.adjust_module_weights(
            org_id=org_id,
            days_back=request.days_back,
            dry_run=request.dry_run
        )

        return {
            "success": True,
            "dry_run": request.dry_run,
            "weight_adjustments": result
        }

    except Exception as e:
        logger.error(f"Optimize weights error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/parameters/{org_id}")
async def optimize_llm_parameters(org_id: str, request: ManualOptimization):
    """
    Optimize LLM prompt parameters based on feedback.
    """
    try:
        adaptive_engine = get_adaptive_engine(supabase)

        result = await adaptive_engine.auto_tune_prompt_parameters(
            org_id=org_id,
            days_back=request.days_back,
            dry_run=request.dry_run
        )

        return {
            "success": True,
            "dry_run": request.dry_run,
            "parameter_adjustments": result
        }

    except Exception as e:
        logger.error(f"Optimize parameters error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization/history/{org_id}")
async def get_optimization_history(
    org_id: str,
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Get optimization history/audit log.
    """
    try:
        adaptive_engine = get_adaptive_engine(supabase)
        history = adaptive_engine.get_optimization_history(org_id, limit)

        return {
            "success": True,
            "count": len(history),
            "history": history
        }

    except Exception as e:
        logger.error(f"Get optimization history error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/recent/{org_id}")
async def get_recent_feedback(
    org_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    include_neutral: bool = Query(default=False)
):
    """
    Get recent feedback entries.
    """
    try:
        feedback_service = get_feedback_service(supabase)
        feedback = feedback_service.get_recent_feedback(org_id, limit, include_neutral)

        return {
            "success": True,
            "count": len(feedback),
            "feedback": feedback
        }

    except Exception as e:
        logger.error(f"Get recent feedback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reasoning/evaluate/{reasoning_log_id}")
async def evaluate_reasoning_log(reasoning_log_id: str, org_id: str = Query(...)):
    """
    Evaluate a specific reasoning log for quality.
    """
    try:
        adaptive_engine = get_adaptive_engine(supabase)
        evaluation = await adaptive_engine.evaluate_reasoning_log(org_id, reasoning_log_id)

        return {
            "success": True,
            "evaluation": evaluation
        }

    except Exception as e:
        logger.error(f"Evaluate reasoning error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
