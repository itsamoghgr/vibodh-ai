"""
Approval API Routes - Phase 4, Human-in-the-Loop
Endpoints for managing AI action approvals and human oversight
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.core.auth import get_current_user, get_org_id
from app.db import supabase, supabase_admin
from app.core.logging import logger

router = APIRouter(prefix="/approvals", tags=["approvals"])


# Request/Response Models
class ApprovalDecisionRequest(BaseModel):
    """Request to approve or reject a pending action"""
    approved: bool
    reason: Optional[str] = None
    notes: Optional[str] = None
    execute_immediately: bool = True


class BulkApprovalRequest(BaseModel):
    """Request to approve/reject multiple actions"""
    action_ids: List[str]
    approved: bool
    reason: Optional[str] = None


class ApprovalStats(BaseModel):
    """Approval statistics for dashboard"""
    pending_count: int
    approved_count: int
    rejected_count: int
    expired_count: int
    avg_approval_time_minutes: Optional[float]


@router.get("/pending", summary="Get pending approvals")
async def get_pending_approvals(
    org_id: str = Query(..., description="Organization ID"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type")
):
    """
    Get all pending action approvals.

    Returns paginated list of actions awaiting approval.
    """
    try:
        # Build query - use admin client to bypass RLS
        query = supabase_admin.table("ai_actions_pending")\
            .select("*, ai_action_plans!inner(agent_type, goal, description)")\
            .eq("org_id", org_id)\
            .eq("status", "pending")\
            .order("created_at", desc=True)\
            .range(offset, offset + limit - 1)

        # Apply filters
        if risk_level:
            query = query.eq("risk_level", risk_level)

        if agent_type:
            query = query.eq("ai_action_plans.agent_type", agent_type)

        result = query.execute()

        return {
            "pending_approvals": result.data or [],
            "count": len(result.data) if result.data else 0,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error fetching pending approvals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plans/{plan_id}", summary="Get approvals for specific plan")
async def get_plan_approvals(
    plan_id: str,
    user=Depends(get_current_user),
    org_id: str = Depends(get_org_id)
):
    """
    Get all pending approvals for a specific action plan.
    """
    try:
        result = supabase.table("ai_actions_pending")\
            .select("*")\
            .eq("org_id", org_id)\
            .eq("plan_id", plan_id)\
            .eq("status", "pending")\
            .order("step_index")\
            .execute()

        return {
            "plan_id": plan_id,
            "pending_actions": result.data or []
        }

    except Exception as e:
        logger.error(f"Error fetching plan approvals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{action_id}/decide", summary="Approve or reject action")
async def decide_approval(
    action_id: str,
    decision: ApprovalDecisionRequest,
    org_id: str = Query(..., description="Organization ID")
):
    """
    Approve or reject a pending action.

    If approved and execute_immediately is True, the action will be executed.
    """
    try:
        # Use placeholder user ID for now (TODO: implement proper auth)
        user_id = "placeholder-user-id"

        # Fetch pending action - use admin client to bypass RLS
        result = supabase_admin.table("ai_actions_pending")\
            .select("*")\
            .eq("id", action_id)\
            .eq("org_id", org_id)\
            .single()\
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Pending action not found")

        pending_action = result.data

        # Check if already processed
        if pending_action["status"] != "pending":
            raise HTTPException(
                status_code=400,
                detail=f"Action already {pending_action['status']}"
            )

        # Update action status
        update_data = {
            "status": "approved" if decision.approved else "rejected",
            "approved_by": None,  # Set to None for now (TODO: use real user_id)
            "approved_at": datetime.utcnow().isoformat()
        }

        if not decision.approved:
            update_data["rejection_reason"] = decision.reason or "Rejected by user"

        supabase_admin.table("ai_actions_pending")\
            .update(update_data)\
            .eq("id", action_id)\
            .execute()

        logger.info(
            f"Action {action_id} {'approved' if decision.approved else 'rejected'}",
            extra={
                "action_id": action_id,
                "approved": decision.approved
            }
        )

        # If approved and should execute immediately, trigger execution
        if decision.approved and decision.execute_immediately:
            # Import here to avoid circular dependency
            from app.services.agent_registry import get_agent_registry

            agent_registry = get_agent_registry(supabase_admin)

            # Queue action for execution
            # This would typically be handled by a background worker
            logger.info(f"Action {action_id} queued for immediate execution")

        return {
            "success": True,
            "action_id": action_id,
            "decision": "approved" if decision.approved else "rejected",
            "will_execute": decision.approved and decision.execute_immediately
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing approval decision: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-decide", summary="Bulk approve or reject actions")
async def bulk_decide(
    request: BulkApprovalRequest,
    user=Depends(get_current_user),
    org_id: str = Depends(get_org_id)
):
    """
    Approve or reject multiple actions at once.

    Useful for batch approval workflows.
    """
    try:
        user_id = user.get("id")

        # Update all actions
        update_data = {
            "status": "approved" if request.approved else "rejected",
            "approved_by": user_id,
            "approved_at": datetime.utcnow().isoformat()
        }

        if not request.approved:
            update_data["rejection_reason"] = request.reason or "Rejected by user"

        result = supabase.table("ai_actions_pending")\
            .update(update_data)\
            .eq("org_id", org_id)\
            .in_("id", request.action_ids)\
            .eq("status", "pending")\
            .execute()

        updated_count = len(result.data) if result.data else 0

        logger.info(
            f"Bulk {'approval' if request.approved else 'rejection'}: "
            f"{updated_count} actions processed",
            extra={
                "approved": request.approved,
                "count": updated_count,
                "user_id": user_id
            }
        )

        return {
            "success": True,
            "processed_count": updated_count,
            "decision": "approved" if request.approved else "rejected"
        }

    except Exception as e:
        logger.error(f"Error processing bulk approval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Get approval statistics")
async def get_approval_stats(
    org_id: str = Query(..., description="Organization ID"),
    days_back: int = Query(30, ge=1, le=365)
):
    """
    Get approval statistics for the organization.

    Returns counts and metrics for pending, approved, and rejected actions.
    """
    try:
        from datetime import timedelta

        cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

        # Get pending count - use admin client to bypass RLS
        pending_result = supabase_admin.table("ai_actions_pending")\
            .select("id", count="exact")\
            .eq("org_id", org_id)\
            .eq("status", "pending")\
            .execute()

        pending_count = pending_result.count or 0

        # Get approved count
        approved_result = supabase_admin.table("ai_actions_pending")\
            .select("id, created_at, approved_at", count="exact")\
            .eq("org_id", org_id)\
            .eq("status", "approved")\
            .gte("created_at", cutoff_date)\
            .execute()

        approved_count = approved_result.count or 0

        # Get rejected count
        rejected_result = supabase_admin.table("ai_actions_pending")\
            .select("id", count="exact")\
            .eq("org_id", org_id)\
            .eq("status", "rejected")\
            .gte("created_at", cutoff_date)\
            .execute()

        rejected_count = rejected_result.count or 0

        # Get expired count
        expired_result = supabase_admin.table("ai_actions_pending")\
            .select("id", count="exact")\
            .eq("org_id", org_id)\
            .eq("status", "expired")\
            .gte("created_at", cutoff_date)\
            .execute()

        expired_count = expired_result.count or 0

        # Calculate average approval time
        avg_approval_time = None
        if approved_result.data:
            approval_times = []
            for action in approved_result.data:
                if action.get("created_at") and action.get("approved_at"):
                    created = datetime.fromisoformat(action["created_at"].replace("Z", "+00:00"))
                    approved = datetime.fromisoformat(action["approved_at"].replace("Z", "+00:00"))
                    approval_times.append((approved - created).total_seconds() / 60)

            if approval_times:
                avg_approval_time = sum(approval_times) / len(approval_times)

        return {
            "pending_count": pending_count,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "expired_count": expired_count,
            "avg_approval_time_minutes": avg_approval_time,
            "period_days": days_back,
            "approval_rate": approved_count / (approved_count + rejected_count) if (approved_count + rejected_count) > 0 else None
        }

    except Exception as e:
        logger.error(f"Error fetching approval stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", summary="Get approval history")
async def get_approval_history(
    user=Depends(get_current_user),
    org_id: str = Depends(get_org_id),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """
    Get approval history (approved/rejected actions).
    """
    try:
        query = supabase.table("ai_actions_pending")\
            .select("*, ai_action_plans!inner(agent_type, goal)")\
            .eq("org_id", org_id)\
            .neq("status", "pending")\
            .order("approved_at", desc=True)\
            .range(offset, offset + limit - 1)

        if status:
            query = query.eq("status", status)

        result = query.execute()

        return {
            "history": result.data or [],
            "count": len(result.data) if result.data else 0,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error fetching approval history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{action_id}/expire", summary="Manually expire pending action")
async def expire_action(
    action_id: str,
    user=Depends(get_current_user),
    org_id: str = Depends(get_org_id)
):
    """
    Manually expire a pending action (e.g., if no longer relevant).
    """
    try:
        result = supabase.table("ai_actions_pending")\
            .update({
                "status": "expired",
                "updated_at": datetime.utcnow().isoformat()
            })\
            .eq("id", action_id)\
            .eq("org_id", org_id)\
            .eq("status", "pending")\
            .execute()

        if not result.data:
            raise HTTPException(
                status_code=404,
                detail="Pending action not found or already processed"
            )

        logger.info(f"Action {action_id} manually expired")

        return {
            "success": True,
            "action_id": action_id,
            "status": "expired"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error expiring action: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
