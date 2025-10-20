"""
Agent API Routes - Phase 4, Step 1
Endpoints for agent management and execution
"""

from fastapi import APIRouter, Query, HTTPException, Body
from typing import Optional, List
from uuid import UUID

from app.services.agent_registry import get_agent_registry
from app.services.action_planning_service import get_action_planning_service
from app.services.safety_service import get_safety_service
from app.db import supabase
from app.models.agent import (
    AgentExecuteRequest, AgentExecuteResponse,
    AgentConfig, AgentRegistryEntry,
    ActionPlanRequest, ActionPlanResponse,
    PendingActionResponse, ActionApprovalRequest,
    ExecutedActionResponse, AgentStatusResponse,
    AgentStatistics, AgentEvent, AgentEventResponse,
    PlanStatus, ActionStatus
)
from app.core.logging import logger, log_error

router = APIRouter(prefix="/agents", tags=["Agents"])


# Agent Registry Endpoints

@router.post("/register")
async def register_agent(
    org_id: UUID,
    config: AgentConfig
):
    """
    Register a new agent or update existing agent configuration.

    Args:
        org_id: Organization ID
        config: Agent configuration

    Returns:
        Agent registry entry
    """
    try:
        logger.info(f"Registering agent {config.agent_type} for org {org_id}")

        agent_registry = get_agent_registry(supabase)
        entry = await agent_registry.register_agent(str(org_id), config)

        return entry

    except Exception as e:
        log_error(e, context="Register agent")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry", response_model=List[AgentRegistryEntry])
async def list_agents(
    org_id: UUID = Query(..., description="Organization ID"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """
    List all registered agents for an organization.

    Args:
        org_id: Organization ID
        status: Optional status filter (active, inactive, testing)

    Returns:
        List of agent registry entries
    """
    try:
        logger.info(f"Listing agents for org {org_id}")

        agent_registry = get_agent_registry(supabase)
        agents = await agent_registry.list_available_agents(
            str(org_id),
            status=status
        )

        return agents

    except Exception as e:
        log_error(e, context="List agents")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{agent_type}", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_type: str,
    org_id: UUID = Query(..., description="Organization ID")
):
    """
    Get current status of a specific agent.

    Args:
        agent_type: Agent type
        org_id: Organization ID

    Returns:
        Agent status
    """
    try:
        logger.info(f"Getting status for agent {agent_type} in org {org_id}")

        agent_registry = get_agent_registry(supabase)
        status = await agent_registry.get_agent_status(str(org_id), agent_type)

        return status

    except Exception as e:
        log_error(e, context="Get agent status")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/{agent_type}", response_model=AgentStatistics)
async def get_agent_statistics(
    agent_type: str,
    org_id: UUID = Query(..., description="Organization ID")
):
    """
    Get performance statistics for an agent.

    Args:
        agent_type: Agent type
        org_id: Organization ID

    Returns:
        Agent statistics
    """
    try:
        logger.info(f"Getting statistics for agent {agent_type} in org {org_id}")

        agent_registry = get_agent_registry(supabase)
        stats = await agent_registry.get_agent_statistics(str(org_id), agent_type)

        return stats

    except Exception as e:
        log_error(e, context="Get agent statistics")
        raise HTTPException(status_code=500, detail=str(e))


# Agent Execution Endpoints

@router.post("/execute", response_model=AgentExecuteResponse)
async def execute_agent(
    request: AgentExecuteRequest,
    org_id: UUID = Query(..., description="Organization ID"),
    user_id: Optional[UUID] = Query(None, description="User ID")
):
    """
    Execute an agent action.

    This endpoint will:
    1. Route the goal to the appropriate agent
    2. Generate an action plan
    3. Validate the plan with safety checks
    4. Create the plan in the database
    5. Queue actions for execution (if approved)

    Args:
        request: Execution request with goal and context
        org_id: Organization ID
        user_id: Optional user ID

    Returns:
        Execution response with plan details
    """
    try:
        logger.info(f"Executing agent for goal: {request.goal}")

        agent_registry = get_agent_registry(supabase)
        action_planning = get_action_planning_service(supabase)
        safety_service = get_safety_service(supabase)

        # Get agent
        try:
            agent = agent_registry.get_agent(str(org_id), request.agent_type)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=f"Agent '{request.agent_type}' not found")

        # Create observation context
        from app.agents.base_agent import ObservationContext
        obs_context = ObservationContext(
            query=request.goal,
            trigger_type="manual",
            org_id=str(org_id),
            user_id=str(user_id) if user_id else None,
            metadata=request.context
        )

        # Check if action needed
        should_act, reason = await agent.observe(obs_context)

        if not should_act:
            return AgentExecuteResponse(
                plan_id=UUID("00000000-0000-0000-0000-000000000000"),  # Placeholder
                agent_type=request.agent_type,
                goal=request.goal,
                status=PlanStatus.CANCELLED,
                plan_created=False,
                requires_approval=False,
                risk_level="low",
                total_steps=0,
                message=reason or "No action required"
            )

        # Generate plan
        plan = await agent.plan(request.goal, request.context)

        # Validate with safety
        is_valid, risk_level, issues = await safety_service.validate_plan(plan, str(org_id))

        if not is_valid and not request.dry_run:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Plan validation failed",
                    "issues": issues
                }
            )

        # Handle dry run
        if request.dry_run:
            return AgentExecuteResponse(
                plan_id=UUID("00000000-0000-0000-0000-000000000000"),  # Placeholder
                agent_type=request.agent_type,
                goal=plan.goal,
                status=PlanStatus.DRAFT,
                plan_created=False,
                requires_approval=plan.requires_approval,
                risk_level=risk_level.value,
                total_steps=plan.total_steps,
                message="Dry run completed - no plan created",
                next_action="Review the plan details"
            )

        # Create plan in database
        from app.models.agent import TriggerType
        plan_response = await action_planning.create_action_plan(
            org_id=str(org_id),
            plan=plan,
            agent_type=request.agent_type,
            trigger_type=TriggerType.MANUAL,
            trigger_source={"manual": True},
            user_id=str(user_id) if user_id else None
        )

        # Determine next action
        if plan.requires_approval and not request.auto_approve:
            next_action = "Review and approve the plan"
            message = f"Plan requires approval (risk: {risk_level.value})"
        elif request.auto_approve and risk_level.value == "low":
            # Auto-approve low risk
            await action_planning.queue_actions(
                plan_response.id,
                plan.steps,
                str(org_id)
            )
            next_action = "Monitor execution progress"
            message = "Plan auto-approved and queued for execution"
        else:
            next_action = "Monitor execution progress"
            message = "Plan created and queued"

        return AgentExecuteResponse(
            plan_id=plan_response.id,
            agent_type=request.agent_type,
            goal=plan.goal,
            status=plan_response.status,
            plan_created=True,
            requires_approval=plan.requires_approval,
            risk_level=risk_level.value,
            total_steps=plan.total_steps,
            message=message,
            next_action=next_action
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context="Execute agent")
        raise HTTPException(status_code=500, detail=str(e))


# Action Plan Endpoints

@router.get("/plans", response_model=List[ActionPlanResponse])
async def list_action_plans(
    org_id: UUID = Query(..., description="Organization ID"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum plans to return")
):
    """
    List action plans for an organization.

    Args:
        org_id: Organization ID
        agent_type: Optional agent type filter
        status: Optional status filter
        limit: Maximum number of plans

    Returns:
        List of action plans
    """
    try:
        logger.info(f"Listing plans for org {org_id}")

        action_planning = get_action_planning_service(supabase)
        plans = await action_planning.list_plans(
            str(org_id),
            agent_type=agent_type,
            status=PlanStatus(status) if status else None,
            limit=limit
        )

        return plans

    except Exception as e:
        log_error(e, context="List plans")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plans/{plan_id}", response_model=ActionPlanResponse)
async def get_action_plan(
    plan_id: UUID,
    org_id: UUID = Query(..., description="Organization ID")
):
    """
    Get a specific action plan.

    Args:
        plan_id: Plan ID
        org_id: Organization ID

    Returns:
        Action plan details
    """
    try:
        logger.info(f"Getting plan {plan_id}")

        action_planning = get_action_planning_service(supabase)
        plan = await action_planning.get_plan(str(plan_id), str(org_id))

        return plan

    except Exception as e:
        log_error(e, context="Get plan")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/plans/{plan_id}")
async def cancel_action_plan(
    plan_id: UUID,
    org_id: UUID = Query(..., description="Organization ID")
):
    """
    Cancel an action plan and its pending actions.

    Args:
        plan_id: Plan ID
        org_id: Organization ID

    Returns:
        Cancelled plan
    """
    try:
        logger.info(f"Cancelling plan {plan_id}")

        action_planning = get_action_planning_service(supabase)
        plan = await action_planning.cancel_plan(str(plan_id), str(org_id))

        return plan

    except Exception as e:
        log_error(e, context="Cancel plan")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plans/{plan_id}/approve")
async def approve_plan(
    plan_id: UUID,
    org_id: UUID = Query(..., description="Organization ID"),
    user_id: UUID = Query(..., description="Approver user ID"),
    approval_data: dict = Body(default={"approved": True})
):
    """
    Approve a pending action plan and optionally execute it.

    Args:
        plan_id: Plan ID
        org_id: Organization ID
        user_id: User ID of approver
        approval_data: Approval details

    Returns:
        Updated plan
    """
    try:
        from datetime import datetime

        approved = approval_data.get("approved", True)

        if not approved:
            # This is actually a rejection
            reason = approval_data.get("reason", "Rejected by user")
            logger.info(f"Rejecting plan {plan_id}: {reason}")

            result = supabase.table("ai_action_plans")\
                .update({
                    "status": "cancelled",
                    "approval_status": "rejected",
                    "approved_by": str(user_id),
                    "approved_at": datetime.utcnow().isoformat(),
                    "metadata": {"rejection_reason": reason}
                })\
                .eq("id", str(plan_id))\
                .eq("org_id", str(org_id))\
                .execute()

            return {"success": True, "plan": result.data[0] if result.data else None}

        logger.info(f"Approving plan {plan_id}")

        # Update plan status to approved
        result = supabase.table("ai_action_plans")\
            .update({
                "status": "approved",
                "approval_status": "approved",
                "approved_by": str(user_id),
                "approved_at": datetime.utcnow().isoformat()
            })\
            .eq("id", str(plan_id))\
            .eq("org_id", str(org_id))\
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Plan not found")

        plan_data = result.data[0]

        # Execute the plan if auto-execute is enabled
        if approval_data.get("execute", True):
            from app.services.orchestrator_service import get_orchestrator_service
            orchestrator = get_orchestrator_service(supabase)

            # Execute the plan using orchestrator's execute method
            exec_result = await orchestrator._approve_and_execute_plan(
                plan_data=plan_data,
                org_id=str(org_id),
                user_id=str(user_id)
            )

            return {
                "success": True,
                "plan": plan_data,
                "execution": exec_result
            }

        return {"success": True, "plan": plan_data}

    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context="Approve plan")
        raise HTTPException(status_code=500, detail=str(e))


# Pending Actions Endpoints

@router.get("/pending", response_model=List[PendingActionResponse])
async def list_pending_actions(
    org_id: UUID = Query(..., description="Organization ID"),
    plan_id: Optional[UUID] = Query(None, description="Filter by plan ID"),
    requires_approval: Optional[bool] = Query(None, description="Filter by approval requirement")
):
    """
    List pending actions for an organization.

    Args:
        org_id: Organization ID
        plan_id: Optional plan ID filter
        requires_approval: Optional approval filter

    Returns:
        List of pending actions
    """
    try:
        logger.info(f"Listing pending actions for org {org_id}")

        action_planning = get_action_planning_service(supabase)
        actions = await action_planning.get_pending_actions(
            str(org_id),
            plan_id=str(plan_id) if plan_id else None,
            requires_approval=requires_approval
        )

        return actions

    except Exception as e:
        log_error(e, context="List pending actions")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/actions/{action_id}/approve")
async def approve_action(
    action_id: UUID,
    org_id: UUID = Query(..., description="Organization ID"),
    user_id: UUID = Query(..., description="Approver user ID")
):
    """
    Approve a pending action.

    Args:
        action_id: Action ID
        org_id: Organization ID
        user_id: User ID of approver

    Returns:
        Updated action
    """
    try:
        logger.info(f"Approving action {action_id}")

        action_planning = get_action_planning_service(supabase)
        action = await action_planning.approve_action(
            str(action_id),
            str(user_id),
            str(org_id)
        )

        return action

    except Exception as e:
        log_error(e, context="Approve action")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/actions/{action_id}/reject")
async def reject_action(
    action_id: UUID,
    request: ActionApprovalRequest,
    org_id: UUID = Query(..., description="Organization ID"),
    user_id: UUID = Query(..., description="Rejector user ID")
):
    """
    Reject a pending action.

    Args:
        action_id: Action ID
        request: Rejection details
        org_id: Organization ID
        user_id: User ID of rejector

    Returns:
        Updated action
    """
    try:
        logger.info(f"Rejecting action {action_id}: {request.reason}")

        action_planning = get_action_planning_service(supabase)
        action = await action_planning.reject_action(
            str(action_id),
            str(user_id),
            request.reason or "No reason provided",
            str(org_id)
        )

        return action

    except Exception as e:
        log_error(e, context="Reject action")
        raise HTTPException(status_code=500, detail=str(e))


# Executed Actions Endpoints

@router.get("/executed", response_model=List[ExecutedActionResponse])
async def list_executed_actions(
    org_id: UUID = Query(..., description="Organization ID"),
    plan_id: Optional[UUID] = Query(None, description="Filter by plan ID"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum actions to return")
):
    """
    List executed actions for an organization.

    Args:
        org_id: Organization ID
        plan_id: Optional plan ID filter
        agent_type: Optional agent type filter
        status: Optional status filter
        limit: Maximum number of actions

    Returns:
        List of executed actions
    """
    try:
        logger.info(f"Listing executed actions for org {org_id}")

        query = supabase.table("ai_actions_executed")\
            .select("*")\
            .eq("org_id", str(org_id))

        if plan_id:
            query = query.eq("plan_id", str(plan_id))
        if agent_type:
            query = query.eq("agent_type", agent_type)
        if status:
            query = query.eq("status", status)

        result = query.order("started_at", desc=True).limit(limit).execute()

        return [ExecutedActionResponse(**action) for action in result.data]

    except Exception as e:
        log_error(e, context="List executed actions")
        raise HTTPException(status_code=500, detail=str(e))


# Agent Event Endpoints

@router.get("/events", response_model=List[AgentEventResponse])
async def list_agent_events(
    org_id: UUID = Query(..., description="Organization ID"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    plan_id: Optional[UUID] = Query(None, description="Filter by plan ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum events to return")
):
    """
    List agent events for an organization.

    Args:
        org_id: Organization ID
        agent_type: Optional agent type filter
        event_type: Optional event type filter
        plan_id: Optional plan ID filter
        limit: Maximum number of events

    Returns:
        List of agent events
    """
    try:
        logger.info(f"Listing agent events for org {org_id}")

        query = supabase.table("ai_agent_events")\
            .select("*")\
            .eq("org_id", str(org_id))

        if agent_type:
            query = query.eq("agent_type", agent_type)
        if event_type:
            query = query.eq("event_type", event_type)
        if plan_id:
            query = query.eq("plan_id", str(plan_id))

        result = query.order("created_at", desc=True).limit(limit).execute()

        return [AgentEventResponse(**event) for event in result.data]

    except Exception as e:
        log_error(e, context="List agent events")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events")
async def record_agent_event(
    event: AgentEvent,
    org_id: UUID = Query(..., description="Organization ID"),
    user_id: Optional[UUID] = Query(None, description="User ID")
):
    """
    Record an agent event.

    Args:
        event: Event details
        org_id: Organization ID
        user_id: Optional user ID

    Returns:
        Created event
    """
    try:
        logger.info(f"Recording event for agent {event.agent_type}: {event.event_type}")

        event_data = {
            "org_id": str(org_id),
            "agent_type": event.agent_type,
            "event_type": event.event_type.value,
            "event_source": event.event_source,
            "plan_id": str(event.plan_id) if event.plan_id else None,
            "action_id": str(event.action_id) if event.action_id else None,
            "payload": event.payload,
            "metadata": event.metadata,
            "user_id": str(user_id) if user_id else None
        }

        result = supabase.table("ai_agent_events").insert(event_data).execute()

        if result.data:
            return AgentEventResponse(**result.data[0])
        else:
            raise Exception("Failed to record event")

    except Exception as e:
        log_error(e, context="Record agent event")
        raise HTTPException(status_code=500, detail=str(e))