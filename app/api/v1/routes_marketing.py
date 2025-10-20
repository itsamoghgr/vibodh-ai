"""
Marketing API Routes - Phase 4, Step 2
Endpoints for marketing agent operations
"""

from fastapi import APIRouter, Query, HTTPException, Body
from typing import Optional, List
from uuid import UUID

from app.services.marketing_service import get_marketing_service
from app.services.agent_registry import get_agent_registry
from app.services.action_planning_service import get_action_planning_service
from app.db import supabase
from app.models.marketing import (
    CampaignPlan, CampaignStatus, CampaignType,
    MarketingContent, ContentGenerationRequest, ContentGenerationResponse,
    MarketingMetrics, PerformanceAnalysis,
    MarketingTask, MarketingAnnouncement,
    MarketingContext
)
from app.models.agent import AgentExecuteRequest, AgentExecuteResponse
from app.agents.base_agent import ObservationContext
from app.core.logging import logger, log_error

router = APIRouter(prefix="/marketing", tags=["Marketing"])


# Campaign Management Endpoints

@router.post("/campaigns", response_model=CampaignPlan)
async def create_campaign(
    campaign: CampaignPlan,
    org_id: UUID = Query(..., description="Organization ID"),
    user_id: Optional[UUID] = Query(None, description="User ID"),
    auto_execute: bool = Query(False, description="Auto-execute campaign")
):
    """
    Create a new marketing campaign.

    If auto_execute is True, the campaign will be sent to the Marketing Agent
    for execution planning.

    Args:
        campaign: Campaign plan details
        org_id: Organization ID
        user_id: User ID
        auto_execute: Whether to automatically execute

    Returns:
        Created campaign plan
    """
    try:
        logger.info(f"Creating campaign: {campaign.title}")

        marketing_service = get_marketing_service(supabase)

        # Store campaign
        # TODO: Add database storage
        campaign.id = UUID("00000000-0000-0000-0000-000000000001")  # Placeholder
        campaign.created_at = campaign.created_at or campaign.created_at
        campaign.status = CampaignStatus.PLANNING

        if auto_execute:
            # Send to Marketing Agent for execution
            agent_registry = get_agent_registry(supabase)

            try:
                agent = agent_registry.get_agent(str(org_id), "marketing_agent")

                # Create context
                context = ObservationContext(
                    query=f"Execute campaign: {campaign.title}",
                    trigger_type="manual",
                    org_id=str(org_id),
                    user_id=str(user_id) if user_id else None,
                    metadata={
                        "campaign": campaign.dict(),
                        "auto_execute": True
                    }
                )

                # Check if action needed
                should_act, reason = await agent.observe(context)

                if should_act:
                    # Generate plan
                    plan = await agent.plan(
                        f"Execute marketing campaign: {campaign.title}",
                        {"campaign": campaign.dict()}
                    )

                    # Create action plan in database
                    action_planning = get_action_planning_service(supabase)
                    from app.models.agent import TriggerType

                    plan_response = await action_planning.create_action_plan(
                        org_id=str(org_id),
                        plan=plan,
                        agent_type="marketing_agent",
                        trigger_type=TriggerType.MANUAL,
                        trigger_source={"campaign_id": str(campaign.id)},
                        user_id=str(user_id) if user_id else None
                    )

                    campaign.status = CampaignStatus.APPROVED
                    logger.info(f"Campaign {campaign.id} sent to agent for execution")

            except Exception as e:
                logger.error(f"Failed to execute campaign with agent: {e}")
                # Continue without agent execution

        return campaign

    except Exception as e:
        log_error(e, context="Create campaign")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/campaigns", response_model=List[CampaignPlan])
async def list_campaigns(
    org_id: UUID = Query(..., description="Organization ID"),
    status: Optional[CampaignStatus] = Query(None, description="Filter by status"),
    campaign_type: Optional[CampaignType] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum campaigns to return")
):
    """
    List marketing campaigns.

    Args:
        org_id: Organization ID
        status: Optional status filter
        campaign_type: Optional type filter
        limit: Maximum number of campaigns

    Returns:
        List of campaigns
    """
    try:
        logger.info(f"Listing campaigns for org {org_id}")

        # TODO: Implement actual database query
        campaigns = []

        # Mock data
        mock_campaign = CampaignPlan(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            title="Q1 Product Launch",
            description="Launch campaign for new product",
            campaign_type=CampaignType.PRODUCT_LAUNCH,
            goals=["Increase awareness", "Generate leads"],
            target_audience="Tech professionals",
            channels=["slack", "email"],
            timeline_days=30,
            key_messages=["Innovation", "Efficiency", "Value"],
            success_metrics=["Engagement rate > 5%", "100+ leads"],
            estimated_reach="1000-5000",
            status=CampaignStatus.PLANNING
        )
        campaigns.append(mock_campaign)

        return campaigns

    except Exception as e:
        log_error(e, context="List campaigns")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/campaigns/{campaign_id}", response_model=CampaignPlan)
async def get_campaign(
    campaign_id: UUID,
    org_id: UUID = Query(..., description="Organization ID")
):
    """
    Get campaign details.

    Args:
        campaign_id: Campaign ID
        org_id: Organization ID

    Returns:
        Campaign details
    """
    try:
        logger.info(f"Getting campaign {campaign_id}")

        # TODO: Implement actual database query
        campaign = CampaignPlan(
            id=campaign_id,
            title="Q1 Product Launch",
            description="Launch campaign for new product",
            campaign_type=CampaignType.PRODUCT_LAUNCH,
            goals=["Increase awareness", "Generate leads"],
            target_audience="Tech professionals",
            channels=["slack", "email"],
            timeline_days=30,
            key_messages=["Innovation", "Efficiency", "Value"],
            success_metrics=["Engagement rate > 5%", "100+ leads"],
            estimated_reach="1000-5000",
            status=CampaignStatus.PLANNING
        )

        return campaign

    except Exception as e:
        log_error(e, context="Get campaign")
        raise HTTPException(status_code=500, detail=str(e))


# Content Generation Endpoints

@router.post("/content/generate", response_model=ContentGenerationResponse)
async def generate_content(
    request: ContentGenerationRequest,
    org_id: UUID = Query(..., description="Organization ID")
):
    """
    Generate marketing content using AI.

    Args:
        request: Content generation request
        org_id: Organization ID

    Returns:
        Generated content response
    """
    try:
        logger.info(f"Generating {request.content_type} content")

        marketing_service = get_marketing_service(supabase)
        response = await marketing_service.generate_campaign_content(request, str(org_id))

        return response

    except Exception as e:
        log_error(e, context="Generate content")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/content", response_model=List[MarketingContent])
async def list_content(
    org_id: UUID = Query(..., description="Organization ID"),
    campaign_id: Optional[UUID] = Query(None, description="Filter by campaign"),
    content_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum content items")
):
    """
    List marketing content.

    Args:
        org_id: Organization ID
        campaign_id: Optional campaign filter
        content_type: Optional type filter
        limit: Maximum items

    Returns:
        List of content items
    """
    try:
        logger.info(f"Listing content for org {org_id}")

        # TODO: Implement actual database query
        content = []

        return content

    except Exception as e:
        log_error(e, context="List content")
        raise HTTPException(status_code=500, detail=str(e))


# Performance Tracking Endpoints

@router.get("/metrics/{campaign_id}", response_model=MarketingMetrics)
async def get_campaign_metrics(
    campaign_id: UUID,
    org_id: UUID = Query(..., description="Organization ID")
):
    """
    Get campaign performance metrics.

    Args:
        campaign_id: Campaign ID
        org_id: Organization ID

    Returns:
        Campaign metrics
    """
    try:
        logger.info(f"Getting metrics for campaign {campaign_id}")

        marketing_service = get_marketing_service(supabase)
        metrics = await marketing_service.track_campaign_performance(campaign_id, str(org_id))

        return metrics

    except Exception as e:
        log_error(e, context="Get metrics")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{campaign_id}", response_model=PerformanceAnalysis)
async def analyze_campaign(
    campaign_id: UUID,
    org_id: UUID = Query(..., description="Organization ID")
):
    """
    Analyze campaign performance.

    Args:
        campaign_id: Campaign ID
        org_id: Organization ID

    Returns:
        Performance analysis
    """
    try:
        logger.info(f"Analyzing campaign {campaign_id}")

        marketing_service = get_marketing_service(supabase)

        # Get metrics
        metrics = await marketing_service.track_campaign_performance(campaign_id, str(org_id))

        # Analyze
        analysis = await marketing_service.analyze_campaign_performance(
            campaign_id, metrics, str(org_id)
        )

        return analysis

    except Exception as e:
        log_error(e, context="Analyze campaign")
        raise HTTPException(status_code=500, detail=str(e))


# Task Management Endpoints

@router.get("/tasks", response_model=List[MarketingTask])
async def list_marketing_tasks(
    org_id: UUID = Query(..., description="Organization ID"),
    campaign_id: Optional[UUID] = Query(None, description="Filter by campaign"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum tasks")
):
    """
    List marketing tasks.

    Args:
        org_id: Organization ID
        campaign_id: Optional campaign filter
        status: Optional status filter
        limit: Maximum tasks

    Returns:
        List of marketing tasks
    """
    try:
        logger.info(f"Listing tasks for org {org_id}")

        # TODO: Implement actual database query
        tasks = []

        return tasks

    except Exception as e:
        log_error(e, context="List tasks")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks")
async def create_marketing_task(
    task: MarketingTask,
    org_id: UUID = Query(..., description="Organization ID")
):
    """
    Create a marketing task.

    Args:
        task: Task details
        org_id: Organization ID

    Returns:
        Created task
    """
    try:
        logger.info(f"Creating task: {task.title}")

        # TODO: Implement task creation
        task.id = UUID("00000000-0000-0000-0000-000000000001")  # Placeholder

        return task

    except Exception as e:
        log_error(e, context="Create task")
        raise HTTPException(status_code=500, detail=str(e))


# Announcement Endpoints

@router.post("/announcements")
async def post_announcement(
    announcement: MarketingAnnouncement,
    org_id: UUID = Query(..., description="Organization ID")
):
    """
    Post a marketing announcement.

    Args:
        announcement: Announcement details
        org_id: Organization ID

    Returns:
        Posting result
    """
    try:
        logger.info(f"Posting announcement to {announcement.channel}")

        marketing_service = get_marketing_service(supabase)
        result = await marketing_service.post_announcement(announcement, str(org_id))

        return result

    except Exception as e:
        log_error(e, context="Post announcement")
        raise HTTPException(status_code=500, detail=str(e))


# Context Analysis Endpoints

@router.get("/context", response_model=MarketingContext)
async def get_marketing_context(
    org_id: UUID = Query(..., description="Organization ID"),
    lookback_days: int = Query(30, ge=1, le=90, description="Days to analyze")
):
    """
    Get marketing context analysis.

    Args:
        org_id: Organization ID
        lookback_days: Days to look back

    Returns:
        Marketing context
    """
    try:
        logger.info(f"Getting marketing context for org {org_id}")

        marketing_service = get_marketing_service(supabase)
        context = await marketing_service.analyze_marketing_context(str(org_id), lookback_days)

        return context

    except Exception as e:
        log_error(e, context="Get context")
        raise HTTPException(status_code=500, detail=str(e))


# Agent Execution Endpoint

@router.post("/execute", response_model=AgentExecuteResponse)
async def execute_marketing_action(
    request: AgentExecuteRequest,
    org_id: UUID = Query(..., description="Organization ID"),
    user_id: Optional[UUID] = Query(None, description="User ID")
):
    """
    Execute a marketing action via the Marketing Agent.

    Args:
        request: Execution request
        org_id: Organization ID
        user_id: User ID

    Returns:
        Execution response
    """
    try:
        logger.info(f"Executing marketing action: {request.goal}")

        # Override agent type to ensure marketing agent
        request.agent_type = "marketing_agent"

        agent_registry = get_agent_registry(supabase)
        action_planning = get_action_planning_service(supabase)

        # Get agent
        try:
            agent = agent_registry.get_agent(str(org_id), "marketing_agent")
        except ValueError as e:
            raise HTTPException(status_code=404, detail="Marketing Agent not found")

        # Create observation context
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
                plan_id=UUID("00000000-0000-0000-0000-000000000000"),
                agent_type="marketing_agent",
                goal=request.goal,
                status="cancelled",
                plan_created=False,
                requires_approval=False,
                risk_level="low",
                total_steps=0,
                message=reason or "No marketing action required"
            )

        # Generate plan
        plan = await agent.plan(request.goal, request.context)

        # Create action plan
        from app.models.agent import TriggerType, PlanStatus

        plan_response = await action_planning.create_action_plan(
            org_id=str(org_id),
            plan=plan,
            agent_type="marketing_agent",
            trigger_type=TriggerType.MANUAL,
            trigger_source={"manual": True},
            user_id=str(user_id) if user_id else None
        )

        return AgentExecuteResponse(
            plan_id=plan_response.id,
            agent_type="marketing_agent",
            goal=plan.goal,
            status=PlanStatus(plan_response.status),
            plan_created=True,
            requires_approval=plan.requires_approval,
            risk_level=plan.risk_level,
            total_steps=plan.total_steps,
            message=f"Marketing plan created with {plan.total_steps} steps",
            next_action="Review plan in dashboard" if plan.requires_approval else "Monitor execution"
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context="Execute marketing action")
        raise HTTPException(status_code=500, detail=str(e))