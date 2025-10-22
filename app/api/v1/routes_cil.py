"""
CIL API Routes - Phase 5, Cognitive Intelligence Layer
Endpoints for policy management, telemetry, and meta-learning
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime

from app.core.logging import logger
from app.services.cil_policy_service import get_cil_policy_service
from app.services.cil_telemetry_service import get_cil_telemetry_service
from app.services.cil_prompt_optimizer import get_cil_prompt_optimizer
from app.workers import get_cil_worker

router = APIRouter(prefix="/cil", tags=["cil"])


# Request/Response Models
class PolicyConfigRequest(BaseModel):
    """Request to manually update a policy"""
    policy_config: Dict[str, Any]
    change_reason: Optional[str] = None
    approval_required: bool = False


class PolicyResponse(BaseModel):
    """Policy information response"""
    id: str
    org_id: str
    version: int
    is_active: bool
    policy_config: Dict[str, Any]
    created_at: str
    activated_at: Optional[str] = None


class ProposalApprovalRequest(BaseModel):
    """Request to approve or reject a policy proposal"""
    approved: bool
    review_notes: Optional[str] = None
    reviewed_by: Optional[str] = None


class TelemetryStatsResponse(BaseModel):
    """Telemetry statistics response"""
    total_records: int
    by_source: Dict[str, int]
    success_rate: float
    period_days: int


# ============================================================================
# Policy Endpoints
# ============================================================================

@router.get("/policy/{org_id}", summary="Get active policy")
async def get_active_policy(
    org_id: str
):
    """
    Get the currently active policy for an organization.

    If no policy exists, returns the default policy.
    """
    try:
        policy_service = get_cil_policy_service()
        policy = policy_service.get_active_policy(org_id)

        if not policy:
            raise HTTPException(status_code=404, detail="No policy found for organization")

        return {
            "success": True,
            "policy": policy
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching active policy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/policy/{org_id}", summary="Create new policy")
async def create_policy(
    org_id: str,
    request: PolicyConfigRequest
):
    """
    Manually create a new policy version.

    If approval_required=False, the policy is activated immediately.
    Otherwise, it becomes a pending proposal.
    """
    try:
        policy_service = get_cil_policy_service()

        policy = policy_service.create_policy(
            org_id=org_id,
            policy_config=request.policy_config,
            change_reason=request.change_reason or "Manual policy creation",
            approval_required=request.approval_required,
            created_by="api_user"
        )

        if not policy:
            raise HTTPException(status_code=500, detail="Failed to create policy")

        return {
            "success": True,
            "policy": policy,
            "message": "Policy created and activated" if not request.approval_required else "Policy created, pending approval"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating policy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/policy/{org_id}/version/{version}", summary="Get specific policy version")
async def get_policy_version(
    org_id: str,
    version: int
):
    """Get a specific policy version by version number."""
    try:
        policy_service = get_cil_policy_service()
        policy = policy_service.get_policy_by_version(org_id, version)

        if not policy:
            raise HTTPException(status_code=404, detail=f"Policy version {version} not found")

        return {
            "success": True,
            "policy": policy
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching policy version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/policy/{org_id}/history", summary="Get policy version history")
async def get_policy_history(
    org_id: str,
    limit: int = Query(10, ge=1, le=50)
):
    """Get policy version history for an organization."""
    try:
        policy_service = get_cil_policy_service()
        policies = policy_service.get_policy_history(org_id, limit)

        return {
            "success": True,
            "policies": policies,
            "count": len(policies)
        }

    except Exception as e:
        logger.error(f"Error fetching policy history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/policy/{org_id}/compare", summary="Compare two policy versions")
async def compare_policies(
    org_id: str,
    version_a: int = Query(..., description="First version to compare"),
    version_b: int = Query(..., description="Second version to compare")
):
    """Compare two policy versions and show differences."""
    try:
        policy_service = get_cil_policy_service()
        comparison = policy_service.compare_policies(org_id, version_a, version_b)

        if 'error' in comparison:
            raise HTTPException(status_code=400, detail=comparison['error'])

        return {
            "success": True,
            "comparison": comparison
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing policies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/policy/{policy_id}/activate", summary="Activate a policy")
async def activate_policy(
    policy_id: str,
    activated_by: Optional[str] = Query(None, description="User ID who is activating")
):
    """
    Activate a policy (typically after approval).

    Deactivates the current active policy and activates this one.
    """
    try:
        policy_service = get_cil_policy_service()
        success = policy_service.activate_policy(policy_id, activated_by)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to activate policy")

        return {
            "success": True,
            "message": "Policy activated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating policy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Telemetry Endpoints
# ============================================================================

@router.post("/telemetry/ingest/{org_id}", summary="Trigger telemetry ingestion")
async def ingest_telemetry(
    org_id: str,
    hours_back: int = Query(1, ge=1, le=24, description="Hours of history to ingest")
):
    """
    Manually trigger telemetry ingestion from all sources.

    Typically runs automatically, but can be triggered manually.
    """
    try:
        telemetry_service = get_cil_telemetry_service()
        results = await telemetry_service.ingest_all_sources(org_id, hours_back)

        return {
            "success": True,
            "ingestion_results": results,
            "message": f"Ingested {sum(v for k, v in results.items() if k != 'errors')} total records"
        }

    except Exception as e:
        logger.error(f"Error ingesting telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telemetry/stats/{org_id}", summary="Get telemetry statistics")
async def get_telemetry_stats(
    org_id: str,
    days_back: int = Query(7, ge=1, le=90, description="Days of history to analyze")
):
    """Get telemetry statistics for an organization."""
    try:
        telemetry_service = get_cil_telemetry_service()
        stats = telemetry_service.get_telemetry_stats(org_id, days_back)

        return {
            "success": True,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error fetching telemetry stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Proposal Endpoints (for human-in-the-loop approval)
# ============================================================================

@router.get("/proposals/{org_id}", summary="Get pending policy proposals")
async def get_pending_proposals(
    org_id: str,
    status: str = Query("pending", description="Filter by status"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get policy proposals awaiting approval.

    CIL generates these when it discovers potential improvements.
    """
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        query = supabase.table('cil_policy_proposals')\
            .select('*')\
            .eq('org_id', org_id)\
            .eq('status', status)\
            .order('created_at', desc=True)\
            .limit(limit)

        result = query.execute()

        return {
            "success": True,
            "proposals": result.data or [],
            "count": len(result.data) if result.data else 0
        }

    except Exception as e:
        logger.error(f"Error fetching proposals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/proposals/{org_id}/{proposal_id}", summary="Get specific proposal")
async def get_proposal(
    org_id: str,
    proposal_id: str
):
    """Get detailed information about a specific proposal."""
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        result = supabase.table('cil_policy_proposals')\
            .select('*')\
            .eq('id', proposal_id)\
            .eq('org_id', org_id)\
            .single()\
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Proposal not found")

        return {
            "success": True,
            "proposal": result.data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching proposal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proposals/{proposal_id}/review", summary="Approve or reject proposal")
async def review_proposal(
    proposal_id: str,
    request: ProposalApprovalRequest
):
    """
    Approve or reject a policy proposal.

    If approved, the policy is created and activated.
    """
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        # Get the proposal
        proposal_query = supabase.table('cil_policy_proposals')\
            .select('*')\
            .eq('id', proposal_id)\
            .single()\
            .execute()

        if not proposal_query.data:
            raise HTTPException(status_code=404, detail="Proposal not found")

        proposal = proposal_query.data

        if proposal['status'] != 'pending':
            raise HTTPException(status_code=400, detail=f"Proposal already {proposal['status']}")

        # Update proposal status
        new_status = 'approved' if request.approved else 'rejected'

        supabase.table('cil_policy_proposals')\
            .update({
                'status': new_status,
                'reviewed_by': request.reviewed_by,
                'reviewed_at': datetime.utcnow().isoformat(),
                'review_notes': request.review_notes
            })\
            .eq('id', proposal_id)\
            .execute()

        # If approved, create and activate the policy
        if request.approved:
            policy_service = get_cil_policy_service()

            import json
            proposed_config = json.loads(proposal['proposed_policy_config']) if isinstance(proposal['proposed_policy_config'], str) else proposal['proposed_policy_config']
            change_details = json.loads(proposal['change_details']) if isinstance(proposal['change_details'], str) else proposal['change_details']

            policy = policy_service.create_policy(
                org_id=proposal['org_id'],
                policy_config=proposed_config,
                learning_cycle_id=proposal.get('learning_cycle_id'),
                change_reason=f"Approved proposal {proposal_id}",
                change_summary=change_details,
                approval_required=False,  # Already approved
                created_by=request.reviewed_by or 'admin'
            )

            if not policy:
                raise HTTPException(status_code=500, detail="Failed to create policy from approved proposal")

        return {
            "success": True,
            "status": new_status,
            "message": f"Proposal {new_status} successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reviewing proposal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Learning Cycles Endpoints
# ============================================================================

@router.get("/learning-cycles/{org_id}", summary="Get learning cycle history")
async def get_learning_cycles(
    org_id: str,
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """Get history of CIL meta-learning cycles."""
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        query = supabase.table('cil_learning_cycles')\
            .select('*')\
            .eq('org_id', org_id)\
            .order('started_at', desc=True)\
            .limit(limit)

        if status:
            query = query.eq('status', status)

        result = query.execute()

        return {
            "success": True,
            "cycles": result.data or [],
            "count": len(result.data) if result.data else 0
        }

    except Exception as e:
        logger.error(f"Error fetching learning cycles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning-cycles/{org_id}/{cycle_id}", summary="Get specific learning cycle")
async def get_learning_cycle(
    org_id: str,
    cycle_id: str
):
    """Get detailed information about a specific learning cycle."""
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        result = supabase.table('cil_learning_cycles')\
            .select('*')\
            .eq('id', cycle_id)\
            .eq('org_id', org_id)\
            .single()\
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Learning cycle not found")

        return {
            "success": True,
            "cycle": result.data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching learning cycle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# System Status
# ============================================================================

@router.get("/status", summary="Get CIL system status")
async def get_cil_status():
    """Get overall CIL system health and status."""
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        # Count policies, proposals, learning cycles
        policies_count = supabase.table('cil_policies').select('id', count='exact').execute()
        proposals_count = supabase.table('cil_policy_proposals').select('id', count='exact').eq('status', 'pending').execute()
        cycles_count = supabase.table('cil_learning_cycles').select('id', count='exact').execute()

        # Get worker status
        worker = get_cil_worker()
        worker_status = worker.get_status()

        return {
            "success": True,
            "status": "operational",
            "statistics": {
                "total_policies": policies_count.count or 0,
                "pending_proposals": proposals_count.count or 0,
                "learning_cycles_completed": cycles_count.count or 0
            },
            "worker": worker_status,
            "version": "1.0.0"
        }

    except Exception as e:
        logger.error(f"Error fetching CIL status: {e}", exc_info=True)
        return {
            "success": False,
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# Admin/Manual Operations
# ============================================================================

@router.post("/admin/trigger-learning/{org_id}", summary="Manually trigger learning cycle")
async def trigger_learning_cycle(
    org_id: str,
    triggered_by: Optional[str] = Query(None, description="Admin user ID")
):
    """
    Manually trigger a meta-learning cycle for an organization.

    Useful for testing or forcing immediate learning.
    """
    try:
        worker = get_cil_worker()
        result = await worker.trigger_learning_cycle_now(org_id)

        if result:
            return {
                "success": True,
                "result": result,
                "message": f"Learning cycle completed: {result.get('proposals_created', 0)} proposals created"
            }
        else:
            return {
                "success": True,
                "result": None,
                "message": "No action needed - insufficient data or no significant findings"
            }

    except Exception as e:
        logger.error(f"Error triggering learning cycle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/worker-status", summary="Get CIL worker status")
async def get_worker_status():
    """Get detailed CIL worker status and scheduled job information."""
    try:
        worker = get_cil_worker()
        status = worker.get_status()

        return {
            "success": True,
            "worker": status
        }

    except Exception as e:
        logger.error(f"Error fetching worker status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Prompt Optimization Endpoints
# ============================================================================

class PromptTemplateRequest(BaseModel):
    """Request to create a new prompt template"""
    template_name: str
    template_type: str  # 'cde_intent', 'agent_system', 'rag_query', etc.
    prompt_text: str
    variables: List[str]
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ABTestRequest(BaseModel):
    """Request to create an A/B test"""
    template_name: str
    challenger_prompt_text: str
    test_duration_hours: int = 168  # 7 days
    traffic_split: float = 0.5  # 50/50
    test_hypothesis: Optional[str] = None


class PromptUsageRequest(BaseModel):
    """Request to record prompt usage"""
    template_id: str
    outcome: str  # 'success' or 'failure'
    response_time_ms: Optional[int] = None
    quality_score: Optional[float] = None
    user_feedback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/prompts/{org_id}/create", summary="Create prompt template")
async def create_prompt_template(
    org_id: str,
    request: PromptTemplateRequest
):
    """
    Create a new prompt template for optimization tracking.

    Templates support variable substitution using {variable_name} syntax.
    """
    try:
        optimizer = get_cil_prompt_optimizer()

        template = optimizer.create_prompt_template(
            org_id=org_id,
            template_name=request.template_name,
            template_type=request.template_type,
            prompt_text=request.prompt_text,
            variables=request.variables,
            description=request.description,
            metadata=request.metadata
        )

        if not template:
            raise HTTPException(status_code=500, detail="Failed to create template")

        return {
            "success": True,
            "template": template,
            "message": f"Created template '{request.template_name}' v1"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating prompt template: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/{org_id}/ab-test", summary="Create A/B test")
async def create_ab_test(
    org_id: str,
    request: ABTestRequest
):
    """
    Create an A/B test with a new prompt variant.

    Traffic is split between champion (current) and challenger (new variant).
    Test runs for specified duration, then winner is automatically selected.
    """
    try:
        optimizer = get_cil_prompt_optimizer()

        challenger = optimizer.create_ab_test(
            org_id=org_id,
            template_name=request.template_name,
            challenger_prompt_text=request.challenger_prompt_text,
            test_duration_hours=request.test_duration_hours,
            traffic_split=request.traffic_split,
            test_hypothesis=request.test_hypothesis
        )

        if not challenger:
            raise HTTPException(status_code=400, detail="Failed to create A/B test")

        return {
            "success": True,
            "challenger": challenger,
            "message": f"Started A/B test for '{request.template_name}'"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts/{org_id}/{template_name}", summary="Get prompt template")
async def get_prompt_template(
    org_id: str,
    template_name: str,
    user_id: Optional[str] = Query(None, description="User ID for consistent A/B assignment")
):
    """
    Get prompt template for use.

    If A/B test is active, automatically routes traffic based on test configuration.
    Returns the selected variant (champion or challenger).
    """
    try:
        optimizer = get_cil_prompt_optimizer()

        template = optimizer.get_prompt_for_use(
            org_id=org_id,
            template_name=template_name,
            user_id=user_id
        )

        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        return {
            "success": True,
            "template": template
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt template: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/record-usage", summary="Record prompt usage")
async def record_prompt_usage(
    request: PromptUsageRequest
):
    """
    Record prompt usage outcome for performance tracking.

    Call this after using a prompt to track its success rate and performance.
    """
    try:
        optimizer = get_cil_prompt_optimizer()

        optimizer.record_prompt_usage(
            template_id=request.template_id,
            outcome=request.outcome,
            response_time_ms=request.response_time_ms,
            quality_score=request.quality_score,
            user_feedback=request.user_feedback,
            metadata=request.metadata
        )

        return {
            "success": True,
            "message": "Usage recorded"
        }

    except Exception as e:
        logger.error(f"Error recording prompt usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts/{org_id}/{template_name}/performance", summary="Get template performance")
async def get_template_performance(
    org_id: str,
    template_name: str,
    days_back: int = Query(30, ge=1, le=365)
):
    """
    Get performance statistics for a prompt template.

    Shows all versions, success rates, usage counts, and A/B test status.
    """
    try:
        optimizer = get_cil_prompt_optimizer()

        performance = optimizer.get_template_performance(
            org_id=org_id,
            template_name=template_name,
            days_back=days_back
        )

        if 'error' in performance:
            raise HTTPException(status_code=404, detail=performance['error'])

        return {
            "success": True,
            "performance": performance
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template performance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/{org_id}/evaluate-tests", summary="Evaluate A/B tests")
async def evaluate_ab_tests(
    org_id: str
):
    """
    Evaluate all active A/B tests and determine winners.

    Automatically promotes winners and deactivates losers.
    """
    try:
        optimizer = get_cil_prompt_optimizer()

        results = await optimizer.evaluate_ab_tests(org_id)

        return {
            "success": True,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Error evaluating A/B tests: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Phase 6, Step 2: Ads Optimization Endpoints
# ============================================================================

@router.get("/ads/telemetry/{org_id}", summary="Get ads telemetry data")
async def get_ads_telemetry(
    org_id: str,
    platform: Optional[str] = Query(None, description="Filter by platform (google_ads, meta_ads)"),
    days_back: int = Query(30, ge=1, le=90, description="Days of history"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get aggregated ad platform telemetry data.

    Shows 30-day rolling metrics including ROAS, CTR, conversions, and performance scores.
    """
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        query = supabase.table('cil_ads_telemetry')\
            .select('*')\
            .eq('org_id', org_id)\
            .order('created_at', desc=True)\
            .limit(limit)

        if platform:
            query = query.eq('platform', platform)

        result = query.execute()

        return {
            "success": True,
            "telemetry": result.data or [],
            "count": len(result.data) if result.data else 0
        }

    except Exception as e:
        logger.error(f"Error fetching ads telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ads/platform-comparison/{org_id}", summary="Compare ad platform performance")
async def get_platform_comparison(
    org_id: str,
    days: int = Query(30, ge=7, le=90, description="Days to analyze")
):
    """
    Compare Google Ads vs Meta Ads performance.

    Returns aggregated metrics including ROAS, CTR, spend, conversions, and performance scores.
    """
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        # Call database function for platform comparison
        result = supabase.rpc(
            'get_platform_performance_comparison',
            {'p_org_id': org_id, 'p_days': days}
        ).execute()

        platforms = result.data or []

        # Calculate insights
        insights = []
        if len(platforms) >= 2:
            sorted_by_roas = sorted(platforms, key=lambda x: x.get('avg_roas', 0), reverse=True)
            best = sorted_by_roas[0]
            worst = sorted_by_roas[-1]

            if best['avg_roas'] > worst['avg_roas'] * 1.5:
                insights.append({
                    "type": "platform_preference",
                    "message": f"{best['platform']} outperforms {worst['platform']} by {(best['avg_roas'] / worst['avg_roas'] - 1) * 100:.1f}% in ROAS",
                    "recommendation": f"Consider shifting budget to {best['platform']}"
                })

        return {
            "success": True,
            "platforms": platforms,
            "insights": insights,
            "period_days": days
        }

    except Exception as e:
        logger.error(f"Error fetching platform comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ads/underperformers/{org_id}", summary="Identify underperforming campaigns")
async def get_underperforming_campaigns(
    org_id: str,
    min_ctr: float = Query(0.5, description="Minimum CTR threshold (%)"),
    min_roas: float = Query(1.0, description="Minimum ROAS threshold"),
    min_quality_score: float = Query(3.0, description="Minimum quality score (Google Ads)")
):
    """
    Find campaigns that are underperforming based on thresholds.

    Useful for identifying candidates for pausing or optimization.
    """
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        # Call database function
        result = supabase.rpc(
            'identify_underperforming_campaigns',
            {
                'p_org_id': org_id,
                'p_min_ctr': min_ctr,
                'p_min_roas': min_roas,
                'p_min_quality_score': min_quality_score
            }
        ).execute()

        campaigns = result.data or []

        return {
            "success": True,
            "campaigns": campaigns,
            "count": len(campaigns),
            "thresholds": {
                "min_ctr": min_ctr,
                "min_roas": min_roas,
                "min_quality_score": min_quality_score
            }
        }

    except Exception as e:
        logger.error(f"Error identifying underperformers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ads/top-performers/{org_id}", summary="Identify top-performing campaigns")
async def get_top_performing_campaigns(
    org_id: str,
    min_roas: float = Query(4.0, description="Minimum ROAS threshold"),
    min_ctr: float = Query(2.5, description="Minimum CTR threshold (%)"),
    min_conversions: int = Query(50, description="Minimum conversion count")
):
    """
    Find high-performing campaigns worth replicating.

    Useful for identifying candidates for cloning or budget increases.
    """
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        # Call database function
        result = supabase.rpc(
            'identify_top_performing_campaigns',
            {
                'p_org_id': org_id,
                'p_min_roas': min_roas,
                'p_min_ctr': min_ctr,
                'p_min_conversions': min_conversions
            }
        ).execute()

        campaigns = result.data or []

        return {
            "success": True,
            "campaigns": campaigns,
            "count": len(campaigns),
            "thresholds": {
                "min_roas": min_roas,
                "min_ctr": min_ctr,
                "min_conversions": min_conversions
            }
        }

    except Exception as e:
        logger.error(f"Error identifying top performers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ads/optimization-history/{org_id}", summary="Get ads optimization history")
async def get_ads_optimization_history(
    org_id: str,
    platform: Optional[str] = Query(None, description="Filter by platform"),
    success_only: bool = Query(False, description="Show only successful optimizations"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get history of applied ads optimizations and their outcomes.

    Tracks what CIL has changed and whether it improved performance.
    """
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        query = supabase.table('cil_ads_optimization_history')\
            .select('*')\
            .eq('org_id', org_id)\
            .order('applied_at', desc=True)\
            .limit(limit)

        if platform:
            query = query.eq('platform', platform)

        if success_only:
            query = query.eq('success', True)

        result = query.execute()

        history = result.data or []

        # Calculate summary stats
        if history:
            total_optimizations = len(history)
            successful = sum(1 for h in history if h.get('success') is True)
            avg_actual_gain = sum(h.get('actual_gain', 0) for h in history if h.get('actual_gain')) / max(sum(1 for h in history if h.get('actual_gain')), 1)

            summary = {
                "total_optimizations": total_optimizations,
                "successful": successful,
                "success_rate": round(successful / total_optimizations, 3) if total_optimizations > 0 else 0,
                "avg_actual_gain": round(avg_actual_gain, 2)
            }
        else:
            summary = {}

        return {
            "success": True,
            "history": history,
            "count": len(history),
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error fetching optimization history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ads/proposals/{org_id}", summary="Get ads optimization proposals")
async def get_ads_proposals(
    org_id: str,
    status: str = Query("pending", description="Filter by status"),
    proposal_type: Optional[str] = Query(None, description="Filter by proposal type"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get CIL-generated ads optimization proposals.

    Includes budget adjustments, campaign pausing, cloning, and platform shifts.
    """
    try:
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        query = supabase.table('cil_policy_proposals')\
            .select('*')\
            .eq('org_id', org_id)\
            .eq('status', status)\
            .in_('proposal_type', [
                'budget_adjustment',
                'pause_campaign',
                'clone_campaign',
                'platform_shift',
                'increase_budget',
                'decrease_budget'
            ])\
            .order('created_at', desc=True)\
            .limit(limit)

        if proposal_type:
            query = query.eq('proposal_type', proposal_type)

        result = query.execute()

        proposals = result.data or []

        # Group by type
        by_type = {}
        for proposal in proposals:
            ptype = proposal.get('proposal_type', 'unknown')
            if ptype not in by_type:
                by_type[ptype] = 0
            by_type[ptype] += 1

        return {
            "success": True,
            "proposals": proposals,
            "count": len(proposals),
            "breakdown_by_type": by_type
        }

    except Exception as e:
        logger.error(f"Error fetching ads proposals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ads/trigger-optimization/{org_id}", summary="Manually trigger ads optimization")
async def trigger_ads_optimization(
    org_id: str,
    triggered_by: Optional[str] = Query(None, description="Admin user ID")
):
    """
    Manually trigger ads optimization algorithms for an organization.

    Runs all 4 algorithms and generates proposals:
    1. Budget reallocation (platform comparison)
    2. Underperformer detection (pause recommendations)
    3. Top performer cloning (replication recommendations)
    4. Platform preference learning (insights)
    """
    try:
        from app.services.cil_ads_optimizer import get_cil_ads_optimizer
        from app.db import get_supabase_admin_client
        supabase = get_supabase_admin_client()

        # Create a temporary learning cycle for tracking
        cycle_data = {
            'org_id': org_id,
            'cycle_type': 'manual_ads_optimization',
            'algorithms_run': ['budget_reallocation', 'underperformer_detection', 'top_performer_cloning', 'platform_learning'],
            'status': 'running'
        }

        cycle_result = supabase.table('cil_learning_cycles').insert(cycle_data).execute()
        cycle_id = cycle_result.data[0]['id'] if cycle_result.data else None

        # Run ads optimizer
        ads_optimizer = get_cil_ads_optimizer()
        proposals = await ads_optimizer.generate_all_proposals(org_id, cycle_id)

        # Store proposals
        proposals_created = 0
        if proposals:
            for proposal in proposals:
                supabase.table('cil_policy_proposals').insert(proposal).execute()
                proposals_created += 1

        # Update learning cycle
        if cycle_id:
            supabase.table('cil_learning_cycles')\
                .update({
                    'status': 'completed',
                    'completed_at': datetime.utcnow().isoformat(),
                    'proposals_created': proposals_created
                })\
                .eq('id', cycle_id)\
                .execute()

        return {
            "success": True,
            "proposals_created": proposals_created,
            "cycle_id": cycle_id,
            "message": f"Generated {proposals_created} ads optimization proposals"
        }

    except Exception as e:
        logger.error(f"Error triggering ads optimization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ads/telemetry/ingest/{org_id}", summary="Trigger ads telemetry ingestion")
async def trigger_ads_telemetry_ingestion(
    org_id: str
):
    """
    Manually trigger ads telemetry ingestion for an organization.

    Aggregates last 30 days of ad metrics from ad_campaigns and ad_metrics tables.
    """
    try:
        from app.services.cil_telemetry_service import get_cil_telemetry_service
        from datetime import datetime, timedelta

        telemetry_service = get_cil_telemetry_service()
        cutoff_time = datetime.utcnow() - timedelta(hours=1)

        records_ingested = await telemetry_service._ingest_ads_telemetry(
            org_id=org_id,
            cutoff_time=cutoff_time
        )

        return {
            "success": True,
            "records_ingested": records_ingested,
            "message": f"Ingested {records_ingested} ads telemetry records"
        }

    except Exception as e:
        logger.error(f"Error ingesting ads telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Phase 6.5: Knowledge Graph - Ads Integration
# ============================================================================

@router.post("/ads/kg/link-campaign/{campaign_id}", summary="Link campaign to knowledge graph")
async def link_campaign_to_kg(
    campaign_id: str,
    org_id: str = Query(..., description="Organization ID")
):
    """
    Link an ad campaign to the knowledge graph.

    Creates KG entities for:
    - Campaign performance metrics
    - Platform relationships
    """
    try:
        from app.services.kg_ads_linker import get_kg_ads_linker

        kg_linker = get_kg_ads_linker()

        result = await kg_linker.link_campaign_performance(
            campaign_id=campaign_id,
            org_id=org_id
        )

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Failed to link campaign')
            )

        return {
            "success": True,
            "result": result,
            "message": f"Linked campaign to KG: {result.get('entities_created', 0)} entities, "
                       f"{result.get('edges_created', 0)} edges"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error linking campaign to KG: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ads/kg/link-optimization/{optimization_id}", summary="Link optimization to KG")
async def link_optimization_to_kg(
    optimization_id: str,
    org_id: str = Query(..., description="Organization ID")
):
    """
    Link a CIL optimization action to the knowledge graph.

    Creates:
    - Optimization action entity
    - Relationship to campaign
    - Relationship to CIL agent
    """
    try:
        from app.services.kg_ads_linker import get_kg_ads_linker

        kg_linker = get_kg_ads_linker()

        result = await kg_linker.link_optimization_action(
            optimization_id=optimization_id,
            org_id=org_id
        )

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Failed to link optimization')
            )

        return {
            "success": True,
            "result": result,
            "message": "Linked optimization action to KG"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error linking optimization to KG: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ads/kg/campaign-graph/{campaign_id}", summary="Get campaign knowledge graph")
async def get_campaign_knowledge_graph(
    campaign_id: str
):
    """
    Get full knowledge graph for a campaign.

    Returns:
    - All entities linked to campaign
    - All relationships between entities
    - Visualization-ready data
    """
    try:
        from app.services.kg_ads_linker import get_kg_ads_linker

        kg_linker = get_kg_ads_linker()

        graph = await kg_linker.get_campaign_graph(campaign_id=campaign_id)

        if not graph.get('success'):
            raise HTTPException(
                status_code=404,
                detail=graph.get('message', 'Campaign graph not found')
            )

        return {
            "success": True,
            "graph": graph,
            "node_count": graph.get('node_count', 0),
            "edge_count": graph.get('edge_count', 0)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ads/kg/batch-link/{org_id}", summary="Batch link campaigns to KG")
async def batch_link_campaigns(
    org_id: str,
    days_back: int = Query(7, ge=1, le=90, description="Days of campaigns to link")
):
    """
    Batch link recent campaigns to knowledge graph.

    Useful for:
    - Initial setup
    - Backfilling data
    - Rebuilding graph
    """
    try:
        from app.services.kg_ads_linker import get_kg_ads_linker

        kg_linker = get_kg_ads_linker()

        result = await kg_linker.link_campaigns_batch(
            org_id=org_id,
            days_back=days_back
        )

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Batch linking failed')
            )

        return {
            "success": True,
            "result": result,
            "message": f"Batch linked {result.get('campaigns_processed', 0)} campaigns"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch linking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ads/kg/insights/{org_id}", summary="Get campaign insights from KG")
async def get_campaign_kg_insights(
    org_id: str,
    min_confidence: float = Query(0.7, ge=0.0, le=1.0, description="Minimum confidence")
):
    """
    Discover insights from campaign knowledge graph patterns.

    Analyzes:
    - Successful optimization patterns
    - Platform performance patterns
    - Campaign relationships
    """
    try:
        from app.services.kg_ads_linker import get_kg_ads_linker

        kg_linker = get_kg_ads_linker()

        insights = await kg_linker.find_campaign_insights(
            org_id=org_id,
            min_confidence=min_confidence
        )

        return {
            "success": True,
            "insights": insights,
            "count": len(insights)
        }

    except Exception as e:
        logger.error(f"Error getting KG insights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
