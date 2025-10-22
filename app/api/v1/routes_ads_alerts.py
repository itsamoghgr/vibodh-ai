"""
Ads Alerts API Routes - Phase 6.5
Endpoints for managing and monitoring ads performance alerts
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel

from app.core.logging import logger
from app.services.ads_alert_service import get_ads_alert_service
from app.workers.ads_worker import get_ads_worker


router = APIRouter(prefix="/ads/alerts", tags=["ads-alerts"])


# Request/Response Models
class AcknowledgeAlertRequest(BaseModel):
    """Request to acknowledge an alert"""
    acknowledged_by: Optional[str] = None


class TriggerSyncRequest(BaseModel):
    """Request to manually trigger ads sync"""
    org_id: str


# ============================================================================
# Alert Management Endpoints
# ============================================================================

@router.get("/{org_id}", summary="Get ads alerts")
async def get_alerts(
    org_id: str,
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledged status"),
    severity: Optional[str] = Query(None, description="Filter by severity (low/medium/high/critical)"),
    limit: int = Query(50, ge=1, le=200, description="Max results")
):
    """
    Get ads alerts for an organization.

    Filters:
    - acknowledged: true/false/null (all)
    - severity: low/medium/high/critical
    - limit: max results (default 50)
    """
    try:
        alert_service = get_ads_alert_service()

        alerts = await alert_service.get_alerts(
            org_id=org_id,
            acknowledged=acknowledged,
            severity=severity,
            limit=limit
        )

        return {
            "success": True,
            "alerts": alerts,
            "count": len(alerts)
        }

    except Exception as e:
        logger.error(f"Error fetching alerts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{org_id}/stats", summary="Get alert statistics")
async def get_alert_stats(
    org_id: str,
    days_back: int = Query(7, ge=1, le=90, description="Days to analyze")
):
    """
    Get alert statistics for an organization.

    Returns counts by type, severity, and acknowledgment status.
    """
    try:
        alert_service = get_ads_alert_service()

        stats = await alert_service.get_alert_stats(
            org_id=org_id,
            days_back=days_back
        )

        return {
            "success": True,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error fetching alert stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/acknowledge", summary="Acknowledge alert")
async def acknowledge_alert(
    alert_id: str,
    request: AcknowledgeAlertRequest
):
    """
    Acknowledge an alert.

    Marks the alert as acknowledged and records who acknowledged it.
    """
    try:
        alert_service = get_ads_alert_service()

        success = await alert_service.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=request.acknowledged_by
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to acknowledge alert")

        return {
            "success": True,
            "message": "Alert acknowledged successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Worker Control Endpoints
# ============================================================================

@router.get("/worker/status", summary="Get ads worker status")
async def get_worker_status():
    """
    Get ads worker health and status information.

    Returns:
    - Running status
    - Scheduled jobs
    - Next run times
    - Failure tracking
    """
    try:
        worker = get_ads_worker()
        status = worker.get_status()

        return {
            "success": True,
            "worker": status
        }

    except Exception as e:
        logger.error(f"Error fetching worker status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/worker/trigger-sync", summary="Manually trigger ads sync")
async def trigger_sync(
    request: TriggerSyncRequest
):
    """
    Manually trigger ads data sync for an organization.

    Useful for:
    - Testing sync logic
    - Forcing immediate update
    - Recovering from failed scheduled sync
    """
    try:
        worker = get_ads_worker()

        result = await worker.trigger_sync_now(org_id=request.org_id)

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Sync failed')
            )

        return {
            "success": True,
            "result": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering sync: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Detection Endpoints (Manual Trigger)
# ============================================================================

@router.post("/{org_id}/detect-roas-drops", summary="Manually detect ROAS drops")
async def detect_roas_drops(
    org_id: str,
    threshold_pct: float = Query(20.0, ge=0, le=100, description="Alert threshold percentage")
):
    """
    Manually trigger ROAS drop detection for an organization.

    Useful for testing or immediate detection outside scheduled run.
    """
    try:
        alert_service = get_ads_alert_service()

        alerts = await alert_service.detect_roas_drops(
            org_id=org_id,
            threshold_pct=threshold_pct
        )

        return {
            "success": True,
            "alerts_created": len(alerts),
            "alerts": alerts,
            "message": f"Detected {len(alerts)} ROAS drop alerts"
        }

    except Exception as e:
        logger.error(f"Error detecting ROAS drops: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{org_id}/detect-budget-overages", summary="Manually detect budget overages")
async def detect_budget_overages(
    org_id: str
):
    """
    Manually trigger budget overage detection for an organization.

    Checks if any campaigns exceeded their daily budget yesterday.
    """
    try:
        alert_service = get_ads_alert_service()

        alerts = await alert_service.detect_budget_overages(
            org_id=org_id
        )

        return {
            "success": True,
            "alerts_created": len(alerts),
            "alerts": alerts,
            "message": f"Detected {len(alerts)} budget overage alerts"
        }

    except Exception as e:
        logger.error(f"Error detecting budget overages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
