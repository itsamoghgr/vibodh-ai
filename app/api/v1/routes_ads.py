"""
Ads API Routes - Phase 6
REST endpoints for managing ad platforms and viewing metrics
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
from pydantic import BaseModel

from app.db import get_supabase_admin_client
from app.services.ads_ingestion_service import get_ads_ingestion_service
from app.services.google_ads_service import get_google_ads_service
from app.services.meta_ads_service import get_meta_ads_service
from app.core.logging import logger


router = APIRouter(prefix="/ads", tags=["Ads"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SyncRequest(BaseModel):
    """Request to trigger ad data sync"""
    days_back: int = 90
    force_refresh: bool = False


class SimulateRequest(BaseModel):
    """Request to generate synthetic data"""
    platform: str  # "google_ads" or "meta_ads"
    num_campaigns: int = 5
    days_back: int = 90


# ============================================================================
# Ad Accounts
# ============================================================================

@router.get("/accounts/{org_id}")
async def list_ad_accounts(
    org_id: str,
    platform: Optional[str] = Query(None, description="Filter by platform: google_ads or meta_ads")
):
    """
    List all ad accounts for an organization.

    Args:
        org_id: Organization ID
        platform: Optional platform filter

    Returns:
        List of ad accounts with metadata
    """
    try:
        supabase = get_supabase_admin_client()

        query = supabase.table("ad_accounts")\
            .select("*")\
            .eq("org_id", org_id)

        if platform:
            query = query.eq("platform", platform)

        result = query.order("created_at", desc=True).execute()

        return {
            "accounts": result.data,
            "count": len(result.data)
        }

    except Exception as e:
        logger.error(f"Failed to list ad accounts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accounts/{org_id}/{account_id}")
async def get_ad_account(org_id: str, account_id: str):
    """Get details of a specific ad account."""
    try:
        supabase = get_supabase_admin_client()

        result = supabase.table("ad_accounts")\
            .select("*")\
            .eq("id", account_id)\
            .eq("org_id", org_id)\
            .single()\
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Ad account not found")

        return result.data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ad account: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Campaigns
# ============================================================================

@router.get("/campaigns/{org_id}")
async def list_campaigns(
    org_id: str,
    platform: Optional[str] = Query(None, description="Filter by platform"),
    account_id: Optional[str] = Query(None, description="Filter by account ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, le=500)
):
    """
    List campaigns for an organization.

    Args:
        org_id: Organization ID
        platform: Optional platform filter
        account_id: Optional account ID filter
        status: Optional status filter
        limit: Maximum number of campaigns to return

    Returns:
        List of campaigns
    """
    try:
        supabase = get_supabase_admin_client()

        query = supabase.table("ad_campaigns")\
            .select("*")\
            .eq("org_id", org_id)

        if platform:
            query = query.eq("platform", platform)

        if account_id:
            query = query.eq("account_id", account_id)

        if status:
            query = query.eq("status", status)

        result = query.order("created_at", desc=True).limit(limit).execute()

        return {
            "campaigns": result.data,
            "count": len(result.data)
        }

    except Exception as e:
        logger.error(f"Failed to list campaigns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/campaigns/{org_id}/{campaign_id}")
async def get_campaign(org_id: str, campaign_id: str):
    """Get details of a specific campaign."""
    try:
        supabase = get_supabase_admin_client()

        result = supabase.table("ad_campaigns")\
            .select("*")\
            .eq("id", campaign_id)\
            .eq("org_id", org_id)\
            .single()\
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Campaign not found")

        return result.data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get campaign: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Metrics
# ============================================================================

@router.get("/metrics/{campaign_id}")
async def get_campaign_metrics(
    campaign_id: str,
    start_date: Optional[date] = Query(None, description="Start date (default: 30 days ago)"),
    end_date: Optional[date] = Query(None, description="End date (default: today)")
):
    """
    Get time-series metrics for a campaign.

    Args:
        campaign_id: Campaign UUID
        start_date: Start date for metrics
        end_date: End date for metrics

    Returns:
        List of daily metrics
    """
    try:
        supabase = get_supabase_admin_client()

        # Default date range: last 30 days
        if not end_date:
            end_date = datetime.now().date()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        result = supabase.table("ad_metrics")\
            .select("*")\
            .eq("campaign_id", campaign_id)\
            .gte("metric_date", start_date.isoformat())\
            .lte("metric_date", end_date.isoformat())\
            .order("metric_date", desc=False)\
            .execute()

        return {
            "campaign_id": campaign_id,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "metrics": result.data,
            "count": len(result.data)
        }

    except Exception as e:
        logger.error(f"Failed to get campaign metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{campaign_id}/summary")
async def get_campaign_metrics_summary(
    campaign_id: str,
    days: int = Query(30, description="Number of days to summarize")
):
    """
    Get aggregated metrics summary for a campaign.

    Uses database function for efficient aggregation.
    """
    try:
        supabase = get_supabase_admin_client()

        # Call database function
        result = supabase.rpc(
            "get_campaign_performance_summary",
            {"p_campaign_id": campaign_id, "p_days": days}
        ).execute()

        if not result.data or len(result.data) == 0:
            return {
                "campaign_id": campaign_id,
                "days": days,
                "summary": None,
                "message": "No metrics found for this campaign"
            }

        return {
            "campaign_id": campaign_id,
            "days": days,
            "summary": result.data[0]
        }

    except Exception as e:
        logger.error(f"Failed to get campaign summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analytics & Insights
# ============================================================================

@router.get("/analytics/{org_id}")
async def get_analytics_overview(
    org_id: str,
    platform: Optional[str] = Query(None, description="Filter by platform"),
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get aggregated analytics across all campaigns.

    Returns:
        Overall performance metrics and top campaigns
    """
    try:
        supabase = get_supabase_admin_client()

        # Get all campaigns
        campaigns_query = supabase.table("ad_campaigns")\
            .select("id, campaign_name, platform, status")\
            .eq("org_id", org_id)

        if platform:
            campaigns_query = campaigns_query.eq("platform", platform)

        campaigns = campaigns_query.execute().data

        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        # Aggregate metrics across all campaigns
        total_impressions = 0
        total_clicks = 0
        total_spend = 0
        total_conversions = 0
        total_conversion_value = 0

        campaign_summaries = []

        for campaign in campaigns:
            # Get metrics for each campaign
            metrics = supabase.table("ad_metrics")\
                .select("*")\
                .eq("campaign_id", campaign["id"])\
                .gte("metric_date", start_date.isoformat())\
                .lte("metric_date", end_date.isoformat())\
                .execute()

            if metrics.data:
                campaign_impressions = sum(m["impressions"] for m in metrics.data)
                campaign_clicks = sum(m["clicks"] for m in metrics.data)
                campaign_spend = sum(m["spend"] for m in metrics.data)
                campaign_conversions = sum(m.get("conversions", 0) for m in metrics.data)
                campaign_conv_value = sum(m.get("conversion_value", 0) or 0 for m in metrics.data)

                total_impressions += campaign_impressions
                total_clicks += campaign_clicks
                total_spend += campaign_spend
                total_conversions += campaign_conversions
                total_conversion_value += campaign_conv_value

                # Add to summaries
                campaign_summaries.append({
                    "id": campaign["id"],
                    "name": campaign["campaign_name"],
                    "platform": campaign["platform"],
                    "impressions": campaign_impressions,
                    "clicks": campaign_clicks,
                    "ctr": (campaign_clicks / campaign_impressions * 100) if campaign_impressions > 0 else 0,
                    "spend": campaign_spend,
                    "conversions": campaign_conversions,
                    "roas": (campaign_conv_value / campaign_spend) if campaign_spend > 0 else 0
                })

        # Calculate overall metrics
        avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0
        overall_roas = (total_conversion_value / total_spend) if total_spend > 0 else 0

        # Sort campaigns by performance (ROAS)
        top_campaigns = sorted(campaign_summaries, key=lambda x: x["roas"], reverse=True)[:10]

        return {
            "org_id": org_id,
            "platform": platform,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "overall_metrics": {
                "impressions": total_impressions,
                "clicks": total_clicks,
                "ctr": round(avg_ctr, 2),
                "spend": round(total_spend, 2),
                "conversions": total_conversions,
                "conversion_value": round(total_conversion_value, 2),
                "avg_cpc": round(avg_cpc, 2),
                "roas": round(overall_roas, 2)
            },
            "top_campaigns": top_campaigns,
            "total_campaigns": len(campaigns)
        }

    except Exception as e:
        logger.error(f"Failed to get analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/{org_id}/top-campaigns")
async def get_top_campaigns(
    org_id: str,
    metric: str = Query("roas", description="Metric to rank by: roas, ctr, conversions, engagement_rate"),
    platform: Optional[str] = Query(None),
    days: int = Query(30),
    limit: int = Query(10, le=50)
):
    """
    Get top performing campaigns by a specific metric.

    Uses database function for efficient ranking.
    """
    try:
        supabase = get_supabase_admin_client()

        # Call database function
        result = supabase.rpc(
            "get_top_campaigns_by_metric",
            {
                "p_org_id": org_id,
                "p_platform": platform,
                "p_metric": metric,
                "p_days": days,
                "p_limit": limit
            }
        ).execute()

        return {
            "org_id": org_id,
            "metric": metric,
            "platform": platform,
            "days": days,
            "top_campaigns": result.data
        }

    except Exception as e:
        logger.error(f"Failed to get top campaigns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Sync & Ingestion
# ============================================================================

@router.post("/sync/{org_id}")
async def trigger_sync(
    org_id: str,
    account_id: Optional[str] = Query(None, description="Specific account ID to sync"),
    request: SyncRequest = SyncRequest()
):
    """
    Trigger ad data synchronization for an organization.

    Args:
        org_id: Organization ID
        account_id: Optional specific account to sync (otherwise syncs all)
        request: Sync configuration

    Returns:
        Sync job results
    """
    try:
        supabase = get_supabase_admin_client()
        ingestion_service = get_ads_ingestion_service(supabase)

        # Get accounts to sync
        if account_id:
            accounts_query = supabase.table("ad_accounts")\
                .select("*")\
                .eq("id", account_id)\
                .eq("org_id", org_id)
        else:
            accounts_query = supabase.table("ad_accounts")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("status", "active")

        accounts = accounts_query.execute().data

        if not accounts:
            raise HTTPException(status_code=404, detail="No ad accounts found")

        # Trigger sync for each account
        results = []
        for account in accounts:
            try:
                result = ingestion_service.sync_account_data(
                    account_id=account["id"],
                    org_id=org_id,
                    days_back=request.days_back,
                    force_refresh=request.force_refresh
                )
                results.append({
                    "account_id": account["id"],
                    "account_name": account["account_name"],
                    "platform": account["platform"],
                    **result
                })
            except Exception as e:
                logger.error(f"Sync failed for account {account['id']}: {e}")
                results.append({
                    "account_id": account["id"],
                    "account_name": account["account_name"],
                    "platform": account["platform"],
                    "status": "failed",
                    "error": str(e)
                })

        return {
            "org_id": org_id,
            "accounts_synced": len(results),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger sync: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync/status/{org_id}")
async def get_sync_status(
    org_id: str,
    limit: int = Query(20, le=100)
):
    """
    Get recent sync job history for an organization.

    Returns:
        List of recent sync jobs with status
    """
    try:
        supabase = get_supabase_admin_client()

        result = supabase.table("ad_sync_jobs")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("started_at", desc=True)\
            .limit(limit)\
            .execute()

        return {
            "org_id": org_id,
            "jobs": result.data,
            "count": len(result.data)
        }

    except Exception as e:
        logger.error(f"Failed to get sync status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Mock/Simulation (Development)
# ============================================================================

@router.post("/simulate/{org_id}")
async def simulate_data(
    org_id: str,
    request: SimulateRequest
):
    """
    Generate synthetic ad data for testing (mock mode only).

    Creates mock ad account, campaigns, and metrics.
    """
    try:
        from app.core.config import settings

        if not settings.ADS_MOCK_MODE:
            raise HTTPException(
                status_code=403,
                detail="Data simulation only available in mock mode"
            )

        supabase = get_supabase_admin_client()
        ingestion_service = get_ads_ingestion_service(supabase)

        # Validate platform
        if request.platform not in ["google_ads", "meta_ads"]:
            raise HTTPException(status_code=400, detail="Invalid platform")

        # Generate unique account ID for mock data
        import uuid
        mock_account_id = f"mock_{request.platform}_{uuid.uuid4().hex[:8]}"

        # Create mock ad_account directly (no connections table needed)
        logger.info(f"Creating mock ad account for {request.platform}")

        mock_account = supabase.table("ad_accounts").insert({
            "org_id": org_id,
            "platform": request.platform,
            "account_id": mock_account_id,
            "account_name": f"Mock {request.platform.replace('_', ' ').title()} Account",
            "currency": "USD",
            "timezone": "UTC",
            "status": "active",
            "metadata": {
                "mock_mode": True,
                "access_token": "mock_token",
                "refresh_token": "mock_refresh",
                "simulated": True,
                "created_via": "simulate_endpoint"
            }
        }).execute()

        if not mock_account.data:
            raise HTTPException(status_code=500, detail="Failed to create mock account")

        created_account = mock_account.data[0]
        logger.info(f"Mock account created: {created_account['id']}")

        # Sync data for the mock account (generates campaigns and metrics)
        logger.info(f"Syncing mock data for account {created_account['id']}")

        sync_result = ingestion_service.sync_account_data(
            account_id=created_account["id"],
            org_id=org_id,
            days_back=request.days_back
        )

        return {
            "simulated": True,
            "platform": request.platform,
            "account": created_account,
            "sync_result": sync_result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to simulate data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Validation
# ============================================================================

@router.get("/health/{org_id}")
async def check_ads_health(org_id: str):
    """
    Check health status of ads integration for an organization.

    Returns:
        Health metrics and connection status
    """
    try:
        supabase = get_supabase_admin_client()

        # Get accounts
        accounts = supabase.table("ad_accounts")\
            .select("*")\
            .eq("org_id", org_id)\
            .execute()

        # Get recent sync jobs
        recent_syncs = supabase.table("ad_sync_jobs")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("started_at", desc=True)\
            .limit(10)\
            .execute()

        # Calculate health metrics
        total_accounts = len(accounts.data)
        active_accounts = len([a for a in accounts.data if a["status"] == "active"])

        successful_syncs = len([s for s in recent_syncs.data if s["status"] == "completed"])
        failed_syncs = len([s for s in recent_syncs.data if s["status"] == "failed"])

        last_sync = recent_syncs.data[0] if recent_syncs.data else None

        health_status = "healthy"
        if failed_syncs > successful_syncs:
            health_status = "degraded"
        elif total_accounts == 0:
            health_status = "no_accounts"

        return {
            "org_id": org_id,
            "status": health_status,
            "accounts": {
                "total": total_accounts,
                "active": active_accounts,
                "inactive": total_accounts - active_accounts
            },
            "recent_syncs": {
                "successful": successful_syncs,
                "failed": failed_syncs,
                "last_sync_at": last_sync["completed_at"] if last_sync else None
            },
            "platforms": list(set(a["platform"] for a in accounts.data))
        }

    except Exception as e:
        logger.error(f"Failed to check ads health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed-campaigns/{org_id}")
async def embed_all_campaigns(org_id: str):
    """
    Manually trigger document embedding for all campaigns.
    Debug endpoint to see what errors occur during document creation.
    """
    try:
        supabase = get_supabase_admin_client()
        ingestion = get_ads_ingestion_service(supabase)

        # Get all campaigns for this org
        campaigns = supabase.table("ad_campaigns")\
            .select("*")\
            .eq("org_id", org_id)\
            .execute()

        if not campaigns.data:
            return {
                "success": True,
                "message": "No campaigns to embed",
                "embedded": 0,
                "errors": []
            }

        embedded_count = 0
        errors = []

        for campaign in campaigns.data:
            try:
                ingestion._create_campaign_document(
                    campaign_db_id=campaign["id"],
                    org_id=org_id,
                    campaign_data=campaign
                )
                embedded_count += 1
                logger.info(f"Embedded campaign: {campaign['campaign_name']}")
            except Exception as e:
                error_msg = f"Campaign {campaign['campaign_name']}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Failed to embed campaign: {error_msg}", exc_info=True)

        return {
            "success": len(errors) == 0,
            "message": f"Embedded {embedded_count} campaigns",
            "embedded": embedded_count,
            "total": len(campaigns.data),
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Failed to embed campaigns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup/{org_id}")
async def cleanup_all_ads_data(org_id: str):
    """
    Delete all ads data for an organization (campaigns, metrics, documents, accounts).
    Use with caution - this is for testing/demo purposes.
    """
    try:
        supabase = get_supabase_admin_client()

        # Get all campaigns first (to delete child records)
        campaigns = supabase.table("ad_campaigns")\
            .select("id")\
            .eq("org_id", org_id)\
            .execute()

        campaign_ids = [c["id"] for c in campaigns.data] if campaigns.data else []

        # Get ads-related documents
        ads_docs = supabase.table("documents")\
            .select("id")\
            .eq("org_id", org_id)\
            .in_("source_type", ["google_ads", "meta_ads"])\
            .execute()

        deleted_counts = {
            "campaigns_before": len(campaign_ids),
            "documents_before": len(ads_docs.data) if ads_docs.data else 0
        }

        # Delete in correct order (child to parent)

        # 1. Delete embeddings for ads documents
        if ads_docs.data:
            doc_ids = [d["id"] for d in ads_docs.data]
            supabase.table("embeddings")\
                .delete()\
                .in_("document_id", doc_ids)\
                .execute()

        # 2. Delete ads documents
        if ads_docs.data:
            supabase.table("documents")\
                .delete()\
                .eq("org_id", org_id)\
                .in_("source_type", ["google_ads", "meta_ads"])\
                .execute()

        # 3. Delete CIL ads telemetry
        supabase.table("cil_ads_telemetry")\
            .delete()\
            .eq("org_id", org_id)\
            .execute()

        # 4. Delete ad metrics (linked via campaign_id)
        if campaign_ids:
            supabase.table("ad_metrics")\
                .delete()\
                .in_("campaign_id", campaign_ids)\
                .execute()

        # 5. Delete ad campaigns
        if campaign_ids:
            supabase.table("ad_campaigns")\
                .delete()\
                .eq("org_id", org_id)\
                .execute()

        # 6. Delete ad accounts
        supabase.table("ad_accounts")\
            .delete()\
            .eq("org_id", org_id)\
            .execute()

        # 7. Delete ad sync jobs
        supabase.table("ad_sync_jobs")\
            .delete()\
            .eq("org_id", org_id)\
            .execute()

        logger.info(f"Cleaned up all ads data for org {org_id}")

        return {
            "success": True,
            "message": "All ads data deleted successfully",
            "deleted": deleted_counts
        }

    except Exception as e:
        logger.error(f"Failed to cleanup ads data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
