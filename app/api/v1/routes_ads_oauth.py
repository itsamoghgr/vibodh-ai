"""
OAuth Routes for Ads Platform Integration - Phase 6
Handles OAuth 2.0 flows for Google Ads and Meta Ads
"""

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import RedirectResponse
from typing import Dict, Any
from datetime import datetime

from app.core.logging import logger
from app.core.config import settings
from app.db import get_supabase_admin_client
from app.services.google_ads_service import get_google_ads_service
from app.services.meta_ads_service import get_meta_ads_service

router = APIRouter(prefix="/ads/oauth", tags=["ads-oauth"])


# ============================================================================
# GOOGLE ADS OAUTH FLOW
# ============================================================================

@router.get("/google/authorize")
async def google_ads_authorize(org_id: str = Query(..., description="Organization ID")):
    """
    Step 1: Redirect user to Google OAuth consent screen.

    Args:
        org_id: Organization ID for CSRF state parameter

    Returns:
        Redirect to Google OAuth URL
    """
    try:
        supabase = get_supabase_admin_client()
        google_ads_service = get_google_ads_service(supabase)

        # Generate authorization URL
        redirect_uri = settings.GOOGLE_ADS_REDIRECT_URI
        authorization_url = google_ads_service.get_oauth_authorization_url(org_id, redirect_uri)

        logger.info(f"Redirecting to Google Ads OAuth for org {org_id}")
        return RedirectResponse(url=authorization_url)

    except Exception as e:
        logger.error(f"Failed to initiate Google Ads OAuth: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/google/callback")
async def google_ads_callback(
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="Organization ID"),
    error: str = Query(None, description="OAuth error if any")
):
    """
    Step 2: Handle OAuth callback from Google.

    Args:
        code: Authorization code from Google
        state: Organization ID (CSRF protection)
        error: Error message if authorization failed

    Returns:
        Success message with connection details
    """
    if error:
        logger.error(f"Google Ads OAuth error: {error}")
        # Redirect to frontend with error
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/dashboard/settings/integrations?error=google_ads_{error}"
        )

    try:
        org_id = state
        supabase = get_supabase_admin_client()
        google_ads_service = get_google_ads_service(supabase)

        # Exchange authorization code for tokens
        tokens = google_ads_service.exchange_authorization_code(
            authorization_code=code,
            redirect_uri=settings.GOOGLE_ADS_REDIRECT_URI
        )

        # Fetch accessible Google Ads accounts
        accounts = google_ads_service.list_accessible_customers(tokens["refresh_token"])

        # Store connection in database
        connection_data = {
            "org_id": org_id,
            "platform": "google_ads",
            "status": "active",
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_expires_at": datetime.utcnow().isoformat(),
            "metadata": {
                "scope": tokens.get("scope"),
                "accounts_count": len(accounts)
            },
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # Insert or update connection
        connection_result = supabase.table('connections').upsert(
            connection_data,
            on_conflict='org_id,platform'
        ).execute()

        connection_id = connection_result.data[0]['id']

        # Store ad accounts
        for account in accounts:
            account_data = {
                "org_id": org_id,
                "platform": "google_ads",
                "account_id": account["customer_id"],
                "account_name": account["descriptive_name"],
                "currency": account.get("currency_code", "USD"),
                "timezone": account.get("time_zone", "UTC"),
                "status": "active" if account.get("status") == "ENABLED" else "inactive",
                "connection_id": connection_id,
                "metadata": account,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            supabase.table('ad_accounts').upsert(
                account_data,
                on_conflict='org_id,platform,account_id'
            ).execute()

        logger.info(f"Successfully connected {len(accounts)} Google Ads accounts for org {org_id}")

        # Redirect to frontend success page
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/dashboard/settings/integrations?success=google_ads&accounts={len(accounts)}"
        )

    except Exception as e:
        logger.error(f"Google Ads OAuth callback error: {e}", exc_info=True)
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/dashboard/settings/integrations?error=google_ads_callback_failed"
        )


@router.post("/google/refresh")
async def google_ads_refresh_token(org_id: str):
    """
    Refresh expired Google Ads access token.

    Args:
        org_id: Organization ID

    Returns:
        New access token
    """
    try:
        supabase = get_supabase_admin_client()
        google_ads_service = get_google_ads_service(supabase)

        # Get connection from database
        connection = supabase.table('connections').select('*').eq('org_id', org_id).eq('platform', 'google_ads').single().execute()

        if not connection.data:
            raise HTTPException(status_code=404, detail="Google Ads connection not found")

        refresh_token = connection.data['refresh_token']

        # Refresh token
        new_tokens = google_ads_service.refresh_access_token(refresh_token)

        # Update connection
        supabase.table('connections').update({
            "access_token": new_tokens["access_token"],
            "token_expires_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).eq('org_id', org_id).eq('platform', 'google_ads').execute()

        logger.info(f"Refreshed Google Ads token for org {org_id}")
        return {"success": True, "expires_in": new_tokens["expires_in"]}

    except Exception as e:
        logger.error(f"Failed to refresh Google Ads token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# META ADS OAUTH FLOW
# ============================================================================

@router.get("/meta/authorize")
async def meta_ads_authorize(org_id: str = Query(..., description="Organization ID")):
    """
    Step 1: Redirect user to Meta OAuth consent screen.

    Args:
        org_id: Organization ID for CSRF state parameter

    Returns:
        Redirect to Meta OAuth URL
    """
    try:
        supabase = get_supabase_admin_client()
        meta_ads_service = get_meta_ads_service(supabase)

        # Generate authorization URL
        redirect_uri = settings.META_ADS_REDIRECT_URI
        authorization_url = meta_ads_service.get_oauth_authorization_url(org_id, redirect_uri)

        logger.info(f"Redirecting to Meta Ads OAuth for org {org_id}")
        return RedirectResponse(url=authorization_url)

    except Exception as e:
        logger.error(f"Failed to initiate Meta Ads OAuth: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meta/callback")
async def meta_ads_callback(
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="Organization ID"),
    error: str = Query(None, description="OAuth error if any"),
    error_description: str = Query(None)
):
    """
    Step 2: Handle OAuth callback from Meta.

    Args:
        code: Authorization code from Meta
        state: Organization ID (CSRF protection)
        error: Error message if authorization failed
        error_description: Detailed error description

    Returns:
        Success message with connection details
    """
    if error:
        logger.error(f"Meta Ads OAuth error: {error} - {error_description}")
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/dashboard/settings/integrations?error=meta_ads_{error}"
        )

    try:
        org_id = state
        supabase = get_supabase_admin_client()
        meta_ads_service = get_meta_ads_service(supabase)

        # Exchange authorization code for short-lived token
        short_lived_tokens = meta_ads_service.exchange_authorization_code(
            authorization_code=code,
            redirect_uri=settings.META_ADS_REDIRECT_URI
        )

        # Exchange for long-lived token (60 days)
        long_lived_tokens = meta_ads_service.exchange_short_lived_for_long_lived_token(
            short_lived_token=short_lived_tokens["access_token"]
        )

        # Fetch accessible ad accounts
        accounts = meta_ads_service.list_ad_accounts(long_lived_tokens["access_token"])

        # Store connection in database
        connection_data = {
            "org_id": org_id,
            "platform": "meta_ads",
            "status": "active",
            "access_token": long_lived_tokens["access_token"],
            "refresh_token": None,  # Meta uses long-lived tokens, not refresh tokens
            "token_expires_at": (datetime.utcnow()).isoformat(),  # ~60 days from now
            "metadata": {
                "accounts_count": len(accounts),
                "token_type": "long_lived"
            },
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # Insert or update connection
        connection_result = supabase.table('connections').upsert(
            connection_data,
            on_conflict='org_id,platform'
        ).execute()

        connection_id = connection_result.data[0]['id']

        # Store ad accounts
        for account in accounts:
            account_data = {
                "org_id": org_id,
                "platform": "meta_ads",
                "account_id": account["account_id"],
                "account_name": account["name"],
                "currency": account.get("currency", "USD"),
                "timezone": account.get("timezone_name", "UTC"),
                "status": "active" if account.get("account_status") == 1 else "inactive",
                "connection_id": connection_id,
                "metadata": account,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            supabase.table('ad_accounts').upsert(
                account_data,
                on_conflict='org_id,platform,account_id'
            ).execute()

        logger.info(f"Successfully connected {len(accounts)} Meta Ads accounts for org {org_id}")

        # Redirect to frontend success page
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/dashboard/settings/integrations?success=meta_ads&accounts={len(accounts)}"
        )

    except Exception as e:
        logger.error(f"Meta Ads OAuth callback error: {e}", exc_info=True)
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/dashboard/settings/integrations?error=meta_ads_callback_failed"
        )


@router.post("/meta/exchange-token")
async def meta_ads_exchange_token(org_id: str):
    """
    Exchange Meta Ads short-lived token for long-lived token.
    Can also be used to refresh/renew existing long-lived token.

    Args:
        org_id: Organization ID

    Returns:
        New long-lived access token
    """
    try:
        supabase = get_supabase_admin_client()
        meta_ads_service = get_meta_ads_service(supabase)

        # Get connection from database
        connection = supabase.table('connections').select('*').eq('org_id', org_id).eq('platform', 'meta_ads').single().execute()

        if not connection.data:
            raise HTTPException(status_code=404, detail="Meta Ads connection not found")

        current_token = connection.data['access_token']

        # Exchange for new long-lived token
        new_tokens = meta_ads_service.exchange_short_lived_for_long_lived_token(current_token)

        # Update connection
        supabase.table('connections').update({
            "access_token": new_tokens["access_token"],
            "token_expires_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).eq('org_id', org_id).eq('platform', 'meta_ads').execute()

        logger.info(f"Exchanged Meta Ads token for org {org_id}")
        return {"success": True, "expires_in": new_tokens["expires_in"]}

    except Exception as e:
        logger.error(f"Failed to exchange Meta Ads token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# COMMON ENDPOINTS
# ============================================================================

@router.delete("/disconnect")
async def disconnect_ads_platform(org_id: str, platform: str):
    """
    Disconnect an ads platform (revoke OAuth and delete stored accounts).

    Args:
        org_id: Organization ID
        platform: Platform to disconnect ("google_ads" or "meta_ads")

    Returns:
        Success message
    """
    if platform not in ["google_ads", "meta_ads"]:
        raise HTTPException(status_code=400, detail="Invalid platform")

    try:
        supabase = get_supabase_admin_client()

        # Delete ad accounts
        supabase.table('ad_accounts').delete().eq('org_id', org_id).eq('platform', platform).execute()

        # Delete connection
        supabase.table('connections').delete().eq('org_id', org_id).eq('platform', platform).execute()

        logger.info(f"Disconnected {platform} for org {org_id}")
        return {"success": True, "message": f"Successfully disconnected {platform}"}

    except Exception as e:
        logger.error(f"Failed to disconnect {platform}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_oauth_status(org_id: str):
    """
    Get OAuth connection status for all ads platforms.

    Args:
        org_id: Organization ID

    Returns:
        Connection status for each platform
    """
    try:
        supabase = get_supabase_admin_client()

        # Get connections
        connections = supabase.table('connections').select('*').eq('org_id', org_id).in_('platform', ['google_ads', 'meta_ads']).execute()

        # Get ad accounts count
        ad_accounts = supabase.table('ad_accounts').select('platform, count').eq('org_id', org_id).execute()

        # Build status response
        status = {
            "google_ads": {
                "connected": False,
                "accounts_count": 0,
                "last_sync": None
            },
            "meta_ads": {
                "connected": False,
                "accounts_count": 0,
                "last_sync": None
            }
        }

        # Update with actual data
        for conn in (connections.data or []):
            platform = conn['platform']
            if platform in status:
                status[platform]['connected'] = conn['status'] == 'active'
                status[platform]['last_sync'] = conn.get('updated_at')

        for account in (ad_accounts.data or []):
            platform = account['platform']
            if platform in status:
                status[platform]['accounts_count'] = account.get('count', 0)

        return status

    except Exception as e:
        logger.error(f"Failed to get OAuth status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
