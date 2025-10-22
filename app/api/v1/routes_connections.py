"""
Connection Management API Routes
"""

from fastapi import APIRouter, Query, HTTPException
from datetime import datetime, timedelta

from app.db import supabase, get_supabase_admin_client
from app.core.logging import logger, log_error
from app.services.google_ads_service import get_google_ads_service
from app.services.meta_ads_service import get_meta_ads_service
from app.services.ads_ingestion_service import get_ads_ingestion_service

router = APIRouter(prefix="/connections", tags=["Connections"])


@router.get("")
async def list_connections(org_id: str = Query(...)):
    """List all connections for an organization"""
    try:
        result = supabase.table("connections")\
            .select("*")\
            .eq("org_id", org_id)\
            .execute()

        return {"connections": result.data}
    except Exception as e:
        log_error(e, context="List connections")
        raise HTTPException(status_code=500, detail=f"Failed to fetch connections: {str(e)}")


@router.delete("/{connection_id}")
async def delete_connection(connection_id: str):
    """Delete a connection"""
    try:
        supabase.table("connections")\
            .delete()\
            .eq("id", connection_id)\
            .execute()

        return {"message": "Connection deleted successfully"}
    except Exception as e:
        log_error(e, context="Delete connection")
        raise HTTPException(status_code=500, detail=f"Failed to delete connection: {str(e)}")


# ============================================================================
# Google Ads OAuth (Phase 6)
# ============================================================================

@router.get("/google-ads/connect")
async def connect_google_ads(
    org_id: str = Query(..., description="Organization ID"),
    redirect_uri: str = Query(..., description="OAuth callback URL")
):
    """
    Initiate Google Ads OAuth 2.0 flow.

    Returns authorization URL for user to grant access.
    """
    try:
        supabase_admin = get_supabase_admin_client()
        google_ads_service = get_google_ads_service(supabase_admin)

        auth_url = google_ads_service.get_oauth_authorization_url(org_id, redirect_uri)

        return {
            "authorization_url": auth_url,
            "org_id": org_id
        }

    except Exception as e:
        log_error(e, context="Google Ads OAuth connect")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/google-ads/callback")
async def google_ads_callback(
    code: str = Query(..., description="Authorization code"),
    org_id: str = Query(..., description="Organization ID"),
    redirect_uri: str = Query(..., description="Redirect URI used in authorization")
):
    """
    Handle Google Ads OAuth callback.

    Exchanges authorization code for tokens and stores connection.
    """
    try:
        supabase_admin = get_supabase_admin_client()
        google_ads_service = get_google_ads_service(supabase_admin)
        ingestion_service = get_ads_ingestion_service(supabase_admin)

        # Exchange authorization code for tokens
        tokens = google_ads_service.exchange_authorization_code(code, redirect_uri)

        # Store connection
        connection = supabase_admin.table("connections").insert({
            "org_id": org_id,
            "source_type": "google_ads",
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_expiry": (datetime.utcnow() + timedelta(seconds=tokens["expires_in"])).isoformat(),
            "metadata": {
                "scope": tokens.get("scope", ""),
                "token_type": tokens.get("token_type", "Bearer")
            }
        }).execute()

        connection_id = connection.data[0]["id"]

        # Discover and store ad accounts
        accounts = ingestion_service.connect_google_ads_account(
            org_id=org_id,
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            connection_id=connection_id
        )

        logger.info(f"Google Ads connected for org {org_id}: {len(accounts)} accounts discovered")

        return {
            "success": True,
            "connection_id": connection_id,
            "accounts_discovered": len(accounts),
            "accounts": accounts
        }

    except Exception as e:
        log_error(e, context="Google Ads OAuth callback")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Meta Ads OAuth (Phase 6)
# ============================================================================

@router.get("/meta-ads/connect")
async def connect_meta_ads(
    org_id: str = Query(..., description="Organization ID"),
    redirect_uri: str = Query(..., description="OAuth callback URL")
):
    """
    Initiate Meta Ads (Facebook) OAuth 2.0 flow.

    Returns authorization URL for user to grant access.
    """
    try:
        supabase_admin = get_supabase_admin_client()
        meta_ads_service = get_meta_ads_service(supabase_admin)

        auth_url = meta_ads_service.get_oauth_authorization_url(org_id, redirect_uri)

        return {
            "authorization_url": auth_url,
            "org_id": org_id
        }

    except Exception as e:
        log_error(e, context="Meta Ads OAuth connect")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meta-ads/callback")
async def meta_ads_callback(
    code: str = Query(..., description="Authorization code"),
    org_id: str = Query(..., description="Organization ID"),
    redirect_uri: str = Query(..., description="Redirect URI used in authorization")
):
    """
    Handle Meta Ads OAuth callback.

    Exchanges authorization code for tokens and stores connection.
    """
    try:
        supabase_admin = get_supabase_admin_client()
        meta_ads_service = get_meta_ads_service(supabase_admin)
        ingestion_service = get_ads_ingestion_service(supabase_admin)

        # Exchange authorization code for tokens
        tokens = meta_ads_service.exchange_authorization_code(code, redirect_uri)

        # Exchange short-lived for long-lived token (60 days)
        long_lived_tokens = meta_ads_service.exchange_short_lived_for_long_lived_token(
            tokens["access_token"]
        )

        # Store connection
        connection = supabase_admin.table("connections").insert({
            "org_id": org_id,
            "source_type": "meta_ads",
            "access_token": long_lived_tokens["access_token"],
            "refresh_token": None,  # Meta doesn't use refresh tokens for long-lived tokens
            "token_expiry": (datetime.utcnow() + timedelta(seconds=long_lived_tokens["expires_in"])).isoformat(),
            "metadata": {
                "token_type": long_lived_tokens.get("token_type", "bearer"),
                "is_long_lived": True
            }
        }).execute()

        connection_id = connection.data[0]["id"]

        # Discover and store ad accounts
        accounts = ingestion_service.connect_meta_ads_account(
            org_id=org_id,
            access_token=long_lived_tokens["access_token"],
            connection_id=connection_id
        )

        logger.info(f"Meta Ads connected for org {org_id}: {len(accounts)} accounts discovered")

        return {
            "success": True,
            "connection_id": connection_id,
            "accounts_discovered": len(accounts),
            "accounts": accounts
        }

    except Exception as e:
        log_error(e, context="Meta Ads OAuth callback")
        raise HTTPException(status_code=500, detail=str(e))
