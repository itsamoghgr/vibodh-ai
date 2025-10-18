"""
Slack Integration API Routes
"""

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import RedirectResponse
from uuid import UUID
from typing import Optional
from datetime import datetime

from app.db import supabase
from app.core.config import settings
from app.core.logging import logger, log_error
from app.services.ingestion_service import get_ingestion_service
from app.models.legacy_schemas import SlackIngestRequest

# Import from root for legacy connectors
import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from connectors.slack_connector import get_slack_connector

router = APIRouter(prefix="/slack", tags=["Slack"])


@router.get("/connect")
async def start_slack_oauth(org_id: str = Query(...)):
    """
    Start Slack OAuth flow
    Redirects user to Slack authorization page
    """
    try:
        slack_connector = get_slack_connector()

        # Generate state for CSRF protection
        state = f"{org_id}:{datetime.utcnow().timestamp()}"

        # Get authorization URL
        auth_url = slack_connector.get_authorization_url(state=state)

        return RedirectResponse(url=auth_url)
    except Exception as e:
        log_error(e, context="Slack OAuth start")
        raise HTTPException(status_code=500, detail=f"Failed to start OAuth: {str(e)}")


@router.get("/connect/callback")
async def slack_oauth_callback(code: str, state: str = None):
    """
    Handle Slack OAuth callback
    Exchange code for access token and save connection
    """
    try:
        slack_connector = get_slack_connector()

        # Extract org_id from state
        if not state:
            raise HTTPException(status_code=400, detail="Missing state parameter")

        org_id = state.split(":")[0] if state else None
        if not org_id or org_id.strip() == '':
            raise HTTPException(status_code=400, detail=f"Invalid or empty org_id in state: {state}")

        # Exchange code for token
        token_data = slack_connector.exchange_code_for_token(code)

        # Debug: Print scopes
        logger.info(f"Slack OAuth scopes received: {token_data.get('scope', 'N/A')}")

        # Get workspace info
        workspace_info = slack_connector.get_workspace_info(token_data["access_token"])

        # Save connection to database
        connection_data = {
            "org_id": org_id,
            "source_type": "slack",
            "status": "active",
            "access_token": token_data["access_token"],
            "workspace_name": workspace_info["name"],
            "workspace_id": workspace_info["id"],
            "metadata": {
                "domain": workspace_info.get("domain"),
                "team_id": token_data["team_id"]
            }
        }

        # Insert or update connection
        existing = supabase.table("connections")\
            .select("id")\
            .eq("org_id", org_id)\
            .eq("source_type", "slack")\
            .execute()

        if existing.data:
            # Update existing
            supabase.table("connections")\
                .update(connection_data)\
                .eq("id", existing.data[0]["id"])\
                .execute()
        else:
            # Insert new
            supabase.table("connections").insert(connection_data).execute()

        # Redirect back to frontend integrations page
        return RedirectResponse(url=f"{settings.BACKEND_URL.replace('http://localhost:8000', 'http://localhost:3000')}/dashboard/integrations?slack=connected")

    except Exception as e:
        logger.error(f"OAuth callback error: {e}", exc_info=True)
        return RedirectResponse(url=f"{settings.BACKEND_URL.replace('http://localhost:8000', 'http://localhost:3000')}/dashboard/integrations?error={str(e)}")


@router.post("/ingest")
async def ingest_slack(request: SlackIngestRequest):
    """
    Ingest messages from Slack
    Fetches messages, creates documents, and generates embeddings
    """
    try:
        logger.info(f"Starting Slack ingestion for org_id={request.org_id}, connection_id={request.connection_id}")
        ingestion_service = get_ingestion_service(supabase)

        result = await ingestion_service.ingest_slack(
            org_id=request.org_id,
            connection_id=request.connection_id,
            channel_ids=request.channel_ids,
            days_back=request.days_back
        )

        return {
            "success": True,
            "message": f"Ingested {result['documents_created']} messages from Slack",
            "details": result
        }

    except Exception as e:
        log_error(e, context="Slack ingestion")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events")
async def slack_webhook(request: Request):
    """
    Handle Slack webhook events for real-time sync.
    """
    try:
        body = await request.json()
        logger.info(f"Slack webhook event: {body.get('type')}")

        # Handle URL verification
        if body.get("type") == "url_verification":
            return {"challenge": body.get("challenge")}

        # Handle event callbacks
        if body.get("type") == "event_callback":
            # Process webhook event
            event = body.get("event", {})
            logger.info(f"Slack event type: {event.get('type')}")
            return {"ok": True}

        return {"ok": True}

    except Exception as e:
        log_error(e, context="Slack webhook")
        raise HTTPException(status_code=500, detail=str(e))
