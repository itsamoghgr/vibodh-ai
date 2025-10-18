"""
ClickUp Integration API Routes
"""

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import RedirectResponse
from uuid import UUID
import os

from app.db import supabase
from app.core.config import settings
from app.core.logging import logger, log_error
from app.services.clickup_service import get_clickup_service
from app.services.ingestion_service import get_ingestion_service

router = APIRouter(prefix="/clickup", tags=["ClickUp"])


@router.get("/connect")
async def clickup_connect(org_id: str = Query(...)):
    """
    Initiate ClickUp OAuth 2.0 flow
    Redirects user to ClickUp authorization page.
    """
    try:
        clickup_service = get_clickup_service(supabase)

        # Generate state parameter (org_id for simplicity, should use JWT in production)
        state = org_id

        # Get authorization URL
        auth_url = clickup_service.get_authorization_url(state)

        return RedirectResponse(url=auth_url)

    except Exception as e:
        log_error(e, context="ClickUp OAuth start")
        raise HTTPException(status_code=500, detail=f"Failed to initiate ClickUp OAuth: {str(e)}")


@router.get("/callback")
async def clickup_callback(
    code: str = Query(...),
    state: str = Query(...)
):
    """
    ClickUp OAuth callback endpoint
    Exchanges authorization code for access token and stores in connections table.
    """
    try:
        clickup_service = get_clickup_service(supabase)

        org_id = state  # Extract org_id from state parameter

        # Exchange code for access token
        token_data = clickup_service.exchange_code_for_token(code)
        access_token = token_data.get("access_token")

        if not access_token:
            raise HTTPException(status_code=400, detail="Failed to obtain access token")

        # Get authorized user info
        user_info = clickup_service.get_authorized_user(access_token)

        # Get user's workspaces
        workspaces = clickup_service.get_workspaces(access_token)

        if not workspaces:
            raise HTTPException(status_code=400, detail="No ClickUp workspaces found")

        # Use first workspace (or let user select in production)
        workspace = workspaces[0]
        workspace_id = workspace.get("id")
        workspace_name = workspace.get("name")

        # Store connection in Supabase
        connection_data = {
            "org_id": org_id,
            "source_type": "clickup",
            "access_token": access_token,
            "workspace_id": workspace_id,
            "workspace_name": workspace_name,
            "metadata": {
                "user_id": user_info.get("user", {}).get("id"),
                "user_email": user_info.get("user", {}).get("email"),
                "workspace": workspace
            }
        }

        result = supabase.table("connections").insert(connection_data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to store connection")

        # Redirect to frontend integrations page with success message
        frontend_url = settings.BACKEND_URL.replace('http://localhost:8000', 'http://localhost:3000')
        return RedirectResponse(url=f"{frontend_url}/dashboard/integrations?clickup=connected")

    except Exception as e:
        # Redirect to frontend with error
        frontend_url = settings.BACKEND_URL.replace('http://localhost:8000', 'http://localhost:3000')
        return RedirectResponse(url=f"{frontend_url}/dashboard/integrations?clickup=error&message={str(e)}")


@router.post("/sync/{connection_id}")
async def clickup_sync(connection_id: str, org_id: str = Query(...)):
    """
    Manually trigger ClickUp data sync
    Fetches all tasks and comments from connected ClickUp workspace and ingests them.
    """
    try:
        clickup_service = get_clickup_service(supabase)
        ingestion_service = get_ingestion_service(supabase)

        # Fetch all tasks
        logger.info(f"Fetching all tasks from ClickUp for connection {connection_id}")
        tasks_data = clickup_service.fetch_all_tasks(connection_id, org_id)
        logger.info(f"Fetched {len(tasks_data)} tasks")

        # Normalize and ingest each task
        documents_ingested = 0
        for idx, task_data in enumerate(tasks_data):
            try:
                logger.info(f"Processing task {idx+1}/{len(tasks_data)}")
                document = clickup_service.normalize_task_to_document(task_data, org_id, connection_id)

                # Ingest the document
                await ingestion_service.ingest_document(document)
                documents_ingested += 1
            except Exception as e:
                logger.error(f"Error ingesting task {idx+1}: {e}", exc_info=True)
                continue

        return {
            "success": True,
            "message": f"Synced {documents_ingested}/{len(tasks_data)} tasks from ClickUp",
            "tasks_fetched": len(tasks_data),
            "documents_ingested": documents_ingested
        }

    except Exception as e:
        log_error(e, context="ClickUp sync")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook")
async def clickup_webhook(request: Request):
    """
    Handle ClickUp webhook events for real-time sync.
    """
    try:
        body = await request.json()
        logger.info(f"ClickUp webhook event: {body.get('event')}")

        # Process webhook event
        event_type = body.get("event")
        logger.info(f"ClickUp event type: {event_type}")

        return {"ok": True}

    except Exception as e:
        log_error(e, context="ClickUp webhook")
        raise HTTPException(status_code=500, detail=str(e))
