"""
ClickUp Connector
Abstraction layer for ClickUp integration
"""

from typing import Dict, Any, Optional
from supabase import Client
from app.services.clickup_service import ClickUpService
from app.services.ingestion_service import get_ingestion_service
from app.core.config import settings
from app.core.logging import logger


class ClickUpConnector:
    """ClickUp integration connector"""

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.clickup_service = ClickUpService(settings.CLICKUP_CLIENT_ID, settings.CLICKUP_CLIENT_SECRET)
        self.ingestion_service = get_ingestion_service(supabase)

    async def connect(self, code: str, org_id: str) -> Dict[str, Any]:
        """
        Connect ClickUp workspace using OAuth code.

        Args:
            code: OAuth authorization code
            org_id: Organization ID

        Returns:
            Connection result with workspace info
        """
        logger.info(f"[CLICKUP CONNECTOR] Connecting workspace for org {org_id}")

        # Exchange code for access token
        token_response = self.clickup_service.exchange_code(code)
        access_token = token_response.get("access_token")

        # Get user info
        user_info = self.clickup_service.get_authorized_user(access_token)

        # Store credentials
        self.supabase.table("integration_credentials").upsert({
            "org_id": org_id,
            "integration_type": "clickup",
            "credentials": {"access_token": access_token},
            "metadata": {
                "user": user_info,
                "token_response": token_response
            }
        }).execute()

        logger.info(f"[CLICKUP CONNECTOR] Workspace connected successfully")

        return {
            "user": user_info.get("user", {}),
            "access_token": access_token
        }

    async def sync(self, org_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Sync ClickUp tasks.

        Args:
            org_id: Organization ID
            force: Force full resync

        Returns:
            Sync statistics
        """
        logger.info(f"[CLICKUP CONNECTOR] Starting sync for org {org_id}")

        # Get credentials
        creds = self.supabase.table("integration_credentials")\
            .select("*")\
            .eq("org_id", org_id)\
            .eq("integration_type", "clickup")\
            .single()\
            .execute()

        if not creds.data:
            raise Exception("ClickUp not connected")

        access_token = creds.data["credentials"]["access_token"]

        # Fetch and sync tasks
        synced = 0
        failed = 0

        # Get workspaces
        workspaces = self.clickup_service.get_workspaces(access_token)

        for workspace in workspaces.get("teams", []):
            workspace_id = workspace["id"]

            # Get spaces
            spaces = self.clickup_service.get_spaces(access_token, workspace_id)

            for space in spaces.get("spaces", []):
                space_id = space["id"]

                # Get lists
                lists = self.clickup_service.get_lists(access_token, space_id)

                for list_item in lists.get("lists", []):
                    list_id = list_item["id"]

                    try:
                        # Get tasks
                        tasks = self.clickup_service.get_tasks(access_token, list_id)

                        for task in tasks.get("tasks", []):
                            try:
                                # Normalize and ingest task
                                document = self.clickup_service.normalize_task_to_document(
                                    task,
                                    org_id
                                )

                                # Save to database
                                doc_result = self.supabase.table("documents").upsert(document).execute()
                                doc_id = doc_result.data[0]["id"]

                                # Generate embeddings
                                await self.ingestion_service._generate_embeddings(
                                    document_id=doc_id,
                                    content=document["content"],
                                    org_id=org_id,
                                    metadata=document.get("metadata", {})
                                )

                                synced += 1
                            except Exception as e:
                                logger.error(f"Failed to ingest task {task.get('id')}: {e}")
                                failed += 1

                    except Exception as e:
                        logger.error(f"Failed to sync list {list_id}: {e}")
                        failed += 1

        # Setup webhook after successful sync
        if synced > 0:
            try:
                webhook_url = settings.CLICKUP_WEBHOOK_URL or f"{settings.BACKEND_URL}/api/v1/clickup/webhook"
                webhook_events = ["taskCreated", "taskUpdated", "taskDeleted", "taskCommentPosted"]

                for workspace in workspaces.get("teams", []):
                    self.clickup_service.create_webhook(
                        access_token=access_token,
                        team_id=workspace["id"],
                        endpoint=webhook_url,
                        events=webhook_events
                    )
            except Exception as e:
                logger.error(f"Failed to setup webhook: {e}")

        logger.info(f"[CLICKUP CONNECTOR] Sync complete: {synced} synced, {failed} failed")

        return {
            "synced_count": synced,
            "failed_count": failed
        }

    async def handle_webhook_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle ClickUp webhook event.

        Args:
            event_data: Webhook event payload
        """
        logger.info(f"[CLICKUP CONNECTOR] Handling webhook event: {event_data.get('event')}")

        event_type = event_data.get("event")

        if event_type in ["taskCreated", "taskUpdated"]:
            # Handle task creation/update
            task = event_data.get("task_id")
            # Re-sync specific task
            pass

    async def get_status(self, org_id: str) -> Dict[str, Any]:
        """
        Get ClickUp integration status.

        Args:
            org_id: Organization ID

        Returns:
            Status information
        """
        creds = self.supabase.table("integration_credentials")\
            .select("*")\
            .eq("org_id", org_id)\
            .eq("integration_type", "clickup")\
            .maybe_single()\
            .execute()

        if not creds.data:
            return {"connected": False}

        # Count documents
        docs = self.supabase.table("documents")\
            .select("id", count="exact")\
            .eq("org_id", org_id)\
            .eq("source_type", "clickup")\
            .execute()

        return {
            "connected": True,
            "workspace_name": creds.data.get("metadata", {}).get("user", {}).get("username"),
            "tasks_count": docs.count if hasattr(docs, 'count') else 0,
            "last_sync": creds.data.get("updated_at")
        }
