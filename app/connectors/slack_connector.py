"""
Slack Connector
Abstraction layer for Slack integration
"""

from typing import Dict, Any, Optional
from supabase import Client
from app.services.slack_service import SlackService
from app.services.ingestion_service import get_ingestion_service
from app.core.config import settings
from app.core.logging import logger


class SlackConnector:
    """Slack integration connector"""

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.slack_service = SlackService(settings.SLACK_CLIENT_ID, settings.SLACK_CLIENT_SECRET)
        self.ingestion_service = get_ingestion_service(supabase)

    async def connect(self, code: str, org_id: str) -> Dict[str, Any]:
        """
        Connect Slack workspace using OAuth code.

        Args:
            code: OAuth authorization code
            org_id: Organization ID

        Returns:
            Connection result with workspace info
        """
        logger.info(f"[SLACK CONNECTOR] Connecting workspace for org {org_id}")

        # Exchange code for access token
        token_response = self.slack_service.exchange_code(code, settings.SLACK_REDIRECT_URI)
        access_token = token_response.get("access_token")

        # Store credentials
        self.supabase.table("integration_credentials").upsert({
            "org_id": org_id,
            "integration_type": "slack",
            "credentials": {"access_token": access_token},
            "metadata": token_response
        }).execute()

        logger.info(f"[SLACK CONNECTOR] Workspace connected successfully")

        return {
            "workspace": token_response.get("team", {}).get("name"),
            "access_token": access_token
        }

    async def sync(self, org_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Sync Slack messages.

        Args:
            org_id: Organization ID
            force: Force full resync

        Returns:
            Sync statistics
        """
        logger.info(f"[SLACK CONNECTOR] Starting sync for org {org_id}")

        # Get credentials
        creds = self.supabase.table("integration_credentials")\
            .select("*")\
            .eq("org_id", org_id)\
            .eq("integration_type", "slack")\
            .single()\
            .execute()

        if not creds.data:
            raise Exception("Slack not connected")

        access_token = creds.data["credentials"]["access_token"]

        # Fetch and sync messages
        synced = 0
        failed = 0

        channels = self.slack_service.get_channels(access_token)

        for channel in channels:
            try:
                messages = self.slack_service.get_channel_history(
                    access_token,
                    channel["id"]
                )

                for msg in messages:
                    try:
                        # Ingest message
                        await self.ingestion_service.ingest_slack_message(
                            message=msg,
                            channel=channel,
                            org_id=org_id
                        )
                        synced += 1
                    except Exception as e:
                        logger.error(f"Failed to ingest message: {e}")
                        failed += 1

            except Exception as e:
                logger.error(f"Failed to sync channel {channel['name']}: {e}")
                failed += 1

        logger.info(f"[SLACK CONNECTOR] Sync complete: {synced} synced, {failed} failed")

        return {
            "synced_count": synced,
            "failed_count": failed
        }

    async def handle_webhook_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle Slack webhook event.

        Args:
            event_data: Webhook event payload
        """
        logger.info(f"[SLACK CONNECTOR] Handling webhook event")

        event = event_data.get("event", {})
        event_type = event.get("type")

        if event_type == "message":
            # Handle new message
            org_id = event_data.get("team_id")  # Need to map team_id to org_id
            await self.ingestion_service.ingest_slack_message(
                message=event,
                channel={"id": event.get("channel")},
                org_id=org_id
            )

    async def get_status(self, org_id: str) -> Dict[str, Any]:
        """
        Get Slack integration status.

        Args:
            org_id: Organization ID

        Returns:
            Status information
        """
        creds = self.supabase.table("integration_credentials")\
            .select("*")\
            .eq("org_id", org_id)\
            .eq("integration_type", "slack")\
            .maybe_single()\
            .execute()

        if not creds.data:
            return {"connected": False}

        # Count documents
        docs = self.supabase.table("documents")\
            .select("id", count="exact")\
            .eq("org_id", org_id)\
            .eq("source_type", "slack")\
            .execute()

        return {
            "connected": True,
            "workspace_name": creds.data.get("metadata", {}).get("team", {}).get("name"),
            "channels_count": docs.count if hasattr(docs, 'count') else 0,
            "last_sync": creds.data.get("updated_at")
        }
