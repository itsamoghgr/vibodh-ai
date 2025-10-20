"""
Slack Connector
Abstraction layer for Slack integration
"""

from typing import Dict, Any, Optional
from supabase import Client
from app.services.slack_service import SlackService
from app.core.config import settings
from app.core.logging import logger


class SlackConnector:
    """Slack integration connector"""

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.slack_service = SlackService(settings.SLACK_CLIENT_ID, settings.SLACK_CLIENT_SECRET)
        # Ingestion service will be injected when needed to avoid circular imports
        self.ingestion_service = None

    def _get_ingestion_service(self):
        """Lazy load ingestion service to avoid circular imports"""
        if self.ingestion_service is None:
            from app.services.ingestion_service import get_ingestion_service
            self.ingestion_service = get_ingestion_service(self.supabase)
        return self.ingestion_service

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

        # Get credentials and connection
        creds = self.supabase.table("integration_credentials")\
            .select("*")\
            .eq("org_id", org_id)\
            .eq("integration_type", "slack")\
            .single()\
            .execute()

        if not creds.data:
            raise Exception("Slack not connected")

        # Get connection_id
        connection = self.supabase.table("connections")\
            .select("id")\
            .eq("org_id", org_id)\
            .eq("source_type", "slack")\
            .single()\
            .execute()

        if not connection.data:
            raise Exception("Slack connection not found")

        connection_id = connection.data["id"]
        access_token = creds.data["credentials"]["access_token"]

        # Use the ingestion service's ingest_slack method for bulk sync
        ingestion_service = self._get_ingestion_service()
        result = await ingestion_service.ingest_slack(
            org_id=org_id,
            connection_id=connection_id,
            channel_ids=None,  # All channels
            days_back=3650  # All history
        )

        logger.info(f"[SLACK CONNECTOR] Sync complete: {result['documents_created']} synced, {result.get('documents_skipped', 0)} skipped")

        return {
            "synced_count": result["documents_created"],
            "failed_count": 0
        }

    async def handle_webhook_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle Slack webhook event.

        Args:
            event_data: Webhook event payload (should include org_id)
        """
        logger.info(f"[SLACK CONNECTOR] Handling webhook event")

        event = event_data.get("event", {})
        event_type = event.get("type")
        org_id = event_data.get("org_id")

        if not org_id:
            logger.error("[SLACK CONNECTOR] No org_id provided in event data")
            return

        if event_type == "message":
            # Skip bot messages and message changes to avoid loops
            if event.get("subtype") in ["bot_message", "message_changed", "message_deleted"]:
                logger.info(f"[SLACK CONNECTOR] Skipping {event.get('subtype')} event")
                return

            # Get connection and credentials
            connection = self.supabase.table("connections")\
                .select("id, access_token")\
                .eq("org_id", org_id)\
                .eq("source_type", "slack")\
                .single()\
                .execute()

            if not connection.data:
                logger.error(f"[SLACK CONNECTOR] No Slack connection found for org {org_id}")
                return

            connection_id = connection.data["id"]
            access_token = connection.data["access_token"]

            # Handle new message using the correct method
            logger.info(f"[SLACK CONNECTOR] Processing message from channel {event.get('channel')}")
            ingestion_service = self._get_ingestion_service()
            await ingestion_service.handle_slack_event(
                event=event,
                org_id=org_id,
                connection_id=connection_id,
                access_token=access_token
            )
            logger.info(f"[SLACK CONNECTOR] Successfully ingested message for org {org_id}")

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

    # Wrapper methods to expose SlackService functionality
    def list_channels(self, access_token: str, types: str = "public_channel", auto_join: bool = True):
        """List channels - wrapper for slack_service.list_channels"""
        return self.slack_service.list_channels(access_token, types, auto_join)

    def fetch_messages(self, access_token: str, channel_id: str, days_back: int = 30):
        """Fetch messages - wrapper for slack_service.fetch_messages"""
        return self.slack_service.fetch_messages(access_token, channel_id, days_back)

    def get_user_info(self, access_token: str, user_id: str):
        """Get user info - wrapper for slack_service.get_user_info"""
        return self.slack_service.get_user_info(access_token, user_id)

    def fetch_thread_replies(self, access_token: str, channel_id: str, thread_ts: str):
        """Fetch thread replies - wrapper for slack_service.fetch_thread_replies"""
        return self.slack_service.fetch_thread_replies(access_token, channel_id, thread_ts)
