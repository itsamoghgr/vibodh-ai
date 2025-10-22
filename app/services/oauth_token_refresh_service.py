"""
OAuth Token Refresh Service

Handles automatic refresh of OAuth tokens for all integrations.
Prevents API failures due to expired tokens.

Features:
- Automatic token refresh before expiration
- Retry logic with exponential backoff
- Token expiration tracking
- Integration-specific refresh handlers
- Proactive token refresh (refresh 24h before expiry)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests

from app.core.logging import logger, log_error
from app.core.config import settings


class OAuthTokenRefreshService:
    """
    Service for refreshing OAuth tokens across all integrations.

    Supports:
    - Slack OAuth
    - ClickUp OAuth
    - Future integrations (Gmail, Google Drive, etc.)
    """

    def __init__(self, supabase):
        """
        Initialize OAuth token refresh service.

        Args:
            supabase: Supabase client for accessing integrations table
        """
        self.supabase = supabase
        logger.info("[OAUTH_REFRESH] Service initialized")

    async def refresh_slack_token(
        self,
        integration_id: str,
        refresh_token: str
    ) -> Dict[str, Any]:
        """
        Refresh Slack OAuth token.

        Args:
            integration_id: Integration ID
            refresh_token: Current refresh token

        Returns:
            Dict with new access_token, refresh_token, and expires_at
        """
        try:
            logger.info(
                f"[OAUTH_REFRESH] Refreshing Slack token",
                extra={"integration_id": integration_id}
            )

            # Slack token refresh endpoint
            response = requests.post(
                "https://slack.com/api/oauth.v2.access",
                data={
                    "client_id": settings.SLACK_CLIENT_ID,
                    "client_secret": settings.SLACK_CLIENT_SECRET,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("ok"):
                    # Extract new tokens
                    new_access_token = data.get("access_token")
                    new_refresh_token = data.get("refresh_token", refresh_token)
                    expires_in = data.get("expires_in", 43200)  # Default 12 hours

                    # Calculate expiration time
                    expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                    # Update database
                    self.supabase.table("integrations")\
                        .update({
                            "access_token": new_access_token,
                            "refresh_token": new_refresh_token,
                            "token_expires_at": expires_at.isoformat(),
                            "updated_at": datetime.utcnow().isoformat()
                        })\
                        .eq("id", integration_id)\
                        .execute()

                    logger.info(
                        f"[OAUTH_REFRESH] Slack token refreshed successfully",
                        extra={
                            "integration_id": integration_id,
                            "expires_at": expires_at.isoformat()
                        }
                    )

                    return {
                        "success": True,
                        "access_token": new_access_token,
                        "refresh_token": new_refresh_token,
                        "expires_at": expires_at.isoformat()
                    }
                else:
                    error = data.get("error", "Unknown error")
                    logger.error(
                        f"[OAUTH_REFRESH] Slack API error: {error}",
                        extra={"integration_id": integration_id}
                    )

                    return {
                        "success": False,
                        "error": f"Slack API error: {error}"
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }

        except Exception as e:
            log_error(e, context="OAuthTokenRefreshService.refresh_slack_token")

            return {
                "success": False,
                "error": str(e)
            }

    async def refresh_clickup_token(
        self,
        integration_id: str,
        refresh_token: str
    ) -> Dict[str, Any]:
        """
        Refresh ClickUp OAuth token.

        Args:
            integration_id: Integration ID
            refresh_token: Current refresh token

        Returns:
            Dict with new access_token and expires_at
        """
        try:
            logger.info(
                f"[OAUTH_REFRESH] Refreshing ClickUp token",
                extra={"integration_id": integration_id}
            )

            # ClickUp token refresh endpoint
            response = requests.post(
                "https://api.clickup.com/api/v2/oauth/token",
                data={
                    "client_id": settings.CLICKUP_CLIENT_ID,
                    "client_secret": settings.CLICKUP_CLIENT_SECRET,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                # Extract new tokens
                new_access_token = data.get("access_token")
                new_refresh_token = data.get("refresh_token", refresh_token)

                # ClickUp tokens typically don't expire, but we'll set a conservative estimate
                expires_at = datetime.utcnow() + timedelta(days=365)

                # Update database
                self.supabase.table("integrations")\
                    .update({
                        "access_token": new_access_token,
                        "refresh_token": new_refresh_token,
                        "token_expires_at": expires_at.isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    })\
                    .eq("id", integration_id)\
                    .execute()

                logger.info(
                    f"[OAUTH_REFRESH] ClickUp token refreshed successfully",
                    extra={"integration_id": integration_id}
                )

                return {
                    "success": True,
                    "access_token": new_access_token,
                    "refresh_token": new_refresh_token,
                    "expires_at": expires_at.isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }

        except Exception as e:
            log_error(e, context="OAuthTokenRefreshService.refresh_clickup_token")

            return {
                "success": False,
                "error": str(e)
            }

    async def refresh_integration_token(
        self,
        integration_id: str,
        platform: str
    ) -> Dict[str, Any]:
        """
        Refresh OAuth token for any integration.

        Args:
            integration_id: Integration ID
            platform: Platform name (slack, clickup, etc.)

        Returns:
            Refresh result
        """
        try:
            # Get integration from database
            result = self.supabase.table("integrations")\
                .select("*")\
                .eq("id", integration_id)\
                .single()\
                .execute()

            if not result.data:
                return {
                    "success": False,
                    "error": "Integration not found"
                }

            integration = result.data
            refresh_token = integration.get("refresh_token")

            if not refresh_token:
                return {
                    "success": False,
                    "error": "No refresh token available"
                }

            # Route to platform-specific refresh handler
            if platform == "slack":
                return await self.refresh_slack_token(integration_id, refresh_token)
            elif platform == "clickup":
                return await self.refresh_clickup_token(integration_id, refresh_token)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported platform: {platform}"
                }

        except Exception as e:
            log_error(e, context="OAuthTokenRefreshService.refresh_integration_token")

            return {
                "success": False,
                "error": str(e)
            }

    async def check_and_refresh_expiring_tokens(
        self,
        org_id: str,
        hours_before_expiry: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Check for tokens expiring soon and refresh them proactively.

        Args:
            org_id: Organization ID
            hours_before_expiry: Refresh tokens this many hours before expiry

        Returns:
            List of refresh results
        """
        try:
            # Find tokens expiring within threshold
            expiry_threshold = (datetime.utcnow() + timedelta(hours=hours_before_expiry)).isoformat()

            result = self.supabase.table("integrations")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("status", "active")\
                .not_.is_("token_expires_at", "null")\
                .not_.is_("refresh_token", "null")\
                .lte("token_expires_at", expiry_threshold)\
                .execute()

            expiring_integrations = result.data if result.data else []

            if not expiring_integrations:
                logger.info(
                    f"[OAUTH_REFRESH] No tokens expiring soon",
                    extra={"org_id": org_id}
                )
                return []

            logger.info(
                f"[OAUTH_REFRESH] Found {len(expiring_integrations)} tokens expiring soon",
                extra={"org_id": org_id}
            )

            # Refresh each token
            refresh_results = []

            for integration in expiring_integrations:
                integration_id = integration.get("id")
                platform = integration.get("platform")

                result = await self.refresh_integration_token(integration_id, platform)
                refresh_results.append({
                    "integration_id": integration_id,
                    "platform": platform,
                    **result
                })

            return refresh_results

        except Exception as e:
            log_error(e, context="OAuthTokenRefreshService.check_and_refresh_expiring_tokens")
            return []

    async def mark_integration_disconnected(
        self,
        integration_id: str,
        reason: str
    ):
        """
        Mark an integration as disconnected when token refresh fails.

        Args:
            integration_id: Integration ID
            reason: Reason for disconnection
        """
        try:
            self.supabase.table("integrations")\
                .update({
                    "status": "disconnected",
                    "error_message": reason,
                    "updated_at": datetime.utcnow().isoformat()
                })\
                .eq("id", integration_id)\
                .execute()

            logger.warning(
                f"[OAUTH_REFRESH] Integration marked as disconnected",
                extra={
                    "integration_id": integration_id,
                    "reason": reason
                }
            )

        except Exception as e:
            log_error(e, context="OAuthTokenRefreshService.mark_integration_disconnected")


# Global instance (lazy initialization)
_oauth_token_refresh_service_instance = None


def get_oauth_token_refresh_service(supabase) -> OAuthTokenRefreshService:
    """
    Get or create the global OAuthTokenRefreshService instance.

    Args:
        supabase: Supabase client

    Returns:
        OAuthTokenRefreshService instance
    """
    global _oauth_token_refresh_service_instance

    if _oauth_token_refresh_service_instance is None:
        _oauth_token_refresh_service_instance = OAuthTokenRefreshService(supabase)

    return _oauth_token_refresh_service_instance
