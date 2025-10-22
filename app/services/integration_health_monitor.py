"""
Integration Health Monitoring Service

Monitors the health and availability of external integrations:
- Slack API connectivity
- ClickUp API connectivity
- Email/SMTP connectivity
- OAuth token expiration tracking
- Connection failure detection and alerting

Features:
- Periodic health checks
- Failure detection and retry logic
- Health status tracking in database
- Automatic alerts on integration failures
- Integration status dashboard data
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from app.core.logging import logger, log_error
from app.core.config import settings


class IntegrationStatus(str, Enum):
    """Integration health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class IntegrationType(str, Enum):
    """Types of integrations to monitor"""
    SLACK = "slack"
    CLICKUP = "clickup"
    EMAIL = "email"
    OPENAI = "openai"
    GROQ = "groq"


class IntegrationHealthMonitor:
    """
    Monitors health and connectivity of external integrations.

    Tracks:
    - API connectivity and response times
    - OAuth token validity
    - Error rates
    - Last successful connection timestamps
    """

    def __init__(self, supabase):
        """
        Initialize integration health monitor.

        Args:
            supabase: Supabase client for tracking health status
        """
        self.supabase = supabase
        self._health_cache = {}  # Cache recent health checks
        logger.info("[INTEGRATION_HEALTH] Initialized")

    async def check_slack_health(self, org_id: str) -> Dict[str, Any]:
        """
        Check Slack integration health.

        Args:
            org_id: Organization ID

        Returns:
            Health check result
        """
        try:
            start_time = datetime.utcnow()

            # Get Slack credentials from database
            result = self.supabase.table("integrations")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("platform", "slack")\
                .eq("status", "active")\
                .execute()

            if not result.data:
                return {
                    "integration": "slack",
                    "status": IntegrationStatus.FAILED,
                    "message": "No active Slack integration found",
                    "checked_at": datetime.utcnow().isoformat()
                }

            integration = result.data[0]
            access_token = integration.get("access_token")

            if not access_token:
                return {
                    "integration": "slack",
                    "status": IntegrationStatus.FAILED,
                    "message": "No access token available",
                    "checked_at": datetime.utcnow().isoformat()
                }

            # Test Slack API connectivity
            import requests
            response = requests.get(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10
            )

            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            if response.status_code == 200:
                data = response.json()

                if data.get("ok"):
                    health_result = {
                        "integration": "slack",
                        "status": IntegrationStatus.HEALTHY,
                        "message": "Slack API responding normally",
                        "response_time_ms": response_time_ms,
                        "team_name": data.get("team"),
                        "user": data.get("user"),
                        "checked_at": datetime.utcnow().isoformat()
                    }
                else:
                    health_result = {
                        "integration": "slack",
                        "status": IntegrationStatus.FAILED,
                        "message": f"Slack API error: {data.get('error', 'Unknown error')}",
                        "response_time_ms": response_time_ms,
                        "checked_at": datetime.utcnow().isoformat()
                    }
            else:
                health_result = {
                    "integration": "slack",
                    "status": IntegrationStatus.FAILED,
                    "message": f"HTTP {response.status_code}",
                    "response_time_ms": response_time_ms,
                    "checked_at": datetime.utcnow().isoformat()
                }

            # Track health status
            await self._track_health_status(org_id, health_result)

            return health_result

        except Exception as e:
            log_error(e, context="IntegrationHealthMonitor.check_slack_health")

            health_result = {
                "integration": "slack",
                "status": IntegrationStatus.FAILED,
                "message": f"Health check failed: {str(e)}",
                "checked_at": datetime.utcnow().isoformat()
            }

            await self._track_health_status(org_id, health_result)
            return health_result

    async def check_clickup_health(self, org_id: str) -> Dict[str, Any]:
        """
        Check ClickUp integration health.

        Args:
            org_id: Organization ID

        Returns:
            Health check result
        """
        try:
            start_time = datetime.utcnow()

            # Get ClickUp credentials from database
            result = self.supabase.table("integrations")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("platform", "clickup")\
                .eq("status", "active")\
                .execute()

            if not result.data:
                return {
                    "integration": "clickup",
                    "status": IntegrationStatus.FAILED,
                    "message": "No active ClickUp integration found",
                    "checked_at": datetime.utcnow().isoformat()
                }

            integration = result.data[0]
            access_token = integration.get("access_token")

            if not access_token:
                return {
                    "integration": "clickup",
                    "status": IntegrationStatus.FAILED,
                    "message": "No access token available",
                    "checked_at": datetime.utcnow().isoformat()
                }

            # Test ClickUp API connectivity
            import requests
            response = requests.get(
                "https://api.clickup.com/api/v2/user",
                headers={"Authorization": access_token},
                timeout=10
            )

            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            if response.status_code == 200:
                data = response.json()

                health_result = {
                    "integration": "clickup",
                    "status": IntegrationStatus.HEALTHY,
                    "message": "ClickUp API responding normally",
                    "response_time_ms": response_time_ms,
                    "user": data.get("user", {}).get("username"),
                    "checked_at": datetime.utcnow().isoformat()
                }
            else:
                health_result = {
                    "integration": "clickup",
                    "status": IntegrationStatus.FAILED,
                    "message": f"HTTP {response.status_code}",
                    "response_time_ms": response_time_ms,
                    "checked_at": datetime.utcnow().isoformat()
                }

            # Track health status
            await self._track_health_status(org_id, health_result)

            return health_result

        except Exception as e:
            log_error(e, context="IntegrationHealthMonitor.check_clickup_health")

            health_result = {
                "integration": "clickup",
                "status": IntegrationStatus.FAILED,
                "message": f"Health check failed: {str(e)}",
                "checked_at": datetime.utcnow().isoformat()
            }

            await self._track_health_status(org_id, health_result)
            return health_result

    async def check_email_health(self, org_id: str) -> Dict[str, Any]:
        """
        Check Email/SMTP integration health.

        Args:
            org_id: Organization ID

        Returns:
            Health check result
        """
        try:
            start_time = datetime.utcnow()

            # Check SMTP configuration
            if not settings.SMTP_HOST or not settings.SMTP_USERNAME:
                return {
                    "integration": "email",
                    "status": IntegrationStatus.FAILED,
                    "message": "SMTP not configured",
                    "checked_at": datetime.utcnow().isoformat()
                }

            # Test SMTP connectivity
            import smtplib
            import ssl

            context = ssl.create_default_context()

            try:
                if settings.SMTP_USE_SSL:
                    with smtplib.SMTP_SSL(
                        settings.SMTP_HOST,
                        settings.SMTP_PORT,
                        context=context,
                        timeout=10
                    ) as server:
                        server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                else:
                    with smtplib.SMTP(
                        settings.SMTP_HOST,
                        settings.SMTP_PORT,
                        timeout=10
                    ) as server:
                        if settings.SMTP_USE_TLS:
                            server.starttls(context=context)
                        server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)

                response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                health_result = {
                    "integration": "email",
                    "status": IntegrationStatus.HEALTHY,
                    "message": "SMTP server responding normally",
                    "response_time_ms": response_time_ms,
                    "smtp_host": settings.SMTP_HOST,
                    "checked_at": datetime.utcnow().isoformat()
                }

            except Exception as smtp_error:
                response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                health_result = {
                    "integration": "email",
                    "status": IntegrationStatus.FAILED,
                    "message": f"SMTP connection failed: {str(smtp_error)}",
                    "response_time_ms": response_time_ms,
                    "checked_at": datetime.utcnow().isoformat()
                }

            # Track health status
            await self._track_health_status(org_id, health_result)

            return health_result

        except Exception as e:
            log_error(e, context="IntegrationHealthMonitor.check_email_health")

            health_result = {
                "integration": "email",
                "status": IntegrationStatus.FAILED,
                "message": f"Health check failed: {str(e)}",
                "checked_at": datetime.utcnow().isoformat()
            }

            await self._track_health_status(org_id, health_result)
            return health_result

    async def check_all_integrations(self, org_id: str) -> Dict[str, Any]:
        """
        Check health of all integrations for an organization.

        Args:
            org_id: Organization ID

        Returns:
            Combined health status for all integrations
        """
        results = await asyncio.gather(
            self.check_slack_health(org_id),
            self.check_clickup_health(org_id),
            self.check_email_health(org_id),
            return_exceptions=True
        )

        # Process results
        health_summary = {
            "org_id": org_id,
            "checked_at": datetime.utcnow().isoformat(),
            "integrations": {},
            "overall_status": IntegrationStatus.HEALTHY
        }

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[INTEGRATION_HEALTH] Health check exception: {result}")
                continue

            integration_name = result.get("integration")
            health_summary["integrations"][integration_name] = result

            # Determine overall status (worst status wins)
            if result.get("status") == IntegrationStatus.FAILED:
                health_summary["overall_status"] = IntegrationStatus.FAILED
            elif result.get("status") == IntegrationStatus.DEGRADED and \
                 health_summary["overall_status"] != IntegrationStatus.FAILED:
                health_summary["overall_status"] = IntegrationStatus.DEGRADED

        logger.info(
            f"[INTEGRATION_HEALTH] Overall status: {health_summary['overall_status']}",
            extra={"org_id": org_id}
        )

        return health_summary

    async def _track_health_status(self, org_id: str, health_result: Dict[str, Any]):
        """
        Track integration health status in database.

        Args:
            org_id: Organization ID
            health_result: Health check result to track
        """
        try:
            health_record = {
                "org_id": org_id,
                "integration": health_result.get("integration"),
                "status": health_result.get("status"),
                "message": health_result.get("message"),
                "response_time_ms": health_result.get("response_time_ms"),
                "metadata": {
                    k: v for k, v in health_result.items()
                    if k not in ["integration", "status", "message", "response_time_ms"]
                },
                "checked_at": health_result.get("checked_at")
            }

            self.supabase.table("integration_health_checks")\
                .insert(health_record)\
                .execute()

        except Exception as e:
            log_error(e, context="IntegrationHealthMonitor._track_health_status")

    async def get_integration_status(
        self,
        org_id: str,
        integration_type: Optional[IntegrationType] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent health status for integrations.

        Args:
            org_id: Organization ID
            integration_type: Optional filter for specific integration

        Returns:
            List of recent health checks
        """
        try:
            query = self.supabase.table("integration_health_checks")\
                .select("*")\
                .eq("org_id", org_id)\
                .order("checked_at", desc=True)\
                .limit(50)

            if integration_type:
                query = query.eq("integration", integration_type.value)

            result = query.execute()

            return result.data if result.data else []

        except Exception as e:
            log_error(e, context="IntegrationHealthMonitor.get_integration_status")
            return []

    async def check_token_expiration(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Check for OAuth tokens that are expiring soon.

        Args:
            org_id: Organization ID

        Returns:
            List of integrations with expiring tokens
        """
        try:
            # Check tokens expiring within 7 days
            expiration_threshold = (datetime.utcnow() + timedelta(days=7)).isoformat()

            result = self.supabase.table("integrations")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("status", "active")\
                .not_.is_("token_expires_at", "null")\
                .lte("token_expires_at", expiration_threshold)\
                .execute()

            expiring_tokens = result.data if result.data else []

            if expiring_tokens:
                logger.warning(
                    f"[INTEGRATION_HEALTH] {len(expiring_tokens)} tokens expiring soon",
                    extra={"org_id": org_id}
                )

            return expiring_tokens

        except Exception as e:
            log_error(e, context="IntegrationHealthMonitor.check_token_expiration")
            return []


# Global instance (lazy initialization)
_integration_health_monitor_instance = None


def get_integration_health_monitor(supabase) -> IntegrationHealthMonitor:
    """
    Get or create the global IntegrationHealthMonitor instance.

    Args:
        supabase: Supabase client

    Returns:
        IntegrationHealthMonitor instance
    """
    global _integration_health_monitor_instance

    if _integration_health_monitor_instance is None:
        _integration_health_monitor_instance = IntegrationHealthMonitor(supabase)

    return _integration_health_monitor_instance
