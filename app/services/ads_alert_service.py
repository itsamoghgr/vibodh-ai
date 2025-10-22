"""
Ads Alert Service - Phase 6.5
Monitors ad performance and generates alerts for anomalies

Alert Types:
- ROAS Drop: >20% decrease day-over-day
- Budget Overage: Daily spend exceeds configured limit
- Ingestion Failure: Handled by ads_worker.py
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.core.logging import logger
from app.db import get_supabase_admin_client


class AdsAlertService:
    """
    Service for detecting and managing ads performance alerts.
    """

    def __init__(self):
        self.supabase = get_supabase_admin_client()

    async def detect_roas_drops(
        self,
        org_id: str,
        threshold_pct: float = 20.0
    ) -> List[Dict[str, Any]]:
        """
        Detect campaigns with significant ROAS drops.

        Args:
            org_id: Organization ID
            threshold_pct: Alert if ROAS drops by this percentage (default 20%)

        Returns:
            List of alerts created
        """
        try:
            alerts_created = []

            # Get yesterday's and day-before-yesterday's metrics for comparison
            yesterday = (datetime.utcnow() - timedelta(days=1)).date()
            day_before = (datetime.utcnow() - timedelta(days=2)).date()

            # Fetch recent metrics grouped by campaign
            metrics_query = self.supabase.table('ad_metrics')\
                .select('campaign_id, metric_date, roas')\
                .eq('org_id', org_id)\
                .in_('metric_date', [yesterday.isoformat(), day_before.isoformat()])\
                .execute()

            metrics = metrics_query.data or []

            # Group by campaign
            campaign_metrics = {}
            for metric in metrics:
                campaign_id = metric['campaign_id']
                if campaign_id not in campaign_metrics:
                    campaign_metrics[campaign_id] = {}

                date = metric['metric_date']
                campaign_metrics[campaign_id][date] = metric.get('roas', 0)

            # Detect drops
            for campaign_id, dates in campaign_metrics.items():
                yesterday_str = yesterday.isoformat()
                day_before_str = day_before.isoformat()

                if yesterday_str not in dates or day_before_str not in dates:
                    continue  # Missing data

                roas_yesterday = float(dates[yesterday_str] or 0)
                roas_day_before = float(dates[day_before_str] or 0)

                if roas_day_before == 0:
                    continue  # Avoid division by zero

                # Calculate percentage drop
                pct_change = ((roas_yesterday - roas_day_before) / roas_day_before) * 100

                if pct_change <= -threshold_pct:  # Negative = drop
                    # Get campaign details
                    campaign = self.supabase.table('ad_campaigns')\
                        .select('name, platform')\
                        .eq('id', campaign_id)\
                        .single()\
                        .execute()

                    campaign_data = campaign.data if campaign.data else {}

                    # Create alert
                    alert = await self._create_alert(
                        org_id=org_id,
                        alert_type='roas_drop',
                        severity='high',
                        campaign_id=campaign_id,
                        platform=campaign_data.get('platform', 'unknown'),
                        message=f"ROAS dropped {abs(pct_change):.1f}% for {campaign_data.get('name', campaign_id)}",
                        metadata={
                            'roas_yesterday': roas_yesterday,
                            'roas_day_before': roas_day_before,
                            'pct_change': pct_change,
                            'campaign_name': campaign_data.get('name')
                        }
                    )

                    alerts_created.append(alert)

                    logger.warning(
                        f"ROAS drop detected for campaign {campaign_id}: "
                        f"{roas_day_before:.2f}x → {roas_yesterday:.2f}x ({pct_change:.1f}%)"
                    )

            return alerts_created

        except Exception as e:
            logger.error(f"Error detecting ROAS drops: {e}", exc_info=True)
            return []

    async def detect_budget_overages(
        self,
        org_id: str
    ) -> List[Dict[str, Any]]:
        """
        Detect campaigns that exceeded their daily budget.

        Args:
            org_id: Organization ID

        Returns:
            List of alerts created
        """
        try:
            alerts_created = []

            # Get yesterday's spend
            yesterday = (datetime.utcnow() - timedelta(days=1)).date()

            # Get campaigns with configured budgets
            campaigns_query = self.supabase.table('ad_campaigns')\
                .select('id, name, platform, daily_budget')\
                .eq('org_id', org_id)\
                .eq('status', 'active')\
                .not_.is_('daily_budget', 'null')\
                .execute()

            campaigns = campaigns_query.data or []

            for campaign in campaigns:
                campaign_id = campaign['id']
                daily_budget = float(campaign.get('daily_budget', 0))

                if daily_budget <= 0:
                    continue

                # Get yesterday's spend for this campaign
                metrics_query = self.supabase.table('ad_metrics')\
                    .select('spend')\
                    .eq('campaign_id', campaign_id)\
                    .eq('metric_date', yesterday.isoformat())\
                    .single()\
                    .execute()

                if not metrics_query.data:
                    continue

                actual_spend = float(metrics_query.data.get('spend', 0))

                # Check if over budget
                if actual_spend > daily_budget:
                    overage_pct = ((actual_spend - daily_budget) / daily_budget) * 100

                    # Create alert
                    alert = await self._create_alert(
                        org_id=org_id,
                        alert_type='budget_exceeded',
                        severity='critical',
                        campaign_id=campaign_id,
                        platform=campaign.get('platform', 'unknown'),
                        message=f"Budget exceeded by {overage_pct:.1f}% for {campaign['name']}",
                        metadata={
                            'daily_budget': daily_budget,
                            'actual_spend': actual_spend,
                            'overage_amount': actual_spend - daily_budget,
                            'overage_pct': overage_pct,
                            'campaign_name': campaign['name'],
                            'date': yesterday.isoformat()
                        }
                    )

                    alerts_created.append(alert)

                    logger.warning(
                        f"Budget overage for campaign {campaign_id}: "
                        f"${daily_budget:.2f} budget, ${actual_spend:.2f} actual (+{overage_pct:.1f}%)"
                    )

            return alerts_created

        except Exception as e:
            logger.error(f"Error detecting budget overages: {e}", exc_info=True)
            return []

    async def _create_alert(
        self,
        org_id: str,
        alert_type: str,
        severity: str,
        message: str,
        campaign_id: Optional[str] = None,
        platform: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create an alert in the database.

        Args:
            org_id: Organization ID
            alert_type: Type of alert
            severity: Severity level
            message: Alert message
            campaign_id: Optional campaign ID
            platform: Optional platform
            metadata: Additional metadata

        Returns:
            Created alert record
        """
        alert_data = {
            'org_id': org_id,
            'alert_type': alert_type,
            'severity': severity,
            'campaign_id': campaign_id,
            'platform': platform,
            'message': message,
            'metadata': metadata or {},
            'acknowledged': False,
            'created_at': datetime.utcnow().isoformat()
        }

        result = self.supabase.table('ads_alerts').insert(alert_data).execute()

        alert = result.data[0] if result.data else alert_data

        # Send notification (Slack webhook)
        await self._send_notification(alert)

        return alert

    async def _send_notification(self, alert: Dict[str, Any]):
        """
        Send alert notification via Slack webhook.

        Args:
            alert: Alert data
        """
        try:
            # Get organization's Slack webhook URL
            org_id = alert['org_id']

            org_query = self.supabase.table('organizations')\
                .select('settings')\
                .eq('id', org_id)\
                .single()\
                .execute()

            if not org_query.data:
                return

            settings = org_query.data.get('settings', {})
            webhook_url = settings.get('slack_alerts_webhook')

            if not webhook_url:
                logger.debug(f"No Slack webhook configured for org {org_id}")
                return

            # Format message
            severity_emoji = {
                'low': '🔵',
                'medium': '🟡',
                'high': '🟠',
                'critical': '🔴'
            }

            emoji = severity_emoji.get(alert['severity'], '⚪')

            slack_message = {
                'text': f"{emoji} **Ads Alert: {alert['alert_type']}**",
                'blocks': [
                    {
                        'type': 'header',
                        'text': {
                            'type': 'plain_text',
                            'text': f"{emoji} Ads Alert: {alert['alert_type'].replace('_', ' ').title()}"
                        }
                    },
                    {
                        'type': 'section',
                        'fields': [
                            {
                                'type': 'mrkdwn',
                                'text': f"*Severity:*\n{alert['severity'].upper()}"
                            },
                            {
                                'type': 'mrkdwn',
                                'text': f"*Platform:*\n{alert.get('platform', 'N/A')}"
                            }
                        ]
                    },
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': f"*Message:*\n{alert['message']}"
                        }
                    }
                ]
            }

            # Send to Slack (using httpx)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=slack_message,
                    timeout=10.0
                )

                if response.status_code == 200:
                    logger.info(f"Sent Slack notification for alert {alert.get('id')}")
                else:
                    logger.warning(
                        f"Failed to send Slack notification: {response.status_code}"
                    )

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}", exc_info=True)

    async def get_alerts(
        self,
        org_id: str,
        acknowledged: Optional[bool] = None,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get alerts for an organization.

        Args:
            org_id: Organization ID
            acknowledged: Filter by acknowledged status
            severity: Filter by severity
            limit: Max results

        Returns:
            List of alerts
        """
        try:
            query = self.supabase.table('ads_alerts')\
                .select('*')\
                .eq('org_id', org_id)\
                .order('created_at', desc=True)\
                .limit(limit)

            if acknowledged is not None:
                query = query.eq('acknowledged', acknowledged)

            if severity:
                query = query.eq('severity', severity)

            result = query.execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error fetching alerts: {e}", exc_info=True)
            return []

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: Optional[str] = None
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: User ID who acknowledged

        Returns:
            Success status
        """
        try:
            update_data = {
                'acknowledged': True,
                'acknowledged_at': datetime.utcnow().isoformat(),
                'acknowledged_by': acknowledged_by
            }

            self.supabase.table('ads_alerts')\
                .update(update_data)\
                .eq('id', alert_id)\
                .execute()

            logger.info(f"Acknowledged alert {alert_id}")
            return True

        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}", exc_info=True)
            return False

    async def get_alert_stats(self, org_id: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Get alert statistics for an organization.

        Args:
            org_id: Organization ID
            days_back: Days to analyze

        Returns:
            Alert statistics
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days_back)

            # Get recent alerts
            alerts_query = self.supabase.table('ads_alerts')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff.isoformat())\
                .execute()

            alerts = alerts_query.data or []

            # Calculate stats
            total_alerts = len(alerts)
            by_type = {}
            by_severity = {}
            acknowledged_count = 0

            for alert in alerts:
                alert_type = alert.get('alert_type', 'unknown')
                severity = alert.get('severity', 'unknown')

                by_type[alert_type] = by_type.get(alert_type, 0) + 1
                by_severity[severity] = by_severity.get(severity, 0) + 1

                if alert.get('acknowledged'):
                    acknowledged_count += 1

            return {
                'total_alerts': total_alerts,
                'acknowledged': acknowledged_count,
                'pending': total_alerts - acknowledged_count,
                'by_type': by_type,
                'by_severity': by_severity,
                'period_days': days_back
            }

        except Exception as e:
            logger.error(f"Error getting alert stats: {e}", exc_info=True)
            return {}


# Singleton instance
_ads_alert_service: Optional[AdsAlertService] = None


def get_ads_alert_service() -> AdsAlertService:
    """Get singleton ads alert service instance"""
    global _ads_alert_service
    if _ads_alert_service is None:
        _ads_alert_service = AdsAlertService()
    return _ads_alert_service
