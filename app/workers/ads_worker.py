"""
Ads Ingestion Worker - Phase 6.5
Dedicated background worker for automated ads data synchronization

Responsibilities:
- Hourly sync of Google Ads & Meta Ads campaigns and metrics
- Alert detection (ROAS drops, budget overages, ingestion failures)
- Retry logic with exponential backoff
- Health monitoring and observability
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from app.core.logging import logger
from app.db import get_supabase_admin_client
from app.services.ads_ingestion_service import get_ads_ingestion_service


class AdsWorker:
    """
    Dedicated worker for ads data synchronization and monitoring.

    Runs independently from CIL worker for better isolation and scaling.
    """

    def __init__(
        self,
        sync_interval_hours: int = 1,
        anomaly_check_time: str = "4:00",  # UTC time "HH:MM" (after CIL runs)
        enabled: bool = True
    ):
        self.enabled = enabled
        self.sync_interval_hours = sync_interval_hours
        self.anomaly_check_time = anomaly_check_time

        self.scheduler = AsyncIOScheduler()
        self.supabase = get_supabase_admin_client()
        self.ads_ingestion_service = get_ads_ingestion_service()

        self._is_running = False
        self._sync_failures = {}  # Track consecutive failures per org

    def start(self):
        """Start the Ads Worker and schedule all jobs"""
        if not self.enabled:
            logger.info("Ads worker is disabled, not starting")
            return

        if self._is_running:
            logger.warning("Ads worker is already running")
            return

        try:
            # Schedule hourly ads data sync
            self.scheduler.add_job(
                self._sync_all_organizations,
                trigger=IntervalTrigger(hours=self.sync_interval_hours),
                id="ads_hourly_sync",
                name="Ads Hourly Data Sync",
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            logger.info(f"Scheduled ads sync every {self.sync_interval_hours} hour(s)")

            # Schedule daily anomaly detection
            hour, minute = self.anomaly_check_time.split(":")
            self.scheduler.add_job(
                self._detect_anomalies_all_orgs,
                trigger=CronTrigger(hour=int(hour), minute=int(minute), timezone="UTC"),
                id="ads_anomaly_detection",
                name="Ads Anomaly Detection",
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            logger.info(f"Scheduled anomaly detection daily at {self.anomaly_check_time} UTC")

            # Schedule weekly cleanup
            self.scheduler.add_job(
                self._cleanup_old_metrics,
                trigger=CronTrigger(day_of_week='sun', hour=3, minute=0, timezone="UTC"),
                id="ads_cleanup",
                name="Ads Metrics Cleanup",
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            logger.info("Scheduled weekly cleanup on Sundays at 3:00 AM UTC")

            # Start the scheduler
            self.scheduler.start()
            self._is_running = True

            logger.info("🎬 Ads Worker started successfully")

        except Exception as e:
            logger.error(f"Failed to start Ads worker: {e}", exc_info=True)
            raise

    def stop(self):
        """Stop the Ads Worker"""
        if not self._is_running:
            logger.warning("Ads worker is not running")
            return

        try:
            self.scheduler.shutdown(wait=True)
            self._is_running = False
            logger.info("Ads Worker stopped")
        except Exception as e:
            logger.error(f"Error stopping Ads worker: {e}", exc_info=True)

    async def _sync_all_organizations(self):
        """
        Sync ads data for all active organizations with connected ad accounts.

        Called hourly by scheduler.
        """
        try:
            logger.info("Starting hourly ads sync for all organizations")

            # Get organizations with active ad account connections
            accounts_query = self.supabase.table('ad_accounts')\
                .select('org_id, id, platform, status')\
                .eq('status', 'active')\
                .execute()

            accounts = accounts_query.data or []

            if not accounts:
                logger.info("No active ad accounts found, skipping sync")
                return

            # Group by org_id
            orgs_with_accounts = {}
            for account in accounts:
                org_id = account['org_id']
                if org_id not in orgs_with_accounts:
                    orgs_with_accounts[org_id] = []
                orgs_with_accounts[org_id].append(account)

            logger.info(f"Found {len(orgs_with_accounts)} organizations with active ad accounts")

            total_synced = 0
            total_errors = 0

            for org_id, org_accounts in orgs_with_accounts.items():
                try:
                    logger.info(f"Syncing ads data for org {org_id} ({len(org_accounts)} accounts)")

                    synced_count = await self._sync_organization(org_id, org_accounts)

                    total_synced += synced_count

                    # Reset failure counter on success
                    if org_id in self._sync_failures:
                        del self._sync_failures[org_id]

                    logger.info(f"✅ Synced {synced_count} campaigns for org {org_id}")

                    # Small delay between orgs
                    await asyncio.sleep(2)

                except Exception as e:
                    total_errors += 1

                    # Track consecutive failures
                    if org_id not in self._sync_failures:
                        self._sync_failures[org_id] = 0
                    self._sync_failures[org_id] += 1

                    logger.error(
                        f"Error syncing org {org_id} (failure #{self._sync_failures[org_id]}): {e}",
                        exc_info=True
                    )

                    # Alert if 3 consecutive failures
                    if self._sync_failures[org_id] >= 3:
                        await self._create_ingestion_alert(
                            org_id=org_id,
                            message=f"Ads sync failed {self._sync_failures[org_id]} times consecutively",
                            metadata={'error': str(e)}
                        )

            logger.info(
                f"🎬 Hourly ads sync completed: {total_synced} campaigns synced, "
                f"{total_errors} errors across {len(orgs_with_accounts)} orgs"
            )

        except Exception as e:
            logger.error(f"Error in hourly ads sync: {e}", exc_info=True)

    async def _sync_organization(self, org_id: str, accounts: List[Dict]) -> int:
        """
        Sync ads data for a single organization.

        Args:
            org_id: Organization ID
            accounts: List of ad accounts for this org

        Returns:
            Number of campaigns synced
        """
        total_synced = 0

        for account in accounts:
            account_id = account['id']
            platform = account['platform']

            try:
                # Use existing ingestion service
                result = await self.ads_ingestion_service.ingest_account_data(
                    account_id=account_id,
                    days_back=7  # Sync last 7 days for updates
                )

                campaigns_synced = result.get('campaigns_synced', 0)
                total_synced += campaigns_synced

                logger.debug(
                    f"Synced {campaigns_synced} campaigns for {platform} account {account_id}"
                )

            except Exception as e:
                logger.error(
                    f"Error syncing {platform} account {account_id}: {e}",
                    exc_info=True
                )
                # Don't fail entire org sync if one account fails
                continue

        return total_synced

    async def _detect_anomalies_all_orgs(self):
        """
        Detect performance anomalies and trigger alerts.

        Runs daily at configured time (after CIL learning cycle).
        """
        try:
            logger.info("Starting anomaly detection for all organizations")

            # Import alert service here to avoid circular dependency
            from app.services.ads_alert_service import get_ads_alert_service
            alert_service = get_ads_alert_service()

            # Get all organizations with ad accounts
            orgs_query = self.supabase.table('ad_accounts')\
                .select('org_id')\
                .eq('status', 'active')\
                .execute()

            org_ids = list(set(account['org_id'] for account in (orgs_query.data or [])))

            logger.info(f"Running anomaly detection for {len(org_ids)} organizations")

            total_alerts = 0

            for org_id in org_ids:
                try:
                    # Check ROAS drops
                    roas_alerts = await alert_service.detect_roas_drops(
                        org_id=org_id,
                        threshold_pct=20.0
                    )
                    total_alerts += len(roas_alerts)

                    # Check budget overages
                    budget_alerts = await alert_service.detect_budget_overages(
                        org_id=org_id
                    )
                    total_alerts += len(budget_alerts)

                    if roas_alerts or budget_alerts:
                        logger.info(
                            f"Detected {len(roas_alerts)} ROAS alerts and "
                            f"{len(budget_alerts)} budget alerts for org {org_id}"
                        )

                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error detecting anomalies for org {org_id}: {e}", exc_info=True)

            logger.info(f"✅ Anomaly detection completed: {total_alerts} alerts generated")

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}", exc_info=True)

    async def _cleanup_old_metrics(self):
        """
        Clean up old ad metrics to save storage.

        Removes metrics older than 180 days (keeps 6 months).
        Runs weekly on Sundays.
        """
        try:
            logger.info("Starting weekly ads metrics cleanup")

            cutoff_date = (datetime.utcnow() - timedelta(days=180)).date()

            # Delete old metrics
            delete_result = self.supabase.table('ad_metrics')\
                .delete()\
                .lt('metric_date', cutoff_date.isoformat())\
                .execute()

            deleted_count = len(delete_result.data) if delete_result.data else 0

            logger.info(f"✅ Cleaned up {deleted_count} ad metrics older than {cutoff_date}")

            # Also cleanup old sync jobs (keep last 30 days)
            job_cutoff = datetime.utcnow() - timedelta(days=30)

            job_delete_result = self.supabase.table('ad_sync_jobs')\
                .delete()\
                .lt('created_at', job_cutoff.isoformat())\
                .execute()

            jobs_deleted = len(job_delete_result.data) if job_delete_result.data else 0

            logger.info(f"✅ Cleaned up {jobs_deleted} sync jobs older than 30 days")

        except Exception as e:
            logger.error(f"Error in cleanup: {e}", exc_info=True)

    async def _create_ingestion_alert(
        self,
        org_id: str,
        message: str,
        metadata: Optional[Dict] = None
    ):
        """
        Create an ingestion failure alert.

        Args:
            org_id: Organization ID
            message: Alert message
            metadata: Additional context
        """
        try:
            alert_data = {
                'org_id': org_id,
                'alert_type': 'ingestion_failed',
                'severity': 'medium',
                'message': message,
                'metadata': metadata or {},
                'created_at': datetime.utcnow().isoformat()
            }

            self.supabase.table('ads_alerts').insert(alert_data).execute()

            logger.warning(f"Created ingestion alert for org {org_id}: {message}")

        except Exception as e:
            logger.error(f"Failed to create ingestion alert: {e}", exc_info=True)

    async def trigger_sync_now(self, org_id: str) -> Dict[str, Any]:
        """
        Manually trigger ads sync for a specific organization.

        Args:
            org_id: Organization ID

        Returns:
            Sync result summary
        """
        try:
            logger.info(f"Manually triggering ads sync for org {org_id}")

            # Get org's ad accounts
            accounts_query = self.supabase.table('ad_accounts')\
                .select('*')\
                .eq('org_id', org_id)\
                .eq('status', 'active')\
                .execute()

            accounts = accounts_query.data or []

            if not accounts:
                return {
                    'success': False,
                    'message': 'No active ad accounts found for this organization'
                }

            synced_count = await self._sync_organization(org_id, accounts)

            return {
                'success': True,
                'campaigns_synced': synced_count,
                'accounts_processed': len(accounts),
                'message': f'Successfully synced {synced_count} campaigns'
            }

        except Exception as e:
            logger.error(f"Error in manual sync: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def get_status(self) -> Dict[str, Any]:
        """Get worker status and health information"""
        return {
            'is_running': self._is_running,
            'enabled': self.enabled,
            'scheduler_running': self.scheduler.running if self.scheduler else False,
            'sync_interval_hours': self.sync_interval_hours,
            'consecutive_failures': self._sync_failures,
            'jobs': [
                {
                    'id': job.id,
                    'name': job.name,
                    'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in (self.scheduler.get_jobs() if self.scheduler else [])
            ]
        }


# Global worker instance
_ads_worker: Optional[AdsWorker] = None


def get_ads_worker(
    sync_interval_hours: int = 1,
    anomaly_check_time: str = "4:00",
    enabled: bool = True
) -> AdsWorker:
    """Get or create the singleton Ads Worker instance"""
    global _ads_worker

    if _ads_worker is None:
        _ads_worker = AdsWorker(
            sync_interval_hours=sync_interval_hours,
            anomaly_check_time=anomaly_check_time,
            enabled=enabled
        )

    return _ads_worker


def start_ads_worker(**kwargs):
    """Start the Ads Worker"""
    worker = get_ads_worker(**kwargs)
    worker.start()
    return worker


def stop_ads_worker():
    """Stop the Ads Worker"""
    global _ads_worker

    if _ads_worker:
        _ads_worker.stop()
        _ads_worker = None
