"""
CIL Async Worker
Background worker that runs CIL meta-learning cycles on a schedule
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.core.logging import logger
from app.db import get_supabase_admin_client
from app.services.cil_meta_learning_service import get_cil_meta_learning_service
from app.services.cil_telemetry_service import get_cil_telemetry_service
from app.services.cil_policy_service import get_cil_policy_service
from app.services.cil_prompt_optimizer import get_cil_prompt_optimizer


class CILWorker:
    """
    Background worker for CIL operations

    Responsibilities:
    - Run nightly meta-learning cycles
    - Ingest telemetry data periodically
    - Process auto-apply proposals
    - Maintain learning cycle health
    """

    def __init__(
        self,
        learning_cycle_time: str = "2:00",  # UTC time "HH:MM"
        telemetry_interval_minutes: int = 5,
        proposal_check_interval_minutes: int = 15,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.learning_cycle_time = learning_cycle_time
        self.telemetry_interval_minutes = telemetry_interval_minutes
        self.proposal_check_interval_minutes = proposal_check_interval_minutes

        self.scheduler = AsyncIOScheduler()
        self.supabase = get_supabase_admin_client()

        self.meta_learning_service = get_cil_meta_learning_service()
        self.telemetry_service = get_cil_telemetry_service()
        self.policy_service = get_cil_policy_service()
        self.prompt_optimizer = get_cil_prompt_optimizer()

        self._is_running = False

    def start(self):
        """Start the CIL worker and schedule all jobs"""
        if not self.enabled:
            logger.info("CIL worker is disabled, not starting")
            return

        if self._is_running:
            logger.warning("CIL worker is already running")
            return

        try:
            # Schedule nightly meta-learning cycle
            hour, minute = self.learning_cycle_time.split(":")
            self.scheduler.add_job(
                self._run_all_learning_cycles,
                trigger=CronTrigger(hour=int(hour), minute=int(minute), timezone="UTC"),
                id="cil_nightly_learning",
                name="CIL Nightly Meta-Learning Cycle",
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            logger.info(f"Scheduled nightly meta-learning at {self.learning_cycle_time} UTC")

            # Schedule telemetry ingestion
            self.scheduler.add_job(
                self._ingest_all_telemetry,
                trigger=IntervalTrigger(minutes=self.telemetry_interval_minutes),
                id="cil_telemetry_ingestion",
                name="CIL Telemetry Ingestion",
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            logger.info(f"Scheduled telemetry ingestion every {self.telemetry_interval_minutes} minutes")

            # Schedule proposal auto-apply check
            self.scheduler.add_job(
                self._process_auto_apply_proposals,
                trigger=IntervalTrigger(minutes=self.proposal_check_interval_minutes),
                id="cil_proposal_processor",
                name="CIL Proposal Auto-Apply Processor",
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            logger.info(f"Scheduled proposal processing every {self.proposal_check_interval_minutes} minutes")

            # Schedule A/B test evaluation (daily at 3 AM UTC, after learning cycle)
            self.scheduler.add_job(
                self._evaluate_prompt_ab_tests,
                trigger=CronTrigger(hour=3, minute=0, timezone="UTC"),
                id="cil_ab_test_evaluation",
                name="CIL A/B Test Evaluation",
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            logger.info("Scheduled A/B test evaluation daily at 3:00 AM UTC")

            # Phase 6, Step 2: Schedule hourly ads telemetry ingestion
            self.scheduler.add_job(
                self._ingest_ads_telemetry_hourly,
                trigger=IntervalTrigger(hours=1),
                id="cil_ads_telemetry_hourly",
                name="CIL Ads Telemetry Hourly Ingestion",
                replace_existing=True,
                max_instances=1,
                coalesce=True
            )
            logger.info("Scheduled ads telemetry ingestion hourly")

            # Start the scheduler
            self.scheduler.start()
            self._is_running = True

            logger.info("🧠 CIL Worker started successfully")

        except Exception as e:
            logger.error(f"Failed to start CIL worker: {e}", exc_info=True)
            raise

    def stop(self):
        """Stop the CIL worker"""
        if not self._is_running:
            logger.warning("CIL worker is not running")
            return

        try:
            self.scheduler.shutdown(wait=True)
            self._is_running = False
            logger.info("CIL Worker stopped")
        except Exception as e:
            logger.error(f"Error stopping CIL worker: {e}", exc_info=True)

    async def _run_all_learning_cycles(self):
        """
        Run meta-learning cycles for all active organizations
        Called nightly by scheduler
        """
        try:
            logger.info("Starting nightly CIL meta-learning cycles")

            # Get all active organizations
            orgs_query = self.supabase.table('organizations')\
                .select('id, name')\
                .eq('is_active', True)\
                .execute()

            organizations = orgs_query.data or []

            if not organizations:
                logger.info("No active organizations found, skipping learning cycle")
                return

            logger.info(f"Running learning cycles for {len(organizations)} organizations")

            results = []
            for org in organizations:
                org_id = org['id']
                org_name = org.get('name', org_id)

                try:
                    logger.info(f"Running learning cycle for org: {org_name} ({org_id})")

                    # Run meta-learning cycle with 7 days of data
                    result = await self.meta_learning_service.run_learning_cycle(
                        org_id=org_id,
                        days_back=7,
                        algorithms=None  # Run all algorithms
                    )

                    if result:
                        results.append({
                            'org_id': org_id,
                            'org_name': org_name,
                            'status': 'success',
                            'cycle_id': result.get('cycle_id'),
                            'proposals_created': result.get('proposals_created', 0)
                        })
                        logger.info(
                            f"✅ Learning cycle completed for {org_name}: "
                            f"{result.get('proposals_created', 0)} proposals created"
                        )
                    else:
                        results.append({
                            'org_id': org_id,
                            'org_name': org_name,
                            'status': 'no_action',
                            'message': 'No significant findings or insufficient data'
                        })
                        logger.info(f"No action needed for {org_name}")

                    # Small delay between orgs to avoid overwhelming the system
                    await asyncio.sleep(2)

                except Exception as e:
                    logger.error(
                        f"Error running learning cycle for org {org_name}: {e}",
                        exc_info=True
                    )
                    results.append({
                        'org_id': org_id,
                        'org_name': org_name,
                        'status': 'error',
                        'error': str(e)
                    })

            # Log summary
            success_count = sum(1 for r in results if r['status'] == 'success')
            error_count = sum(1 for r in results if r['status'] == 'error')

            logger.info(
                f"🧠 Nightly CIL learning cycles completed: "
                f"{success_count} successful, {error_count} errors, "
                f"{len(organizations)} total orgs"
            )

        except Exception as e:
            logger.error(f"Error in nightly learning cycle: {e}", exc_info=True)

    async def _ingest_all_telemetry(self):
        """
        Ingest telemetry data for all active organizations
        Called every N minutes by scheduler
        """
        try:
            # Get all active organizations
            orgs_query = self.supabase.table('organizations')\
                .select('id')\
                .eq('is_active', True)\
                .execute()

            organizations = orgs_query.data or []

            if not organizations:
                return

            total_ingested = 0
            total_errors = 0

            for org in organizations:
                org_id = org['id']

                try:
                    # Ingest last 1 hour of data
                    results = await self.telemetry_service.ingest_all_sources(
                        org_id=org_id,
                        hours_back=1
                    )

                    ingested = sum(v for k, v in results.items() if k != 'errors')
                    total_ingested += ingested
                    total_errors += results.get('errors', 0)

                    if ingested > 0:
                        logger.debug(f"Ingested {ingested} telemetry records for org {org_id}")

                except Exception as e:
                    logger.error(f"Error ingesting telemetry for org {org_id}: {e}")
                    total_errors += 1

            if total_ingested > 0 or total_errors > 0:
                logger.info(
                    f"Telemetry ingestion: {total_ingested} records ingested, "
                    f"{total_errors} errors, {len(organizations)} orgs"
                )

        except Exception as e:
            logger.error(f"Error in telemetry ingestion: {e}", exc_info=True)

    async def _process_auto_apply_proposals(self):
        """
        Process proposals that are ready for auto-apply
        Called every N minutes by scheduler
        """
        try:
            # Find pending minor proposals whose auto_apply_after time has passed
            now = datetime.utcnow()

            proposals_query = self.supabase.table('cil_policy_proposals')\
                .select('*')\
                .eq('status', 'pending')\
                .eq('change_type', 'minor')\
                .not_.is_('auto_apply_after', 'null')\
                .lte('auto_apply_after', now.isoformat())\
                .execute()

            proposals = proposals_query.data or []

            if not proposals:
                return

            logger.info(f"Processing {len(proposals)} auto-apply proposals")

            for proposal in proposals:
                proposal_id = proposal['id']
                org_id = proposal['org_id']

                try:
                    import json

                    # Parse the proposed config
                    proposed_config = proposal['proposed_policy_config']
                    if isinstance(proposed_config, str):
                        proposed_config = json.loads(proposed_config)

                    change_details = proposal.get('change_details')
                    if isinstance(change_details, str):
                        change_details = json.loads(change_details)

                    # Create and activate the new policy
                    policy = self.policy_service.create_policy(
                        org_id=org_id,
                        policy_config=proposed_config,
                        learning_cycle_id=proposal.get('learning_cycle_id'),
                        change_reason=f"Auto-applied proposal {proposal_id}",
                        change_summary=change_details,
                        approval_required=False,  # Auto-apply
                        created_by="cil_worker_auto_apply"
                    )

                    if policy:
                        # Mark proposal as auto-applied
                        self.supabase.table('cil_policy_proposals')\
                            .update({
                                'status': 'auto_applied',
                                'reviewed_at': now.isoformat(),
                                'reviewed_by': 'cil_worker',
                                'review_notes': 'Auto-applied after timeout'
                            })\
                            .eq('id', proposal_id)\
                            .execute()

                        logger.info(
                            f"✅ Auto-applied proposal {proposal_id} for org {org_id} "
                            f"(new policy v{policy['version']})"
                        )
                    else:
                        logger.error(f"Failed to create policy from proposal {proposal_id}")

                except Exception as e:
                    logger.error(
                        f"Error processing auto-apply proposal {proposal_id}: {e}",
                        exc_info=True
                    )

        except Exception as e:
            logger.error(f"Error in auto-apply proposal processing: {e}", exc_info=True)

    async def _ingest_ads_telemetry_hourly(self):
        """
        Ingest ads telemetry data for all active organizations.

        Phase 6, Step 2: Aggregates 30-day ad metrics from ad_campaigns and ad_metrics tables,
        calculates performance scores, and stores in cil_ads_telemetry table.

        Called hourly by scheduler.
        """
        try:
            logger.info("Starting hourly ads telemetry ingestion")

            # Get all active organizations
            orgs_query = self.supabase.table('organizations')\
                .select('id, name')\
                .eq('is_active', True)\
                .execute()

            organizations = orgs_query.data or []

            if not organizations:
                logger.info("No active organizations found, skipping ads telemetry ingestion")
                return

            total_records = 0
            total_errors = 0

            for org in organizations:
                org_id = org['id']
                org_name = org.get('name', org_id)

                try:
                    # Ingest ads telemetry using the telemetry service
                    # This will aggregate metrics from the last 30 days
                    cutoff_time = datetime.utcnow() - timedelta(hours=1)

                    records_ingested = await self.telemetry_service._ingest_ads_telemetry(
                        org_id=org_id,
                        cutoff_time=cutoff_time
                    )

                    if records_ingested > 0:
                        total_records += records_ingested
                        logger.info(
                            f"Ingested {records_ingested} ads telemetry records for org {org_name}"
                        )

                    await asyncio.sleep(0.5)  # Small delay between orgs

                except Exception as e:
                    logger.error(
                        f"Error ingesting ads telemetry for org {org_name}: {e}",
                        exc_info=True
                    )
                    total_errors += 1

            logger.info(
                f"✅ Hourly ads telemetry ingestion completed: "
                f"{total_records} records ingested, {total_errors} errors, "
                f"{len(organizations)} orgs processed"
            )

        except Exception as e:
            logger.error(f"Error in hourly ads telemetry ingestion: {e}", exc_info=True)

    async def _evaluate_prompt_ab_tests(self):
        """
        Evaluate A/B tests for all organizations
        Called daily at 3 AM UTC by scheduler
        """
        try:
            logger.info("Starting daily A/B test evaluation")

            # Get all active organizations
            orgs_query = self.supabase.table('organizations')\
                .select('id, name')\
                .eq('is_active', True)\
                .execute()

            organizations = orgs_query.data or []

            if not organizations:
                logger.info("No active organizations found, skipping A/B test evaluation")
                return

            total_tests_evaluated = 0
            total_winners_promoted = 0

            for org in organizations:
                org_id = org['id']
                org_name = org.get('name', org_id)

                try:
                    # Evaluate all A/B tests for this org
                    results = await self.prompt_optimizer.evaluate_ab_tests(org_id)

                    if results:
                        for result in results:
                            total_tests_evaluated += 1

                            if result.get('winner') and result['decision'] != 'ongoing':
                                total_winners_promoted += 1
                                logger.info(
                                    f"A/B test result for '{result['template_name']}' in {org_name}: "
                                    f"{result['winner']} wins ({result['decision']})"
                                )

                    await asyncio.sleep(1)  # Small delay between orgs

                except Exception as e:
                    logger.error(
                        f"Error evaluating A/B tests for org {org_name}: {e}",
                        exc_info=True
                    )

            logger.info(
                f"✅ A/B test evaluation completed: "
                f"{total_tests_evaluated} tests evaluated, "
                f"{total_winners_promoted} winners promoted"
            )

        except Exception as e:
            logger.error(f"Error in A/B test evaluation: {e}", exc_info=True)

    async def trigger_learning_cycle_now(self, org_id: str) -> Optional[Dict[str, Any]]:
        """
        Manually trigger a learning cycle for a specific organization
        Useful for testing or on-demand learning

        Args:
            org_id: Organization ID

        Returns:
            Learning cycle result or None
        """
        try:
            logger.info(f"Manually triggering learning cycle for org {org_id}")

            result = await self.meta_learning_service.run_learning_cycle(
                org_id=org_id,
                days_back=7,
                algorithms=None
            )

            if result:
                logger.info(
                    f"✅ Manual learning cycle completed for {org_id}: "
                    f"{result.get('proposals_created', 0)} proposals created"
                )
            else:
                logger.info(f"No action needed for {org_id}")

            return result

        except Exception as e:
            logger.error(f"Error in manual learning cycle: {e}", exc_info=True)
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get worker status and scheduled job information"""
        return {
            'is_running': self._is_running,
            'enabled': self.enabled,
            'scheduler_running': self.scheduler.running if self.scheduler else False,
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
_cil_worker: Optional[CILWorker] = None


def get_cil_worker(
    learning_cycle_time: str = "2:00",
    telemetry_interval_minutes: int = 5,
    proposal_check_interval_minutes: int = 15,
    enabled: bool = True
) -> CILWorker:
    """Get or create the singleton CIL worker instance"""
    global _cil_worker

    if _cil_worker is None:
        _cil_worker = CILWorker(
            learning_cycle_time=learning_cycle_time,
            telemetry_interval_minutes=telemetry_interval_minutes,
            proposal_check_interval_minutes=proposal_check_interval_minutes,
            enabled=enabled
        )

    return _cil_worker


def start_cil_worker(**kwargs):
    """Start the CIL worker"""
    worker = get_cil_worker(**kwargs)
    worker.start()
    return worker


def stop_cil_worker():
    """Stop the CIL worker"""
    global _cil_worker

    if _cil_worker:
        _cil_worker.stop()
        _cil_worker = None
