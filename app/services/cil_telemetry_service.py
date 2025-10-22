"""
CIL Telemetry Service
Ingests and normalizes events from all system sources for meta-learning
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID
import asyncio

from app.core.logging import logger
from app.db import get_supabase_admin_client


class CILTelemetryService:
    """
    Ingests events from multiple sources and normalizes into cil_telemetry table

    Sources:
    - ai_reflections: Post-execution agent learnings
    - ai_actions_pending: Approval decisions
    - ai_action_plans: Plan execution outcomes
    - ai_agent_events: Agent coordination events
    - reasoning_logs: Cognitive decision engine reasoning
    """

    def __init__(self):
        self.supabase = get_supabase_admin_client()
        self._last_ingestion_time: Dict[str, datetime] = {}

    async def ingest_all_sources(self, org_id: str, hours_back: int = 1) -> Dict[str, int]:
        """
        Ingest events from all sources for the given org

        Args:
            org_id: Organization ID
            hours_back: How many hours of history to ingest

        Returns:
            Dictionary with ingestion counts per source
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

        results = {
            'reflections': 0,
            'approvals': 0,
            'action_plans': 0,
            'agent_events': 0,
            'ads_telemetry': 0,  # Phase 6, Step 2
            'errors': 0
        }

        try:
            # Ingest reflections
            results['reflections'] = await self._ingest_reflections(org_id, cutoff_time)

            # Ingest approval decisions
            results['approvals'] = await self._ingest_approvals(org_id, cutoff_time)

            # Ingest action plan outcomes
            results['action_plans'] = await self._ingest_action_plans(org_id, cutoff_time)

            # Ingest agent events
            results['agent_events'] = await self._ingest_agent_events(org_id, cutoff_time)

            # Phase 6, Step 2: Ingest ads telemetry
            results['ads_telemetry'] = await self._ingest_ads_telemetry(org_id, cutoff_time)

            logger.info(f"CIL telemetry ingestion completed for org {org_id}", extra=results)

        except Exception as e:
            logger.error(f"Error in CIL telemetry ingestion: {e}", exc_info=True)
            results['errors'] = 1

        return results

    async def _ingest_reflections(self, org_id: str, cutoff_time: datetime) -> int:
        """Ingest events from ai_reflections table"""
        try:
            # Get reflections not yet in telemetry
            reflections_query = self.supabase.table('ai_reflections')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff_time.isoformat())\
                .execute()

            reflections = reflections_query.data or []

            telemetry_records = []
            for reflection in reflections:
                # Check if already ingested
                existing = self.supabase.table('cil_telemetry')\
                    .select('id')\
                    .eq('source_type', 'reflection')\
                    .eq('source_id', reflection['id'])\
                    .execute()

                if existing.data:
                    continue  # Already ingested

                # Extract quality score from reflection
                quality_score = reflection.get('overall_success')
                if quality_score is not None and isinstance(quality_score, bool):
                    quality_score = 1.0 if quality_score else 0.5

                telemetry_record = {
                    'org_id': org_id,
                    'source_type': 'reflection',
                    'source_id': reflection['id'],
                    'agent_type': reflection.get('agent_type'),
                    'outcome': 'success' if reflection.get('overall_success') else 'failure',
                    'outcome_quality_score': quality_score,
                    'metadata': {
                        'summary': reflection.get('summary'),
                        'insights': reflection.get('insights'),
                        'improvements_suggested': reflection.get('improvements_suggested'),
                        'performance_metrics': reflection.get('performance_metrics')
                    },
                    'created_at': reflection.get('created_at')
                }

                telemetry_records.append(telemetry_record)

            if telemetry_records:
                self.supabase.table('cil_telemetry').insert(telemetry_records).execute()
                logger.info(f"Ingested {len(telemetry_records)} reflections into CIL telemetry")

            return len(telemetry_records)

        except Exception as e:
            logger.error(f"Error ingesting reflections: {e}", exc_info=True)
            return 0

    async def _ingest_approvals(self, org_id: str, cutoff_time: datetime) -> int:
        """Ingest events from ai_actions_pending table"""
        try:
            # Get approved/rejected actions
            approvals_query = self.supabase.table('ai_actions_pending')\
                .select('*, ai_action_plans(agent_type, goal)')\
                .eq('org_id', org_id)\
                .gte('updated_at', cutoff_time.isoformat())\
                .in_('status', ['approved', 'rejected', 'expired'])\
                .execute()

            approvals = approvals_query.data or []

            telemetry_records = []
            for approval in approvals:
                # Check if already ingested
                existing = self.supabase.table('cil_telemetry')\
                    .select('id')\
                    .eq('source_type', 'approval')\
                    .eq('source_id', approval['id'])\
                    .execute()

                if existing.data:
                    continue

                # Calculate approval time
                approval_time_minutes = None
                if approval.get('created_at') and approval.get('approved_at'):
                    created = datetime.fromisoformat(approval['created_at'].replace('Z', '+00:00'))
                    approved = datetime.fromisoformat(approval['approved_at'].replace('Z', '+00:00'))
                    approval_time_minutes = int((approved - created).total_seconds() / 60)

                # Determine outcome
                outcome = approval['status']  # 'approved', 'rejected', 'expired'

                plan_data = approval.get('ai_action_plans', {})

                telemetry_record = {
                    'org_id': org_id,
                    'source_type': 'approval',
                    'source_id': approval['id'],
                    'agent_type': plan_data.get('agent_type') if isinstance(plan_data, dict) else None,
                    'risk_level': approval.get('risk_level'),
                    'outcome': outcome,
                    'required_approval': True,
                    'approval_decision': outcome,
                    'approval_time_minutes': approval_time_minutes,
                    'metadata': {
                        'action_name': approval.get('action_name'),
                        'description': approval.get('description'),
                        'rejection_reason': approval.get('rejection_reason')
                    },
                    'created_at': approval.get('updated_at')
                }

                telemetry_records.append(telemetry_record)

            if telemetry_records:
                self.supabase.table('cil_telemetry').insert(telemetry_records).execute()
                logger.info(f"Ingested {len(telemetry_records)} approvals into CIL telemetry")

            return len(telemetry_records)

        except Exception as e:
            logger.error(f"Error ingesting approvals: {e}", exc_info=True)
            return 0

    async def _ingest_action_plans(self, org_id: str, cutoff_time: datetime) -> int:
        """Ingest events from ai_action_plans table"""
        try:
            # Get completed/failed action plans
            plans_query = self.supabase.table('ai_action_plans')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('updated_at', cutoff_time.isoformat())\
                .in_('status', ['completed', 'failed', 'cancelled'])\
                .execute()

            plans = plans_query.data or []

            telemetry_records = []
            for plan in plans:
                # Check if already ingested
                existing = self.supabase.table('cil_telemetry')\
                    .select('id')\
                    .eq('source_type', 'action_plan')\
                    .eq('source_id', plan['id'])\
                    .execute()

                if existing.data:
                    continue

                # Determine outcome
                outcome = 'success' if plan['status'] == 'completed' else 'failure'

                # Calculate quality score based on completion percentage
                quality_score = None
                if plan.get('completed_steps') is not None and plan.get('total_steps') is not None:
                    total = plan['total_steps']
                    completed = plan['completed_steps']
                    if total > 0:
                        quality_score = completed / total

                telemetry_record = {
                    'org_id': org_id,
                    'source_type': 'action_plan',
                    'source_id': plan['id'],
                    'agent_type': plan.get('agent_type'),
                    'outcome': outcome,
                    'outcome_quality_score': quality_score,
                    'response_time_ms': plan.get('execution_time_ms'),
                    'risk_level': plan.get('risk_level'),
                    'metadata': {
                        'goal': plan.get('goal'),
                        'status': plan.get('status'),
                        'total_steps': plan.get('total_steps'),
                        'completed_steps': plan.get('completed_steps')
                    },
                    'created_at': plan.get('updated_at')
                }

                telemetry_records.append(telemetry_record)

            if telemetry_records:
                self.supabase.table('cil_telemetry').insert(telemetry_records).execute()
                logger.info(f"Ingested {len(telemetry_records)} action plans into CIL telemetry")

            return len(telemetry_records)

        except Exception as e:
            logger.error(f"Error ingesting action plans: {e}", exc_info=True)
            return 0

    async def _ingest_agent_events(self, org_id: str, cutoff_time: datetime) -> int:
        """Ingest events from ai_agent_events table"""
        try:
            # Get recent agent events
            events_query = self.supabase.table('ai_agent_events')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff_time.isoformat())\
                .execute()

            events = events_query.data or []

            telemetry_records = []
            for event in events:
                # Check if already ingested
                existing = self.supabase.table('cil_telemetry')\
                    .select('id')\
                    .eq('source_type', 'agent_event')\
                    .eq('source_id', event['id'])\
                    .execute()

                if existing.data:
                    continue

                # Agent events are coordination signals, outcome is implicit
                outcome = event.get('metadata', {}).get('outcome', 'success')

                telemetry_record = {
                    'org_id': org_id,
                    'source_type': 'agent_event',
                    'source_id': event['id'],
                    'agent_type': event.get('agent_type'),
                    'outcome': outcome,
                    'metadata': {
                        'event_type': event.get('event_type'),
                        'event_data': event.get('metadata')
                    },
                    'created_at': event.get('created_at')
                }

                telemetry_records.append(telemetry_record)

            if telemetry_records:
                self.supabase.table('cil_telemetry').insert(telemetry_records).execute()
                logger.info(f"Ingested {len(telemetry_records)} agent events into CIL telemetry")

            return len(telemetry_records)

        except Exception as e:
            logger.error(f"Error ingesting agent events: {e}", exc_info=True)
            return 0

    def get_telemetry_stats(self, org_id: str, days_back: int = 7) -> Dict[str, Any]:
        """Get telemetry statistics for an organization"""
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

            # Get total records
            total_query = self.supabase.table('cil_telemetry')\
                .select('id', count='exact')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff_date)\
                .execute()

            total_records = total_query.count or 0

            # Get records by source type
            by_source = {}
            for source_type in ['reflection', 'approval', 'action_plan', 'agent_event']:
                source_query = self.supabase.table('cil_telemetry')\
                    .select('id', count='exact')\
                    .eq('org_id', org_id)\
                    .eq('source_type', source_type)\
                    .gte('created_at', cutoff_date)\
                    .execute()

                by_source[source_type] = source_query.count or 0

            # Get success rate
            success_query = self.supabase.table('cil_telemetry')\
                .select('id', count='exact')\
                .eq('org_id', org_id)\
                .eq('outcome', 'success')\
                .gte('created_at', cutoff_date)\
                .execute()

            success_count = success_query.count or 0
            success_rate = success_count / total_records if total_records > 0 else 0

            return {
                'total_records': total_records,
                'by_source': by_source,
                'success_rate': round(success_rate, 3),
                'period_days': days_back
            }

        except Exception as e:
            logger.error(f"Error getting telemetry stats: {e}", exc_info=True)
            return {
                'total_records': 0,
                'by_source': {},
                'success_rate': 0,
                'period_days': days_back,
                'error': str(e)
            }

    async def _ingest_ads_telemetry(self, org_id: str, cutoff_time: datetime) -> int:
        """
        Ingest ads platform performance metrics into CIL telemetry.

        Phase 6, Step 2: Fetches aggregated ad metrics from ad_campaigns and ad_metrics
        tables, calculates performance scores, and stores in cil_ads_telemetry.

        Args:
            org_id: Organization ID
            cutoff_time: Only process campaigns updated after this time

        Returns:
            Number of campaigns ingested
        """
        try:
            # Get active ad accounts for this org
            accounts_query = self.supabase.table('ad_accounts')\
                .select('id, platform')\
                .eq('org_id', org_id)\
                .eq('status', 'active')\
                .execute()

            if not accounts_query.data:
                logger.debug(f"No active ad accounts for org {org_id}")
                return 0

            telemetry_records = []
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=30)

            for account in accounts_query.data:
                # Get campaigns for this account
                campaigns_query = self.supabase.table('ad_campaigns')\
                    .select('id, campaign_name, platform')\
                    .eq('account_id', account['id'])\
                    .in_('status', ['active', 'paused'])\
                    .execute()

                if not campaigns_query.data:
                    continue

                for campaign in campaigns_query.data:
                    # Check if already ingested for this time period
                    existing = self.supabase.table('cil_ads_telemetry')\
                        .select('id')\
                        .eq('campaign_id', campaign['id'])\
                        .eq('metrics_end_date', end_date.isoformat())\
                        .execute()

                    if existing.data:
                        continue  # Already ingested for this period

                    # Get aggregated metrics using database function
                    metrics_query = self.supabase.rpc(
                        'get_campaign_performance_summary',
                        {
                            'p_campaign_id': campaign['id'],
                            'p_days': 30
                        }
                    ).execute()

                    if not metrics_query.data or len(metrics_query.data) == 0:
                        continue  # No metrics available

                    metrics = metrics_query.data[0]

                    # Skip if no activity
                    if not metrics.get('total_impressions') or metrics['total_impressions'] == 0:
                        continue

                    # Calculate performance score
                    roas = float(metrics.get('avg_roas', 0) or 0)
                    ctr = float(metrics.get('avg_ctr', 0) or 0)
                    conversions = int(metrics.get('total_conversions', 0) or 0)

                    # Normalize to 0-1 score
                    performance_score = min(1.0, max(0.0,
                        (min(roas / 5.0, 1.0) * 0.5) +        # ROAS weight: 50%
                        (min(ctr / 3.0, 1.0) * 0.3) +         # CTR weight: 30%
                        (min(conversions / 100.0, 1.0) * 0.2) # Conversions weight: 20%
                    ))

                    telemetry_record = {
                        'org_id': org_id,
                        'platform': campaign['platform'],
                        'campaign_id': campaign['id'],
                        'campaign_name': campaign['campaign_name'],
                        'impressions': int(metrics.get('total_impressions', 0)),
                        'clicks': int(metrics.get('total_clicks', 0)),
                        'ctr': float(metrics.get('avg_ctr', 0)),
                        'spend': float(metrics.get('total_spend', 0)),
                        'conversions': conversions,
                        'conversion_rate': None,  # Not in summary function
                        'cpa': float(metrics.get('avg_cpa', 0)) if metrics.get('avg_cpa') else None,
                        'roas': roas,
                        'quality_score': float(metrics.get('avg_quality_score', 0)) if metrics.get('avg_quality_score') else None,
                        'engagement_rate': None,  # Platform-specific, handled separately
                        'avg_cpc': None,  # Not in summary
                        'avg_cpm': None,  # Not in summary
                        'reach': None,
                        'frequency': None,
                        'metrics_start_date': start_date.isoformat(),
                        'metrics_end_date': end_date.isoformat(),
                        'days_analyzed': int(metrics.get('days_active', 30)),
                        'learning_cycle_id': None,  # Will be set by meta-learning service
                        'source_agent': 'marketing_agent',
                        'performance_score': round(performance_score, 2),
                        'metadata': {
                            'summary_generated_at': datetime.utcnow().isoformat(),
                            'raw_metrics': metrics
                        }
                    }

                    telemetry_records.append(telemetry_record)

            # Bulk insert
            if telemetry_records:
                self.supabase.table('cil_ads_telemetry').insert(telemetry_records).execute()
                logger.info(
                    f"Ingested {len(telemetry_records)} ads telemetry records for org {org_id}",
                    extra={
                        'org_id': org_id,
                        'records_count': len(telemetry_records),
                        'platforms': list(set(r['platform'] for r in telemetry_records))
                    }
                )

            return len(telemetry_records)

        except Exception as e:
            logger.error(f"Error ingesting ads telemetry: {e}", exc_info=True)
            return 0


# Singleton instance
_cil_telemetry_service: Optional[CILTelemetryService] = None


def get_cil_telemetry_service() -> CILTelemetryService:
    """Get singleton CIL telemetry service instance"""
    global _cil_telemetry_service
    if _cil_telemetry_service is None:
        _cil_telemetry_service = CILTelemetryService()
    return _cil_telemetry_service
