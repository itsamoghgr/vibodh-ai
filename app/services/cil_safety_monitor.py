"""
CIL Safety & Evaluation Monitor
Monitors system safety, detects anomalies, and evaluates CIL impact
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import json
from statistics import mean, stdev

from app.core.logging import logger
from app.db import get_supabase_admin_client


class CILSafetyMonitor:
    """
    Monitors CIL safety and evaluates impact of policy changes

    Responsibilities:
    - Detect anomalies in system behavior
    - Monitor policy change impact (before/after comparison)
    - Alert on safety violations
    - Track drift in key metrics
    - Evaluate CIL effectiveness
    """

    def __init__(self):
        self.supabase = get_supabase_admin_client()

        # Safety thresholds
        self.max_confidence_drift = 0.2  # 20% drift triggers alert
        self.max_error_rate_increase = 0.15  # 15% error rate increase = bad
        self.min_success_rate = 0.6  # 60% minimum success rate
        self.max_response_time_increase = 1.5  # 1.5x response time = concerning

    async def run_safety_check(self, org_id: str) -> Dict[str, Any]:
        """
        Run comprehensive safety check

        Args:
            org_id: Organization ID

        Returns:
            Safety check results with alerts
        """
        try:
            logger.info(f"Starting safety check for org {org_id}")

            results = {
                'org_id': org_id,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'safe',
                'alerts': [],
                'warnings': [],
                'metrics': {},
                'policy_impact': {}
            }

            # 1. Check for anomalies in recent telemetry
            anomalies = await self._detect_anomalies(org_id)
            if anomalies['critical']:
                results['status'] = 'critical'
                results['alerts'].extend(anomalies['critical'])
            if anomalies['warnings']:
                results['warnings'].extend(anomalies['warnings'])

            # 2. Evaluate recent policy changes
            policy_impact = await self._evaluate_policy_impact(org_id)
            results['policy_impact'] = policy_impact

            if policy_impact.get('negative_impact'):
                results['status'] = 'warning' if results['status'] == 'safe' else results['status']
                results['warnings'].append(
                    f"Recent policy change has negative impact: {policy_impact['summary']}"
                )

            # 3. Check guardrail violations
            violations = await self._check_guardrail_violations(org_id)
            if violations:
                results['status'] = 'critical'
                results['alerts'].extend(violations)

            # 4. Calculate key safety metrics
            metrics = await self._calculate_safety_metrics(org_id)
            results['metrics'] = metrics

            # Store safety check record
            await self._store_safety_check(results)

            logger.info(
                f"Safety check completed for org {org_id}: "
                f"Status={results['status']}, "
                f"Alerts={len(results['alerts'])}, "
                f"Warnings={len(results['warnings'])}"
            )

            return results

        except Exception as e:
            logger.error(f"Error in safety check: {e}", exc_info=True)
            return {
                'org_id': org_id,
                'status': 'error',
                'error': str(e)
            }

    async def _detect_anomalies(self, org_id: str) -> Dict[str, List[str]]:
        """
        Detect anomalies in recent system behavior

        Returns:
            Dict with 'critical' and 'warnings' lists
        """
        critical = []
        warnings = []

        try:
            # Get last 24 hours of telemetry
            cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()

            telemetry_query = self.supabase.table('cil_telemetry')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff)\
                .execute()

            telemetry = telemetry_query.data or []

            if len(telemetry) < 10:
                warnings.append("Insufficient data for anomaly detection (< 10 records in 24h)")
                return {'critical': critical, 'warnings': warnings}

            # Calculate success rate
            success_count = sum(1 for t in telemetry if t.get('outcome') == 'success')
            success_rate = success_count / len(telemetry)

            if success_rate < self.min_success_rate:
                critical.append(
                    f"Success rate critically low: {success_rate:.1%} "
                    f"(threshold: {self.min_success_rate:.1%})"
                )

            # Check for confidence drift
            confidence_scores = [
                t.get('confidence_score')
                for t in telemetry
                if t.get('confidence_score') is not None
            ]

            if len(confidence_scores) >= 10:
                avg_confidence = mean(confidence_scores)
                std_confidence = stdev(confidence_scores)

                if std_confidence > self.max_confidence_drift:
                    warnings.append(
                        f"High confidence variability detected: σ={std_confidence:.3f} "
                        f"(threshold: {self.max_confidence_drift})"
                    )

            # Check for response time spikes
            response_times = [
                t.get('response_time_ms')
                for t in telemetry
                if t.get('response_time_ms') is not None
            ]

            if len(response_times) >= 10:
                recent_times = response_times[-10:]
                older_times = response_times[:-10] if len(response_times) > 10 else recent_times

                avg_recent = mean(recent_times)
                avg_older = mean(older_times) if older_times else avg_recent

                if avg_older > 0 and (avg_recent / avg_older) > self.max_response_time_increase:
                    warnings.append(
                        f"Response time increased significantly: "
                        f"{avg_recent:.0f}ms (was {avg_older:.0f}ms)"
                    )

            # Check for unusual error patterns
            error_count = sum(1 for t in telemetry if t.get('outcome') == 'failure')
            error_rate = error_count / len(telemetry)

            if error_rate > 0.3:  # 30% error rate
                critical.append(f"High error rate detected: {error_rate:.1%}")
            elif error_rate > 0.15:  # 15% error rate
                warnings.append(f"Elevated error rate: {error_rate:.1%}")

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}", exc_info=True)
            warnings.append(f"Anomaly detection error: {str(e)}")

        return {'critical': critical, 'warnings': warnings}

    async def _evaluate_policy_impact(self, org_id: str) -> Dict[str, Any]:
        """
        Evaluate impact of recent policy changes

        Compares metrics before and after policy change
        """
        try:
            # Get most recent policy change
            recent_policy_query = self.supabase.table('cil_policies')\
                .select('*')\
                .eq('org_id', org_id)\
                .eq('is_active', True)\
                .single()\
                .execute()

            if not recent_policy_query.data:
                return {'message': 'No active policy found'}

            active_policy = recent_policy_query.data
            activated_at = active_policy.get('activated_at')

            if not activated_at:
                return {'message': 'Policy not yet activated'}

            activated_dt = datetime.fromisoformat(activated_at.replace('Z', '+00:00'))
            time_since_activation = datetime.utcnow() - activated_dt.replace(tzinfo=None)

            # Need at least 24 hours of data after activation
            if time_since_activation.total_seconds() < 86400:  # 24 hours
                return {
                    'message': 'Insufficient time since activation',
                    'hours_since_activation': time_since_activation.total_seconds() / 3600
                }

            # Compare 24h before vs 24h after activation
            before_start = (activated_dt - timedelta(hours=48)).replace(tzinfo=None).isoformat()
            before_end = activated_dt.replace(tzinfo=None).isoformat()
            after_start = activated_dt.replace(tzinfo=None).isoformat()
            after_end = (activated_dt + timedelta(hours=24)).replace(tzinfo=None).isoformat()

            # Get before metrics
            before_query = self.supabase.table('cil_telemetry')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', before_start)\
                .lt('created_at', before_end)\
                .execute()

            before_data = before_query.data or []

            # Get after metrics
            after_query = self.supabase.table('cil_telemetry')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', after_start)\
                .lt('created_at', after_end)\
                .execute()

            after_data = after_query.data or []

            if len(before_data) < 5 or len(after_data) < 5:
                return {'message': 'Insufficient data for comparison'}

            # Calculate metrics
            before_metrics = self._calculate_metrics(before_data)
            after_metrics = self._calculate_metrics(after_data)

            # Determine impact
            success_rate_change = after_metrics['success_rate'] - before_metrics['success_rate']
            response_time_change = after_metrics['avg_response_time'] - before_metrics['avg_response_time']
            response_time_pct_change = (
                (response_time_change / before_metrics['avg_response_time'])
                if before_metrics['avg_response_time'] > 0 else 0
            )

            negative_impact = (
                success_rate_change < -0.05 or  # 5% drop in success rate
                response_time_pct_change > 0.3   # 30% increase in response time
            )

            return {
                'policy_version': active_policy['version'],
                'activated_at': activated_at,
                'hours_active': time_since_activation.total_seconds() / 3600,
                'before': before_metrics,
                'after': after_metrics,
                'changes': {
                    'success_rate': success_rate_change,
                    'avg_response_time': response_time_change,
                    'response_time_pct': response_time_pct_change
                },
                'negative_impact': negative_impact,
                'summary': (
                    f"Success rate {'+' if success_rate_change >= 0 else ''}{success_rate_change:.1%}, "
                    f"Response time {'+' if response_time_change >= 0 else ''}{response_time_change:.0f}ms "
                    f"({response_time_pct_change:+.1%})"
                )
            }

        except Exception as e:
            logger.error(f"Error evaluating policy impact: {e}", exc_info=True)
            return {'error': str(e)}

    def _calculate_metrics(self, telemetry_data: List[Dict]) -> Dict[str, float]:
        """Calculate metrics from telemetry data"""
        if not telemetry_data:
            return {
                'success_rate': 0.0,
                'avg_response_time': 0.0,
                'avg_confidence': 0.0,
                'count': 0
            }

        success_count = sum(1 for t in telemetry_data if t.get('outcome') == 'success')
        success_rate = success_count / len(telemetry_data)

        response_times = [
            t.get('response_time_ms')
            for t in telemetry_data
            if t.get('response_time_ms') is not None
        ]
        avg_response_time = mean(response_times) if response_times else 0.0

        confidences = [
            t.get('confidence_score')
            for t in telemetry_data
            if t.get('confidence_score') is not None
        ]
        avg_confidence = mean(confidences) if confidences else 0.0

        return {
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'avg_confidence': avg_confidence,
            'count': len(telemetry_data)
        }

    async def _check_guardrail_violations(self, org_id: str) -> List[str]:
        """
        Check if active policy violates safety guardrails

        Returns:
            List of violation messages
        """
        violations = []

        try:
            # Get active policy
            policy_query = self.supabase.table('cil_policies')\
                .select('*')\
                .eq('org_id', org_id)\
                .eq('is_active', True)\
                .single()\
                .execute()

            if not policy_query.data:
                return violations

            policy = policy_query.data
            config = policy['policy_config']
            if isinstance(config, str):
                config = json.loads(config)

            # Check confidence thresholds
            confidence_thresholds = config.get('confidence_thresholds', {})
            for intent, threshold in confidence_thresholds.items():
                if threshold < 0.4:
                    violations.append(
                        f"Confidence threshold too low for {intent}: {threshold} (min: 0.4)"
                    )
                if threshold > 0.95:
                    violations.append(
                        f"Confidence threshold too high for {intent}: {threshold} (max: 0.95)"
                    )

            # Check risk sensitivity
            risk_sensitivity = config.get('risk_sensitivity', {})
            for level, sensitivity in risk_sensitivity.items():
                if sensitivity < 0.2:
                    violations.append(
                        f"Risk sensitivity too low for {level}: {sensitivity} (min: 0.2)"
                    )
                if sensitivity > 0.98:
                    violations.append(
                        f"Risk sensitivity too high for {level}: {sensitivity} (max: 0.98)"
                    )

            # Check module weights are in valid range
            module_weights = config.get('module_weights', {})
            for intent, weights in module_weights.items():
                for module, weight in weights.items():
                    if weight < 0 or weight > 1:
                        violations.append(
                            f"Module weight out of range for {intent}/{module}: {weight} (range: 0-1)"
                        )

        except Exception as e:
            logger.error(f"Error checking guardrails: {e}", exc_info=True)
            violations.append(f"Guardrail check error: {str(e)}")

        return violations

    async def _calculate_safety_metrics(self, org_id: str) -> Dict[str, Any]:
        """Calculate current safety metrics"""
        try:
            # Get last 7 days of telemetry
            cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()

            telemetry_query = self.supabase.table('cil_telemetry')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff)\
                .execute()

            telemetry = telemetry_query.data or []

            if not telemetry:
                return {'message': 'No data available'}

            metrics = self._calculate_metrics(telemetry)

            # Add additional safety metrics
            high_risk_count = sum(
                1 for t in telemetry
                if t.get('risk_level') in ['high', 'critical']
            )

            approval_count = sum(
                1 for t in telemetry
                if t.get('required_approval') is True
            )

            rejected_count = sum(
                1 for t in telemetry
                if t.get('outcome') == 'rejected'
            )

            metrics.update({
                'high_risk_queries': high_risk_count,
                'high_risk_rate': high_risk_count / len(telemetry) if telemetry else 0,
                'approval_required': approval_count,
                'approval_rate': approval_count / len(telemetry) if telemetry else 0,
                'rejection_count': rejected_count,
                'rejection_rate': rejected_count / approval_count if approval_count > 0 else 0,
                'period_days': 7
            })

            return metrics

        except Exception as e:
            logger.error(f"Error calculating safety metrics: {e}", exc_info=True)
            return {'error': str(e)}

    async def _store_safety_check(self, results: Dict[str, Any]):
        """Store safety check results for audit trail"""
        try:
            record = {
                'org_id': results['org_id'],
                'timestamp': results['timestamp'],
                'status': results['status'],
                'alert_count': len(results.get('alerts', [])),
                'warning_count': len(results.get('warnings', [])),
                'alerts': json.dumps(results.get('alerts', [])),
                'warnings': json.dumps(results.get('warnings', [])),
                'metrics': json.dumps(results.get('metrics', {})),
                'policy_impact': json.dumps(results.get('policy_impact', {}))
            }

            self.supabase.table('cil_safety_checks')\
                .insert(record)\
                .execute()

        except Exception as e:
            logger.error(f"Error storing safety check: {e}", exc_info=True)

    def get_safety_history(self, org_id: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get safety check history

        Args:
            org_id: Organization ID
            days_back: Days of history

        Returns:
            Safety check history and trends
        """
        try:
            cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

            checks_query = self.supabase.table('cil_safety_checks')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('timestamp', cutoff)\
                .order('timestamp', desc=True)\
                .execute()

            checks = checks_query.data or []

            if not checks:
                return {
                    'total_checks': 0,
                    'period_days': days_back
                }

            # Count by status
            status_counts = {}
            for check in checks:
                status = check.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1

            # Count alerts and warnings
            total_alerts = sum(check.get('alert_count', 0) for check in checks)
            total_warnings = sum(check.get('warning_count', 0) for check in checks)

            return {
                'total_checks': len(checks),
                'period_days': days_back,
                'status_breakdown': status_counts,
                'total_alerts': total_alerts,
                'total_warnings': total_warnings,
                'recent_checks': checks[:10]  # Last 10 checks
            }

        except Exception as e:
            logger.error(f"Error getting safety history: {e}", exc_info=True)
            return {'error': str(e)}


# Singleton instance
_cil_safety_monitor: Optional[CILSafetyMonitor] = None


def get_cil_safety_monitor() -> CILSafetyMonitor:
    """Get singleton safety monitor instance"""
    global _cil_safety_monitor
    if _cil_safety_monitor is None:
        _cil_safety_monitor = CILSafetyMonitor()
    return _cil_safety_monitor
