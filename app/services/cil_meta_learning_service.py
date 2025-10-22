"""
CIL Meta-Learning Service
Implements 4 core optimization algorithms:
1. Confidence Optimizer - Adjusts confidence thresholds based on success rates
2. Module Router Optimizer - Learns which modules work best per intent
3. Risk Level Adjuster - Calibrates risk sensitivity based on outcomes
4. Agent Performance Optimizer - Learns agent preferences and coordination
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from app.core.logging import logger
from app.db import get_supabase_admin_client
from app.services.cil_policy_service import get_cil_policy_service
from app.services.cil_ads_optimizer import get_cil_ads_optimizer  # Phase 6, Step 2


class CILMetaLearningService:
    """
    Meta-learning engine that analyzes telemetry and generates optimized policies
    """

    def __init__(self):
        self.supabase = get_supabase_admin_client()
        self.policy_service = get_cil_policy_service()
        self.ads_optimizer = get_cil_ads_optimizer()  # Phase 6, Step 2

        # Safety guardrails
        self.MIN_CONFIDENCE_THRESHOLD = 0.4
        self.MAX_RISK_SENSITIVITY = 0.98
        self.MIN_SAMPLE_SIZE = 20  # Minimum telemetry records needed

    async def run_learning_cycle(
        self,
        org_id: str,
        days_back: int = 7,
        algorithms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete meta-learning cycle

        Args:
            org_id: Organization ID
            days_back: Days of historical data to analyze
            algorithms: Which algorithms to run (None = all)

        Returns:
            Learning cycle results
        """
        if algorithms is None:
            algorithms = [
                'confidence_optimizer',
                'module_router',
                'risk_adjuster',
                'agent_optimizer'
            ]

        # Create learning cycle record
        cycle_id = await self._create_learning_cycle(org_id, days_back, algorithms)

        try:
            # Get telemetry data
            telemetry = await self._fetch_telemetry(org_id, days_back)

            if len(telemetry) < self.MIN_SAMPLE_SIZE:
                await self._update_learning_cycle(cycle_id, 'failed', {
                    'error': f'Insufficient data: {len(telemetry)} records (minimum {self.MIN_SAMPLE_SIZE})'
                })
                return {'success': False, 'error': 'Insufficient data'}

            # Get current policy
            current_policy = self.policy_service.get_active_policy(org_id)
            if not current_policy:
                await self._update_learning_cycle(cycle_id, 'failed', {'error': 'No active policy'})
                return {'success': False, 'error': 'No active policy'}

            current_config = current_policy['policy_config']
            findings = {}

            # Run each algorithm
            if 'confidence_optimizer' in algorithms:
                findings['confidence_optimizer'] = await self._optimize_confidence(telemetry, current_config)

            if 'module_router' in algorithms:
                findings['module_router'] = await self._optimize_module_routing(telemetry, current_config)

            if 'risk_adjuster' in algorithms:
                findings['risk_adjuster'] = await self._adjust_risk_levels(telemetry, current_config)

            if 'agent_optimizer' in algorithms:
                findings['agent_optimizer'] = await self._optimize_agent_selection(telemetry, current_config)

            # Phase 6, Step 2: Run ads optimization algorithms
            ads_proposals = await self.ads_optimizer.generate_all_proposals(org_id, cycle_id)

            # Store ads proposals in cil_policy_proposals table
            ads_proposals_created = 0
            if ads_proposals:
                for proposal in ads_proposals:
                    try:
                        self.supabase.table('cil_policy_proposals').insert(proposal).execute()
                        ads_proposals_created += 1
                    except Exception as e:
                        logger.error(f"Error inserting ads proposal: {e}")

            # Generate new policy if improvements found
            new_policy_config, change_summary = await self._merge_findings(
                current_config,
                findings
            )

            policies_generated = 0
            proposals_created = ads_proposals_created  # Start with ads proposals

            if change_summary:
                # Determine if this is a major or minor change
                is_major_change = self._is_major_change(change_summary)

                # Create proposal
                proposal_id = await self._create_policy_proposal(
                    org_id=org_id,
                    current_policy_id=current_policy['id'],
                    proposed_config=new_policy_config,
                    change_summary=change_summary,
                    learning_cycle_id=cycle_id,
                    is_major=is_major_change
                )

                if proposal_id:
                    proposals_created = 1

                    # If minor change, also create policy (will auto-apply after timeout)
                    if not is_major_change:
                        policies_generated = 1

            # Update learning cycle
            await self._update_learning_cycle(cycle_id, 'completed', {
                'findings': findings,
                'policies_generated': policies_generated,
                'proposals_created': proposals_created,
                'telemetry_records_analyzed': len(telemetry)
            })

            logger.info(
                f"CIL learning cycle completed for org {org_id}",
                extra={
                    'cycle_id': cycle_id,
                    'proposals': proposals_created,
                    'algorithms': algorithms
                }
            )

            return {
                'success': True,
                'cycle_id': cycle_id,
                'findings': findings,
                'proposals_created': proposals_created,
                'policies_generated': policies_generated
            }

        except Exception as e:
            logger.error(f"Error in learning cycle: {e}", exc_info=True)
            await self._update_learning_cycle(cycle_id, 'failed', {'error': str(e)})
            return {'success': False, 'error': str(e)}

    async def _optimize_confidence(
        self,
        telemetry: List[Dict],
        current_config: Dict
    ) -> Dict[str, Any]:
        """
        Algorithm 1: Confidence Optimizer

        Analyzes correlation between confidence scores and actual success rates.
        Adjusts confidence thresholds to maximize success rate.
        """
        try:
            current_thresholds = current_config.get('confidence_thresholds', {})
            recommendations = {}

            # Group by intent type
            by_intent = defaultdict(list)
            for record in telemetry:
                if record.get('intent_type') and record.get('confidence_score') is not None:
                    by_intent[record['intent_type']].append(record)

            for intent, records in by_intent.items():
                if len(records) < 10:  # Need sufficient data
                    continue

                # Calculate success rate at different confidence thresholds
                threshold_results = {}

                for threshold in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
                    above_threshold = [r for r in records if r['confidence_score'] >= threshold]

                    if len(above_threshold) > 0:
                        successes = sum(1 for r in above_threshold if r['outcome'] == 'success')
                        success_rate = successes / len(above_threshold)
                        threshold_results[threshold] = {
                            'success_rate': success_rate,
                            'sample_size': len(above_threshold)
                        }

                # Find optimal threshold (highest success rate with reasonable sample size)
                optimal_threshold = current_thresholds.get(intent, 0.7)
                optimal_success_rate = 0

                for threshold, results in threshold_results.items():
                    if results['sample_size'] >= 5:  # Minimum sample
                        if results['success_rate'] > optimal_success_rate:
                            optimal_success_rate = results['success_rate']
                            optimal_threshold = threshold

                current_threshold = current_thresholds.get(intent, 0.7)

                # Only recommend change if improvement > 5% and within guardrails
                if (
                    abs(optimal_threshold - current_threshold) > 0.05 and
                    optimal_threshold >= self.MIN_CONFIDENCE_THRESHOLD
                ):
                    recommendations[intent] = {
                        'current_threshold': current_threshold,
                        'recommended_threshold': optimal_threshold,
                        'expected_success_rate': round(optimal_success_rate, 3),
                        'confidence': 0.85,
                        'reasoning': f"Success rate improves to {optimal_success_rate:.1%} at {optimal_threshold} threshold"
                    }

            return recommendations

        except Exception as e:
            logger.error(f"Error in confidence optimizer: {e}", exc_info=True)
            return {}

    async def _optimize_module_routing(
        self,
        telemetry: List[Dict],
        current_config: Dict
    ) -> Dict[str, Any]:
        """
        Algorithm 2: Module Router Optimizer

        Learns which knowledge modules (RAG, KG, Memory, Insight) perform best
        for different intent types.
        """
        try:
            current_weights = current_config.get('module_weights', {})
            recommendations = {}

            # Group by intent type
            by_intent = defaultdict(list)
            for record in telemetry:
                if record.get('intent_type') and record.get('modules_used'):
                    by_intent[record['intent_type']].append(record)

            for intent, records in by_intent.items():
                if len(records) < 10:
                    continue

                # Calculate success rate with/without each module
                module_performance = {}

                for module in ['rag', 'kg', 'memory', 'insight']:
                    with_module = [r for r in records if module in (r.get('modules_used') or [])]
                    without_module = [r for r in records if module not in (r.get('modules_used') or [])]

                    if with_module and without_module:
                        with_success = sum(1 for r in with_module if r['outcome'] == 'success') / len(with_module)
                        without_success = sum(1 for r in without_module if r['outcome'] == 'success') / len(without_module)

                        improvement = with_success - without_success

                        module_performance[module] = {
                            'success_with': round(with_success, 3),
                            'success_without': round(without_success, 3),
                            'improvement': round(improvement, 3),
                            'sample_size': len(with_module)
                        }

                # Generate recommendations
                intent_recommendations = {}
                current_intent_weights = current_weights.get(intent, {})

                for module, perf in module_performance.items():
                    if perf['sample_size'] >= 5 and abs(perf['improvement']) > 0.1:  # 10% improvement threshold
                        current_weight = current_intent_weights.get(module, 0.5)

                        # Adjust weight based on improvement
                        if perf['improvement'] > 0:
                            # Module helps, increase weight
                            new_weight = min(1.0, current_weight + 0.1)
                        else:
                            # Module hurts, decrease weight
                            new_weight = max(0.0, current_weight - 0.1)

                        if abs(new_weight - current_weight) > 0.05:
                            intent_recommendations[module] = {
                                'current_weight': current_weight,
                                'recommended_weight': round(new_weight, 2),
                                'improvement': perf['improvement'],
                                'reasoning': f"{module.upper()} {'improves' if perf['improvement'] > 0 else 'degrades'} success rate by {abs(perf['improvement']):.1%}"
                            }

                if intent_recommendations:
                    recommendations[intent] = intent_recommendations

            return recommendations

        except Exception as e:
            logger.error(f"Error in module router optimizer: {e}", exc_info=True)
            return {}

    async def _adjust_risk_levels(
        self,
        telemetry: List[Dict],
        current_config: Dict
    ) -> Dict[str, Any]:
        """
        Algorithm 3: Risk Level Adjuster

        Calibrates risk sensitivity based on approval rates and false positives.
        """
        try:
            current_sensitivity = current_config.get('risk_sensitivity', {})
            recommendations = {}

            # Analyze approval decisions by risk level
            by_risk = defaultdict(list)
            for record in telemetry:
                if record.get('risk_level') and record.get('required_approval'):
                    by_risk[record['risk_level']].append(record)

            for risk_level, records in by_risk.items():
                if len(records) < 5:
                    continue

                # Calculate metrics
                total = len(records)
                approved = sum(1 for r in records if r.get('approval_decision') == 'approved')
                rejected = sum(1 for r in records if r.get('approval_decision') == 'rejected')
                successes_after_approval = sum(1 for r in records
                                               if r.get('approval_decision') == 'approved' and r.get('outcome') == 'success')

                approval_rate = approved / total if total > 0 else 0
                false_positive_rate = (approved - successes_after_approval) / approved if approved > 0 else 0

                current_sens = current_sensitivity.get(risk_level, 0.5)

                # Adjust sensitivity
                # If too many approvals (>90%), we're being too cautious
                # If too many false positives (>20%), we're not cautious enough
                new_sensitivity = current_sens

                if approval_rate > 0.9 and false_positive_rate < 0.1:
                    # Too cautious, decrease sensitivity
                    new_sensitivity = max(0.1, current_sens - 0.1)
                    reasoning = f"High approval rate ({approval_rate:.1%}) with low false positives - can be less strict"

                elif false_positive_rate > 0.2:
                    # Too many bad approvals, increase sensitivity
                    new_sensitivity = min(self.MAX_RISK_SENSITIVITY, current_sens + 0.1)
                    reasoning = f"High false positive rate ({false_positive_rate:.1%}) - need stricter checking"
                else:
                    reasoning = f"Risk level {risk_level} is well-calibrated"

                if abs(new_sensitivity - current_sens) > 0.05:
                    recommendations[risk_level] = {
                        'current_sensitivity': current_sens,
                        'recommended_sensitivity': round(new_sensitivity, 2),
                        'approval_rate': round(approval_rate, 3),
                        'false_positive_rate': round(false_positive_rate, 3),
                        'reasoning': reasoning
                    }

            return recommendations

        except Exception as e:
            logger.error(f"Error in risk adjuster: {e}", exc_info=True)
            return {}

    async def _optimize_agent_selection(
        self,
        telemetry: List[Dict],
        current_config: Dict
    ) -> Dict[str, Any]:
        """
        Algorithm 4: Agent Performance Optimizer

        Learns which agents perform best and optimal coordination patterns.
        """
        try:
            current_preferences = current_config.get('agent_preferences', {})
            recommendations = {}

            # Analyze agent performance
            by_agent = defaultdict(list)
            for record in telemetry:
                if record.get('agent_type'):
                    by_agent[record['agent_type']].append(record)

            agent_metrics = {}

            for agent, records in by_agent.items():
                if len(records) < 5:
                    continue

                successes = sum(1 for r in records if r['outcome'] == 'success')
                success_rate = successes / len(records)

                # Average response time
                response_times = [r['response_time_ms'] for r in records if r.get('response_time_ms')]
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0

                agent_metrics[agent] = {
                    'success_rate': round(success_rate, 3),
                    'avg_response_time_ms': int(avg_response_time),
                    'total_executions': len(records)
                }

            # Generate recommendations for agent priorities
            # Higher success rate + faster response = higher priority
            sorted_agents = sorted(
                agent_metrics.items(),
                key=lambda x: (x[1]['success_rate'], -x[1]['avg_response_time_ms']),
                reverse=True
            )

            for priority, (agent, metrics) in enumerate(sorted_agents, start=1):
                current_pref = current_preferences.get(agent, {})
                current_priority = current_pref.get('priority', 999)

                if current_priority != priority:
                    recommendations[agent] = {
                        'current_priority': current_priority,
                        'recommended_priority': priority,
                        'success_rate': metrics['success_rate'],
                        'avg_response_time_ms': metrics['avg_response_time_ms'],
                        'reasoning': f"Agent ranks #{priority} based on {metrics['success_rate']:.1%} success rate"
                    }

            return recommendations

        except Exception as e:
            logger.error(f"Error in agent optimizer: {e}", exc_info=True)
            return {}

    async def _merge_findings(
        self,
        current_config: Dict,
        findings: Dict[str, Any]
    ) -> Tuple[Dict, Dict]:
        """
        Merge all algorithm findings into a new policy configuration
        """
        new_config = json.loads(json.dumps(current_config))  # Deep copy
        change_summary = {}

        # Apply confidence optimizer recommendations
        if 'confidence_optimizer' in findings and findings['confidence_optimizer']:
            for intent, recommendation in findings['confidence_optimizer'].items():
                if 'confidence_thresholds' not in new_config:
                    new_config['confidence_thresholds'] = {}

                new_config['confidence_thresholds'][intent] = recommendation['recommended_threshold']

                change_summary[f'confidence_thresholds.{intent}'] = {
                    'old': recommendation['current_threshold'],
                    'new': recommendation['recommended_threshold'],
                    'reason': recommendation['reasoning']
                }

        # Apply module router recommendations
        if 'module_router' in findings and findings['module_router']:
            for intent, modules in findings['module_router'].items():
                if 'module_weights' not in new_config:
                    new_config['module_weights'] = {}
                if intent not in new_config['module_weights']:
                    new_config['module_weights'][intent] = {}

                for module, recommendation in modules.items():
                    new_config['module_weights'][intent][module] = recommendation['recommended_weight']

                    change_summary[f'module_weights.{intent}.{module}'] = {
                        'old': recommendation['current_weight'],
                        'new': recommendation['recommended_weight'],
                        'reason': recommendation['reasoning']
                    }

        # Apply risk adjuster recommendations
        if 'risk_adjuster' in findings and findings['risk_adjuster']:
            for risk_level, recommendation in findings['risk_adjuster'].items():
                if 'risk_sensitivity' not in new_config:
                    new_config['risk_sensitivity'] = {}

                new_config['risk_sensitivity'][risk_level] = recommendation['recommended_sensitivity']

                change_summary[f'risk_sensitivity.{risk_level}'] = {
                    'old': recommendation['current_sensitivity'],
                    'new': recommendation['recommended_sensitivity'],
                    'reason': recommendation['reasoning']
                }

        # Apply agent optimizer recommendations
        if 'agent_optimizer' in findings and findings['agent_optimizer']:
            for agent, recommendation in findings['agent_optimizer'].items():
                if 'agent_preferences' not in new_config:
                    new_config['agent_preferences'] = {}
                if agent not in new_config['agent_preferences']:
                    new_config['agent_preferences'][agent] = {}

                new_config['agent_preferences'][agent]['priority'] = recommendation['recommended_priority']

                change_summary[f'agent_preferences.{agent}.priority'] = {
                    'old': recommendation['current_priority'],
                    'new': recommendation['recommended_priority'],
                    'reason': recommendation['reasoning']
                }

        return new_config, change_summary

    def _is_major_change(self, change_summary: Dict) -> bool:
        """
        Determine if changes are major (require approval) or minor (auto-apply)

        Major = affects >20% of queries or changes risk/confidence significantly
        """
        # Count types of changes
        confidence_changes = sum(1 for k in change_summary.keys() if k.startswith('confidence_thresholds'))
        risk_changes = sum(1 for k in change_summary.keys() if k.startswith('risk_sensitivity'))

        # Major if more than 3 confidence thresholds changed or any risk sensitivity changed
        if confidence_changes > 3 or risk_changes > 0:
            return True

        # Check magnitude of changes
        for change in change_summary.values():
            old = change.get('old', 0)
            new = change.get('new', 0)

            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                if abs(new - old) > 0.2:  # 20% change
                    return True

        return False

    async def _fetch_telemetry(self, org_id: str, days_back: int) -> List[Dict]:
        """Fetch telemetry data for analysis"""
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

            result = self.supabase.table('cil_telemetry')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff_date)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error fetching telemetry: {e}", exc_info=True)
            return []

    async def _create_learning_cycle(self, org_id: str, days_back: int, algorithms: List[str]) -> str:
        """Create a learning cycle record"""
        try:
            cycle_data = {
                'org_id': org_id,
                'cycle_type': 'scheduled',
                'algorithms_run': algorithms,
                'analysis_start_date': (datetime.utcnow() - timedelta(days=days_back)).isoformat(),
                'analysis_end_date': datetime.utcnow().isoformat(),
                'telemetry_records_analyzed': 0,
                'status': 'running'
            }

            result = self.supabase.table('cil_learning_cycles').insert(cycle_data).execute()

            return result.data[0]['id'] if result.data else None

        except Exception as e:
            logger.error(f"Error creating learning cycle: {e}", exc_info=True)
            return None

    async def _update_learning_cycle(self, cycle_id: str, status: str, data: Dict):
        """Update learning cycle with results"""
        try:
            update_data = {
                'status': status,
                'completed_at': datetime.utcnow().isoformat(),
                **data
            }

            self.supabase.table('cil_learning_cycles')\
                .update(update_data)\
                .eq('id', cycle_id)\
                .execute()

        except Exception as e:
            logger.error(f"Error updating learning cycle: {e}", exc_info=True)

    async def _create_policy_proposal(
        self,
        org_id: str,
        current_policy_id: str,
        proposed_config: Dict,
        change_summary: Dict,
        learning_cycle_id: str,
        is_major: bool
    ) -> Optional[str]:
        """Create a policy proposal"""
        try:
            # Calculate estimated impact
            impact = {
                'affected_query_percentage': min(len(change_summary) * 0.1, 1.0),
                'total_changes': len(change_summary)
            }

            proposal_data = {
                'org_id': org_id,
                'current_policy_id': current_policy_id,
                'proposed_policy_config': json.dumps(proposed_config),
                'change_type': 'major' if is_major else 'minor',
                'change_impact': json.dumps(impact),
                'change_details': json.dumps(change_summary),
                'confidence_in_change': 0.85,
                'learning_cycle_id': learning_cycle_id,
                'telemetry_sample_size': 100,  # TODO: actual value
                'priority': 'high' if is_major else 'normal',
                'auto_apply_after': (datetime.utcnow() + timedelta(hours=24)).isoformat() if not is_major else None
            }

            result = self.supabase.table('cil_policy_proposals').insert(proposal_data).execute()

            return result.data[0]['id'] if result.data else None

        except Exception as e:
            logger.error(f"Error creating policy proposal: {e}", exc_info=True)
            return None


# Singleton instance
_cil_meta_learning_service: Optional[CILMetaLearningService] = None


def get_cil_meta_learning_service() -> CILMetaLearningService:
    """Get singleton CIL meta-learning service instance"""
    global _cil_meta_learning_service
    if _cil_meta_learning_service is None:
        _cil_meta_learning_service = CILMetaLearningService()
    return _cil_meta_learning_service
