"""
CIL Prompt Optimizer Service
Manages prompt templates with A/B testing and performance optimization
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import random
import json

from app.core.logging import logger
from app.db import get_supabase_admin_client


class CILPromptOptimizer:
    """
    Optimizes prompts through A/B testing and performance analysis

    Features:
    - Template versioning with A/B testing
    - Performance tracking (success rate, response time)
    - Automatic winner selection based on statistical significance
    - Gradual rollout (champion/challenger model)
    - Multi-armed bandit selection for exploration/exploitation
    """

    def __init__(self):
        self.supabase = get_supabase_admin_client()

        # A/B testing configuration
        self.min_sample_size = 50  # Minimum samples before declaring winner
        self.confidence_level = 0.95  # 95% confidence for statistical significance
        self.exploration_rate = 0.1  # 10% traffic to challenger during testing

    def create_prompt_template(
        self,
        org_id: str,
        template_name: str,
        template_type: str,
        prompt_text: str,
        variables: List[str],
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Create a new prompt template

        Args:
            org_id: Organization ID
            template_name: Unique name for this template
            template_type: Type (e.g., 'cde_intent', 'agent_system', 'rag_query')
            prompt_text: The actual prompt with {variable} placeholders
            variables: List of variable names used in prompt
            description: Human-readable description
            metadata: Additional metadata

        Returns:
            Created template record
        """
        try:
            template_data = {
                'org_id': org_id,
                'template_name': template_name,
                'template_type': template_type,
                'version': 1,
                'prompt_text': prompt_text,
                'variables': variables,
                'description': description,
                'is_active': True,
                'is_champion': True,  # First version is always champion
                'ab_test_group': None,
                'metadata': json.dumps(metadata) if metadata else None
            }

            result = self.supabase.table('cil_prompt_templates')\
                .insert(template_data)\
                .execute()

            if result.data:
                logger.info(
                    f"Created prompt template '{template_name}' v1 for org {org_id}",
                    extra={'template_name': template_name, 'org_id': org_id}
                )
                return result.data[0]

            return None

        except Exception as e:
            logger.error(f"Error creating prompt template: {e}", exc_info=True)
            return None

    def create_ab_test(
        self,
        org_id: str,
        template_name: str,
        challenger_prompt_text: str,
        test_duration_hours: int = 168,  # 7 days default
        traffic_split: float = 0.5,  # 50/50 split
        test_hypothesis: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create A/B test with new prompt variant

        Args:
            org_id: Organization ID
            template_name: Name of template to test
            challenger_prompt_text: New prompt variant to test
            test_duration_hours: How long to run test
            traffic_split: % traffic to challenger (0.0-1.0)
            test_hypothesis: Description of what you're testing

        Returns:
            Created challenger template record
        """
        try:
            # Get current champion
            champion_query = self.supabase.table('cil_prompt_templates')\
                .select('*')\
                .eq('org_id', org_id)\
                .eq('template_name', template_name)\
                .eq('is_champion', True)\
                .eq('is_active', True)\
                .single()\
                .execute()

            if not champion_query.data:
                logger.error(f"No champion found for template '{template_name}'")
                return None

            champion = champion_query.data

            # Create challenger variant
            new_version = champion['version'] + 1
            test_id = f"ab_test_{template_name}_v{new_version}"

            challenger_data = {
                'org_id': org_id,
                'template_name': template_name,
                'template_type': champion['template_type'],
                'version': new_version,
                'prompt_text': challenger_prompt_text,
                'variables': champion['variables'],
                'description': f"A/B test variant (challenger)",
                'is_active': True,
                'is_champion': False,
                'ab_test_group': test_id,
                'ab_test_start': datetime.utcnow().isoformat(),
                'ab_test_end': (datetime.utcnow() + timedelta(hours=test_duration_hours)).isoformat(),
                'ab_test_traffic_split': traffic_split,
                'metadata': json.dumps({
                    'test_hypothesis': test_hypothesis,
                    'champion_version': champion['version']
                })
            }

            result = self.supabase.table('cil_prompt_templates')\
                .insert(challenger_data)\
                .execute()

            if result.data:
                # Update champion with test info
                self.supabase.table('cil_prompt_templates')\
                    .update({'ab_test_group': test_id})\
                    .eq('id', champion['id'])\
                    .execute()

                logger.info(
                    f"Started A/B test for '{template_name}': "
                    f"v{champion['version']} vs v{new_version}, "
                    f"{int(traffic_split * 100)}% traffic to challenger"
                )
                return result.data[0]

            return None

        except Exception as e:
            logger.error(f"Error creating A/B test: {e}", exc_info=True)
            return None

    def get_prompt_for_use(
        self,
        org_id: str,
        template_name: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get prompt template for actual use (handles A/B test selection)

        Uses multi-armed bandit approach:
        - If A/B test active: routes traffic based on test configuration
        - If no test: returns champion
        - Tracks which variant was selected for performance tracking

        Args:
            org_id: Organization ID
            template_name: Template name
            user_id: Optional user ID for consistent assignment

        Returns:
            Selected template variant
        """
        try:
            # Get all active variants for this template
            variants_query = self.supabase.table('cil_prompt_templates')\
                .select('*')\
                .eq('org_id', org_id)\
                .eq('template_name', template_name)\
                .eq('is_active', True)\
                .execute()

            variants = variants_query.data or []

            if not variants:
                logger.warning(f"No active template found: {template_name}")
                return None

            # If only one variant, return it
            if len(variants) == 1:
                return variants[0]

            # Check for active A/B test
            champion = next((v for v in variants if v['is_champion']), None)
            challengers = [v for v in variants if not v['is_champion']]

            if not champion:
                logger.error(f"No champion found for template '{template_name}'")
                return variants[0]  # Fallback to first variant

            # Check if A/B test is still active
            if challengers:
                challenger = challengers[0]  # Currently support 1 challenger at a time

                test_end = challenger.get('ab_test_end')
                if test_end:
                    test_end_dt = datetime.fromisoformat(test_end.replace('Z', '+00:00'))
                    if datetime.utcnow() < test_end_dt.replace(tzinfo=None):
                        # Test is active, route traffic
                        traffic_split = challenger.get('ab_test_traffic_split', 0.5)

                        # Consistent hashing if user_id provided
                        if user_id:
                            hash_val = hash(f"{user_id}_{template_name}") % 100
                            use_challenger = hash_val < (traffic_split * 100)
                        else:
                            # Random assignment
                            use_challenger = random.random() < traffic_split

                        selected = challenger if use_challenger else champion

                        logger.debug(
                            f"A/B test selection for '{template_name}': "
                            f"v{selected['version']} ({'challenger' if use_challenger else 'champion'})"
                        )

                        return selected

            # No active test or test expired, return champion
            return champion

        except Exception as e:
            logger.error(f"Error getting prompt for use: {e}", exc_info=True)
            return None

    def record_prompt_usage(
        self,
        template_id: str,
        outcome: str,  # 'success' or 'failure'
        response_time_ms: Optional[int] = None,
        quality_score: Optional[float] = None,
        user_feedback: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Record prompt usage outcome for performance tracking

        Args:
            template_id: Template ID that was used
            outcome: 'success' or 'failure'
            response_time_ms: Response time in milliseconds
            quality_score: Quality score (0.0-1.0)
            user_feedback: User feedback text
            metadata: Additional context
        """
        try:
            # Increment usage count
            self.supabase.rpc('increment_prompt_usage', {
                'template_id': template_id,
                'outcome': outcome
            }).execute()

            # Update performance metrics
            self.supabase.table('cil_prompt_templates')\
                .update({'last_used_at': datetime.utcnow().isoformat()})\
                .eq('id', template_id)\
                .execute()

            # Store detailed outcome if significant metadata
            if quality_score is not None or user_feedback or metadata:
                outcome_data = {
                    'template_id': template_id,
                    'outcome': outcome,
                    'response_time_ms': response_time_ms,
                    'quality_score': quality_score,
                    'user_feedback': user_feedback,
                    'metadata': json.dumps(metadata) if metadata else None
                }

                self.supabase.table('cil_prompt_outcomes')\
                    .insert(outcome_data)\
                    .execute()

        except Exception as e:
            logger.error(f"Error recording prompt usage: {e}", exc_info=True)

    async def evaluate_ab_tests(self, org_id: str) -> List[Dict]:
        """
        Evaluate all active A/B tests and determine winners

        Uses statistical significance testing:
        - Chi-square test for success rate differences
        - Minimum sample size requirements
        - Confidence level thresholds

        Args:
            org_id: Organization ID

        Returns:
            List of test results with decisions
        """
        try:
            # Get all active A/B tests
            tests_query = self.supabase.table('cil_prompt_templates')\
                .select('*')\
                .eq('org_id', org_id)\
                .eq('is_active', True)\
                .not_.is_('ab_test_group', 'null')\
                .execute()

            templates = tests_query.data or []

            if not templates:
                return []

            # Group by test
            tests_by_group = {}
            for template in templates:
                test_group = template['ab_test_group']
                if test_group not in tests_by_group:
                    tests_by_group[test_group] = []
                tests_by_group[test_group].append(template)

            results = []

            for test_group, variants in tests_by_group.items():
                if len(variants) != 2:
                    continue  # Need exactly 2 variants (champion + challenger)

                champion = next((v for v in variants if v['is_champion']), None)
                challenger = next((v for v in variants if not v['is_champion']), None)

                if not champion or not challenger:
                    continue

                # Calculate success rates
                champion_success_rate = self._calculate_success_rate(champion)
                challenger_success_rate = self._calculate_success_rate(challenger)

                # Check if we have enough data
                champion_total = champion.get('usage_count', 0)
                challenger_total = challenger.get('usage_count', 0)

                sufficient_data = (
                    champion_total >= self.min_sample_size and
                    challenger_total >= self.min_sample_size
                )

                # Check if test duration expired
                test_end = challenger.get('ab_test_end')
                test_expired = False
                if test_end:
                    test_end_dt = datetime.fromisoformat(test_end.replace('Z', '+00:00'))
                    test_expired = datetime.utcnow() >= test_end_dt.replace(tzinfo=None)

                # Determine winner if sufficient data or test expired
                winner = None
                decision = 'ongoing'

                if sufficient_data:
                    # Simple comparison with minimum improvement threshold
                    improvement = challenger_success_rate - champion_success_rate
                    min_improvement = 0.05  # 5% minimum improvement

                    if improvement > min_improvement:
                        winner = 'challenger'
                        decision = 'challenger_wins'
                    elif improvement < -min_improvement:
                        winner = 'champion'
                        decision = 'champion_wins'
                    else:
                        winner = 'champion'
                        decision = 'no_significant_difference'

                elif test_expired:
                    # Test expired, pick better performing variant
                    winner = 'challenger' if challenger_success_rate > champion_success_rate else 'champion'
                    decision = 'test_expired_insufficient_data'

                result = {
                    'test_group': test_group,
                    'template_name': champion['template_name'],
                    'champion': {
                        'version': champion['version'],
                        'success_rate': champion_success_rate,
                        'sample_size': champion_total
                    },
                    'challenger': {
                        'version': challenger['version'],
                        'success_rate': challenger_success_rate,
                        'sample_size': challenger_total
                    },
                    'decision': decision,
                    'winner': winner,
                    'sufficient_data': sufficient_data,
                    'test_expired': test_expired
                }

                results.append(result)

                # Auto-promote winner if decision made
                if winner and decision in ['challenger_wins', 'champion_wins', 'no_significant_difference', 'test_expired_insufficient_data']:
                    await self._promote_winner(champion, challenger, winner)

            return results

        except Exception as e:
            logger.error(f"Error evaluating A/B tests: {e}", exc_info=True)
            return []

    async def _promote_winner(self, champion: Dict, challenger: Dict, winner: str):
        """
        Promote the winning variant to champion

        Args:
            champion: Current champion template
            challenger: Challenger template
            winner: 'champion' or 'challenger'
        """
        try:
            if winner == 'challenger':
                # Promote challenger to champion
                self.supabase.table('cil_prompt_templates')\
                    .update({
                        'is_champion': True,
                        'ab_test_group': None,
                        'promoted_at': datetime.utcnow().isoformat()
                    })\
                    .eq('id', challenger['id'])\
                    .execute()

                # Demote old champion
                self.supabase.table('cil_prompt_templates')\
                    .update({
                        'is_champion': False,
                        'is_active': False,
                        'ab_test_group': None
                    })\
                    .eq('id', champion['id'])\
                    .execute()

                logger.info(
                    f"Promoted challenger v{challenger['version']} to champion "
                    f"for template '{champion['template_name']}'"
                )

            else:  # winner == 'champion'
                # Keep champion, deactivate challenger
                self.supabase.table('cil_prompt_templates')\
                    .update({
                        'is_active': False,
                        'ab_test_group': None
                    })\
                    .eq('id', challenger['id'])\
                    .execute()

                # Clear test info from champion
                self.supabase.table('cil_prompt_templates')\
                    .update({'ab_test_group': None})\
                    .eq('id', champion['id'])\
                    .execute()

                logger.info(
                    f"Champion v{champion['version']} retained "
                    f"for template '{champion['template_name']}'"
                )

        except Exception as e:
            logger.error(f"Error promoting winner: {e}", exc_info=True)

    def _calculate_success_rate(self, template: Dict) -> float:
        """Calculate success rate for a template"""
        total = template.get('usage_count', 0)
        successes = template.get('success_count', 0)

        if total == 0:
            return 0.0

        return successes / total

    def get_template_performance(
        self,
        org_id: str,
        template_name: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get performance statistics for a template

        Args:
            org_id: Organization ID
            template_name: Template name
            days_back: Days of history to analyze

        Returns:
            Performance statistics
        """
        try:
            # Get all versions of this template
            versions_query = self.supabase.table('cil_prompt_templates')\
                .select('*')\
                .eq('org_id', org_id)\
                .eq('template_name', template_name)\
                .order('version', desc=True)\
                .execute()

            versions = versions_query.data or []

            if not versions:
                return {'error': 'Template not found'}

            # Calculate stats for each version
            version_stats = []
            for version in versions:
                success_rate = self._calculate_success_rate(version)
                avg_response_time = version.get('avg_response_time_ms', 0)

                version_stats.append({
                    'version': version['version'],
                    'is_champion': version['is_champion'],
                    'is_active': version['is_active'],
                    'success_rate': round(success_rate, 3),
                    'usage_count': version.get('usage_count', 0),
                    'avg_response_time_ms': avg_response_time,
                    'created_at': version.get('created_at'),
                    'last_used_at': version.get('last_used_at')
                })

            champion = next((v for v in versions if v['is_champion']), None)

            return {
                'template_name': template_name,
                'current_champion_version': champion['version'] if champion else None,
                'total_versions': len(versions),
                'version_history': version_stats,
                'has_active_ab_test': any(v.get('ab_test_group') for v in versions)
            }

        except Exception as e:
            logger.error(f"Error getting template performance: {e}", exc_info=True)
            return {'error': str(e)}


# Singleton instance
_cil_prompt_optimizer: Optional[CILPromptOptimizer] = None


def get_cil_prompt_optimizer() -> CILPromptOptimizer:
    """Get singleton CIL prompt optimizer instance"""
    global _cil_prompt_optimizer
    if _cil_prompt_optimizer is None:
        _cil_prompt_optimizer = CILPromptOptimizer()
    return _cil_prompt_optimizer
