"""
CIL Ads Optimizer - Phase 6, Step 2
Lightweight rules engine that analyzes ad performance and generates optimization proposals

Implements 4 core algorithms:
1. Budget Reallocation - Shifts budget based on ROAS comparison
2. Low-Performer Pauser - Flags underperforming campaigns
3. High-Performer Cloner - Identifies campaigns worth replicating
4. Platform Preference Learner - Discovers platform-specific patterns
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import UUID
import uuid

from app.core.logging import logger
from app.db import get_supabase_admin_client


class CILAdsOptimizer:
    """
    Generates data-driven optimization proposals for ad campaigns.

    Uses telemetry data from cil_ads_telemetry to learn patterns and
    suggest budget adjustments, campaign pauses, and platform preferences.
    """

    def __init__(self):
        self.supabase = get_supabase_admin_client()

        # Optimization thresholds (configurable)
        self.thresholds = {
            "min_roas": 1.0,           # Below this = losing money
            "excellent_roas": 4.0,      # Above this = high performer
            "min_ctr": 0.5,             # Below this = poor engagement
            "excellent_ctr": 2.5,       # Above this = strong engagement
            "min_conversions": 50,      # Minimum for statistical significance
            "min_quality_score": 3.0,   # Google Ads quality threshold
            "roas_improvement_threshold": 1.5,  # Platform A must be 50% better than B
            "min_spend": 100.0,         # Minimum spend to consider for optimization
            "observation_days": 30      # Days of data to analyze
        }

    async def generate_all_proposals(
        self,
        org_id: str,
        learning_cycle_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run all 4 optimization algorithms and generate proposals.

        Args:
            org_id: Organization ID
            learning_cycle_id: Optional learning cycle ID to link proposals

        Returns:
            List of proposal dictionaries ready for insertion
        """
        logger.info(f"[ADS_OPTIMIZER] Generating optimization proposals for org {org_id}")

        all_proposals = []

        # Algorithm 1: Budget Reallocation
        budget_proposals = await self.optimize_budget_allocation(org_id, learning_cycle_id)
        all_proposals.extend(budget_proposals)

        # Algorithm 2: Low-Performer Detection
        pause_proposals = await self.detect_underperforming_campaigns(org_id, learning_cycle_id)
        all_proposals.extend(pause_proposals)

        # Algorithm 3: High-Performer Cloning
        clone_proposals = await self.identify_top_performers(org_id, learning_cycle_id)
        all_proposals.extend(clone_proposals)

        # Algorithm 4: Platform Preference Learning
        platform_insights = await self.learn_platform_preferences(org_id, learning_cycle_id)
        # Platform insights are stored in ai_meta_knowledge, not as proposals

        logger.info(
            f"[ADS_OPTIMIZER] Generated {len(all_proposals)} proposals for org {org_id}",
            extra={
                'org_id': org_id,
                'budget_proposals': len(budget_proposals),
                'pause_proposals': len(pause_proposals),
                'clone_proposals': len(clone_proposals),
                'platform_insights': len(platform_insights) if platform_insights else 0
            }
        )

        return all_proposals

    async def optimize_budget_allocation(
        self,
        org_id: str,
        learning_cycle_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Algorithm 1: Budget Reallocation

        Analyzes ROAS by platform and suggests budget shifts.

        Rules:
        - If Platform A ROAS > Platform B ROAS * 1.5 → suggest +10% to A, -10% from B
        - If campaign ROAS < 1.0 for 7+ days → suggest decrease budget
        - If campaign ROAS > 4.0 → suggest increase budget +20%

        Returns:
            List of budget adjustment proposals
        """
        logger.debug(f"[ADS_OPTIMIZER] Running budget allocation optimizer for org {org_id}")

        proposals = []

        try:
            # Get platform performance comparison using database function
            platform_perf = self.supabase.rpc(
                'get_platform_performance_comparison',
                {'p_org_id': org_id, 'p_days': self.thresholds['observation_days']}
            ).execute()

            if not platform_perf.data or len(platform_perf.data) < 2:
                logger.debug(f"[ADS_OPTIMIZER] Not enough platforms to compare for org {org_id}")
                return proposals

            platforms = platform_perf.data

            # Sort by ROAS descending
            platforms_sorted = sorted(platforms, key=lambda x: float(x.get('avg_roas', 0) or 0), reverse=True)

            best_platform = platforms_sorted[0]
            worst_platform = platforms_sorted[-1]

            best_roas = float(best_platform.get('avg_roas', 0) or 0)
            worst_roas = float(worst_platform.get('avg_roas', 0) or 0)

            # Check if improvement threshold met
            if worst_roas > 0 and best_roas >= worst_roas * self.thresholds['roas_improvement_threshold']:
                # Generate platform shift proposal
                expected_gain = ((best_roas - worst_roas) / worst_roas) * 100
                confidence = min(0.95, 0.7 + (best_roas - worst_roas) / 10.0)

                proposal = {
                    'id': str(uuid.uuid4()),
                    'org_id': org_id,
                    'learning_cycle_id': learning_cycle_id,
                    'proposal_type': 'platform_shift',
                    'proposed_policy_config': None,  # Not a policy change
                    'change_type': 'minor' if confidence >= 0.8 else 'major',
                    'change_details': {
                        'from_platform': worst_platform['platform'],
                        'to_platform': best_platform['platform'],
                        'from_roas': worst_roas,
                        'to_roas': best_roas,
                        'recommendation': f"Shift budget from {worst_platform['platform']} to {best_platform['platform']}"
                    },
                    'ads_context': {
                        'platforms_compared': [p['platform'] for p in platforms],
                        'performance_delta': best_roas - worst_roas,
                        'best_platform_campaigns': best_platform.get('total_campaigns'),
                        'worst_platform_campaigns': worst_platform.get('total_campaigns')
                    },
                    'confidence_score': round(confidence, 2),
                    'expected_gain': round(expected_gain, 2),
                    'risk_level': 'low' if confidence >= 0.85 else 'medium',
                    'impact_summary': f"{best_platform['platform']} shows {expected_gain:.1f}% better ROAS",
                    'recommendation': f"Consider allocating more budget to {best_platform['platform']}",
                    'status': 'pending',
                    'auto_apply_after': (datetime.utcnow() + timedelta(hours=24)).isoformat() if confidence >= 0.8 else None
                }

                proposals.append(proposal)
                logger.info(
                    f"[ADS_OPTIMIZER] Budget shift proposal: {worst_platform['platform']} → {best_platform['platform']} "
                    f"(ROAS delta: {best_roas - worst_roas:.2f})"
                )

            # Check individual campaigns for budget adjustments
            # Get recent telemetry
            telemetry = self.supabase.table('cil_ads_telemetry')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', (datetime.utcnow() - timedelta(days=7)).isoformat())\
                .order('created_at', desc=True)\
                .execute()

            for record in telemetry.data or []:
                roas = float(record.get('roas', 0) or 0)
                spend = float(record.get('spend', 0) or 0)

                # Skip if not enough spend
                if spend < self.thresholds['min_spend']:
                    continue

                # High performer - suggest increase
                if roas >= self.thresholds['excellent_roas']:
                    proposal = {
                        'id': str(uuid.uuid4()),
                        'org_id': org_id,
                        'learning_cycle_id': learning_cycle_id,
                        'proposal_type': 'increase_budget',
                        'proposed_policy_config': None,
                        'change_type': 'minor',
                        'change_details': {
                            'campaign_id': record['campaign_id'],
                            'campaign_name': record['campaign_name'],
                            'current_roas': roas,
                            'budget_increase_pct': 20
                        },
                        'ads_context': {
                            'platform': record['platform'],
                            'current_metrics': {
                                'roas': roas,
                                'ctr': record.get('ctr'),
                                'conversions': record.get('conversions')
                            }
                        },
                        'confidence_score': 0.85,
                        'expected_gain': 20.0,
                        'risk_level': 'low',
                        'impact_summary': f"High-performing campaign (ROAS: {roas:.2f}x) can scale",
                        'recommendation': f"Increase budget by 20% for {record['campaign_name']}",
                        'status': 'pending',
                        'auto_apply_after': (datetime.utcnow() + timedelta(hours=24)).isoformat()
                    }
                    proposals.append(proposal)

                # Low performer - suggest decrease
                elif roas < self.thresholds['min_roas']:
                    proposal = {
                        'id': str(uuid.uuid4()),
                        'org_id': org_id,
                        'learning_cycle_id': learning_cycle_id,
                        'proposal_type': 'decrease_budget',
                        'proposed_policy_config': None,
                        'change_type': 'minor',
                        'change_details': {
                            'campaign_id': record['campaign_id'],
                            'campaign_name': record['campaign_name'],
                            'current_roas': roas,
                            'budget_decrease_pct': 30
                        },
                        'ads_context': {
                            'platform': record['platform'],
                            'current_metrics': {
                                'roas': roas,
                                'spend': spend
                            }
                        },
                        'confidence_score': 0.80,
                        'expected_gain': 0.0,  # Preventing loss, not gaining
                        'risk_level': 'low',
                        'impact_summary': f"Underperforming campaign (ROAS: {roas:.2f}x < 1.0)",
                        'recommendation': f"Decrease budget by 30% for {record['campaign_name']} or pause",
                        'status': 'pending',
                        'auto_apply_after': (datetime.utcnow() + timedelta(hours=48)).isoformat()
                    }
                    proposals.append(proposal)

        except Exception as e:
            logger.error(f"[ADS_OPTIMIZER] Error in budget allocation optimizer: {e}", exc_info=True)

        return proposals

    async def detect_underperforming_campaigns(
        self,
        org_id: str,
        learning_cycle_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Algorithm 2: Low-Performer Detection

        Flags campaigns with poor metrics for pausing.

        Criteria:
        - CTR < 0.5% for 14+ days
        - ROAS < 1.0 for 7+ days
        - Quality Score < 3 (Google Ads)

        Returns:
            List of pause campaign proposals
        """
        logger.debug(f"[ADS_OPTIMIZER] Running underperformer detector for org {org_id}")

        proposals = []

        try:
            # Use database function to identify underperformers
            underperformers = self.supabase.rpc(
                'identify_underperforming_campaigns',
                {
                    'p_org_id': org_id,
                    'p_min_ctr': self.thresholds['min_ctr'],
                    'p_min_roas': self.thresholds['min_roas'],
                    'p_min_quality_score': self.thresholds['min_quality_score']
                }
            ).execute()

            for campaign in underperformers.data or []:
                ctr = float(campaign.get('ctr', 0) or 0)
                roas = float(campaign.get('roas', 0) or 0)
                quality_score = float(campaign.get('quality_score', 0) or 0) if campaign.get('quality_score') else None

                # Determine severity
                issues = []
                if ctr < self.thresholds['min_ctr']:
                    issues.append(f"Low CTR: {ctr:.2f}%")
                if roas < self.thresholds['min_roas']:
                    issues.append(f"Negative ROAS: {roas:.2f}x")
                if quality_score and quality_score < self.thresholds['min_quality_score']:
                    issues.append(f"Low Quality Score: {quality_score:.1f}/10")

                confidence = 0.85 if len(issues) >= 2 else 0.75

                proposal = {
                    'id': str(uuid.uuid4()),
                    'org_id': org_id,
                    'learning_cycle_id': learning_cycle_id,
                    'proposal_type': 'pause_campaign',
                    'proposed_policy_config': None,
                    'change_type': 'minor' if confidence >= 0.8 else 'major',
                    'change_details': {
                        'campaign_id': campaign['campaign_id'],
                        'campaign_name': campaign['campaign_name'],
                        'platform': campaign['platform'],
                        'issues': issues,
                        'days_underperforming': campaign.get('days_underperforming', 0)
                    },
                    'ads_context': {
                        'current_metrics': {
                            'ctr': ctr,
                            'roas': roas,
                            'quality_score': quality_score,
                            'spend': campaign.get('spend')
                        }
                    },
                    'confidence_score': confidence,
                    'expected_gain': 0.0,  # Preventing loss
                    'risk_level': 'low',
                    'impact_summary': f"Campaign failing on {len(issues)} metrics: {', '.join(issues)}",
                    'recommendation': f"Pause {campaign['campaign_name']} to stop losses",
                    'status': 'pending',
                    'auto_apply_after': (datetime.utcnow() + timedelta(hours=24)).isoformat() if confidence >= 0.8 else None
                }

                proposals.append(proposal)

            logger.info(f"[ADS_OPTIMIZER] Found {len(proposals)} underperforming campaigns")

        except Exception as e:
            logger.error(f"[ADS_OPTIMIZER] Error detecting underperformers: {e}", exc_info=True)

        return proposals

    async def identify_top_performers(
        self,
        org_id: str,
        learning_cycle_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Algorithm 3: High-Performer Cloner

        Identifies campaigns worth replicating.

        Criteria:
        - ROAS > 4.0
        - CTR > 2.5%
        - Conversions > 50 in last 30 days

        Returns:
            List of clone campaign proposals
        """
        logger.debug(f"[ADS_OPTIMIZER] Running top performer identifier for org {org_id}")

        proposals = []

        try:
            # Use database function to find top performers
            top_performers = self.supabase.rpc(
                'identify_top_performing_campaigns',
                {
                    'p_org_id': org_id,
                    'p_min_roas': self.thresholds['excellent_roas'],
                    'p_min_ctr': self.thresholds['excellent_ctr'],
                    'p_min_conversions': self.thresholds['min_conversions']
                }
            ).execute()

            for campaign in top_performers.data or []:
                roas = float(campaign.get('roas', 0) or 0)
                ctr = float(campaign.get('ctr', 0) or 0)
                conversions = int(campaign.get('conversions', 0) or 0)
                performance_score = float(campaign.get('performance_score', 0) or 0)

                # Higher confidence for better performers
                confidence = min(0.90, 0.65 + performance_score * 0.25)

                proposal = {
                    'id': str(uuid.uuid4()),
                    'org_id': org_id,
                    'learning_cycle_id': learning_cycle_id,
                    'proposal_type': 'clone_campaign',
                    'proposed_policy_config': None,
                    'change_type': 'major',  # Cloning requires approval
                    'change_details': {
                        'source_campaign_id': campaign['campaign_id'],
                        'source_campaign_name': campaign['campaign_name'],
                        'platform': campaign['platform'],
                        'clone_reason': 'high_performance'
                    },
                    'ads_context': {
                        'source_metrics': {
                            'roas': roas,
                            'ctr': ctr,
                            'conversions': conversions,
                            'performance_score': performance_score
                        }
                    },
                    'confidence_score': round(confidence, 2),
                    'expected_gain': 50.0,  # Estimate: clone could add 50% more results
                    'risk_level': 'medium',
                    'impact_summary': f"Top performer (ROAS: {roas:.2f}x, CTR: {ctr:.2f}%, Score: {performance_score:.2f})",
                    'recommendation': f"Clone {campaign['campaign_name']} to scale success",
                    'status': 'pending',
                    'auto_apply_after': None  # Requires manual approval
                }

                proposals.append(proposal)

            logger.info(f"[ADS_OPTIMIZER] Found {len(proposals)} campaigns worth cloning")

        except Exception as e:
            logger.error(f"[ADS_OPTIMIZER] Error identifying top performers: {e}", exc_info=True)

        return proposals

    async def learn_platform_preferences(
        self,
        org_id: str,
        learning_cycle_id: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Algorithm 4: Platform Preference Learner

        Discovers which platform works best for different campaign objectives.

        Learns patterns like:
        - "Product launches → Meta Reels (3.2% CTR avg)"
        - "B2B leads → Google Search (4.5 ROAS avg)"

        Stores insights in ai_meta_knowledge table.

        Returns:
            List of meta-knowledge records created
        """
        logger.debug(f"[ADS_OPTIMIZER] Learning platform preferences for org {org_id}")

        insights = []

        try:
            # Get telemetry grouped by platform
            telemetry = self.supabase.table('cil_ads_telemetry')\
                .select('platform, campaign_name, roas, ctr, conversions, performance_score')\
                .eq('org_id', org_id)\
                .gte('created_at', (datetime.utcnow() - timedelta(days=30)).isoformat())\
                .execute()

            if not telemetry.data or len(telemetry.data) < 5:
                logger.debug(f"[ADS_OPTIMIZER] Not enough data for platform learning")
                return None

            # Group by platform and calculate averages
            platform_stats = {}
            for record in telemetry.data:
                platform = record['platform']
                if platform not in platform_stats:
                    platform_stats[platform] = {
                        'roas_sum': 0,
                        'ctr_sum': 0,
                        'conversions_sum': 0,
                        'performance_sum': 0,
                        'count': 0
                    }

                stats = platform_stats[platform]
                stats['roas_sum'] += float(record.get('roas', 0) or 0)
                stats['ctr_sum'] += float(record.get('ctr', 0) or 0)
                stats['conversions_sum'] += int(record.get('conversions', 0) or 0)
                stats['performance_sum'] += float(record.get('performance_score', 0) or 0)
                stats['count'] += 1

            # Calculate averages and create meta-knowledge
            for platform, stats in platform_stats.items():
                count = stats['count']
                avg_roas = stats['roas_sum'] / count
                avg_ctr = stats['ctr_sum'] / count
                avg_conversions = stats['conversions_sum'] / count
                avg_performance = stats['performance_sum'] / count

                # Generate rule text
                if avg_performance >= 0.7:
                    strength = "excellent"
                elif avg_performance >= 0.5:
                    strength = "good"
                else:
                    strength = "moderate"

                rule_text = (
                    f"{platform} shows {strength} performance: "
                    f"avg ROAS {avg_roas:.2f}x, avg CTR {avg_ctr:.2f}%, "
                    f"based on {count} campaigns"
                )

                # Create meta-knowledge record
                meta_knowledge = {
                    'org_id': org_id,
                    'rule_text': rule_text,
                    'category': 'pattern',
                    'confidence': min(0.95, 0.6 + avg_performance * 0.35),
                    'application_count': 0,
                    'success_rate': avg_performance,
                    'metadata': {
                        'platform': platform,
                        'avg_roas': avg_roas,
                        'avg_ctr': avg_ctr,
                        'avg_conversions': avg_conversions,
                        'campaigns_analyzed': count,
                        'learning_cycle_id': learning_cycle_id
                    }
                }

                # Check if similar rule already exists
                existing = self.supabase.table('ai_meta_knowledge')\
                    .select('id')\
                    .eq('org_id', org_id)\
                    .eq('category', 'pattern')\
                    .ilike('rule_text', f'%{platform}%')\
                    .execute()

                if not existing.data:
                    self.supabase.table('ai_meta_knowledge').insert(meta_knowledge).execute()
                    insights.append(meta_knowledge)
                    logger.info(f"[ADS_OPTIMIZER] Created platform preference rule: {rule_text}")

            return insights

        except Exception as e:
            logger.error(f"[ADS_OPTIMIZER] Error learning platform preferences: {e}", exc_info=True)
            return None


# Singleton instance
_cil_ads_optimizer: Optional[CILAdsOptimizer] = None


def get_cil_ads_optimizer() -> CILAdsOptimizer:
    """Get singleton CIL ads optimizer instance"""
    global _cil_ads_optimizer
    if _cil_ads_optimizer is None:
        _cil_ads_optimizer = CILAdsOptimizer()
    return _cil_ads_optimizer
