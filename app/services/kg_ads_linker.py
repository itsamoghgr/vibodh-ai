"""
Knowledge Graph Ads Linker Service - Phase 6.5
Links ad campaigns to knowledge graph entities and relationships

Responsibilities:
- Link campaigns to performance metrics
- Link CIL optimizations to campaigns
- Create relationships between campaigns and organizational entities
- Discover campaign-related insights
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.core.logging import logger
from app.db import get_supabase_admin_client


class KGAdsLinker:
    """
    Service for linking ads campaigns to knowledge graph.
    """

    def __init__(self):
        self.supabase = get_supabase_admin_client()

    async def link_campaign_performance(
        self,
        campaign_id: str,
        org_id: str
    ) -> Dict[str, Any]:
        """
        Create KG entities and relationships for campaign performance.

        Creates:
        - Performance metric entities (ROAS, CTR, Conversions)
        - achieved_performance relationships

        Args:
            campaign_id: Campaign ID
            org_id: Organization ID

        Returns:
            Linking result with entity/edge counts
        """
        try:
            # Get campaign entity
            campaign_link = self.supabase.table('ads_kg_links')\
                .select('entity_id')\
                .eq('campaign_id', campaign_id)\
                .eq('link_type', 'campaign_entity')\
                .single()\
                .execute()

            if not campaign_link.data:
                logger.warning(f"Campaign {campaign_id} not linked to KG")
                return {'success': False, 'message': 'Campaign not linked to KG'}

            campaign_entity_id = campaign_link.data['entity_id']

            # Get latest metrics
            metrics = self.supabase.table('ad_metrics')\
                .select('*')\
                .eq('campaign_id', campaign_id)\
                .order('metric_date', desc=True)\
                .limit(1)\
                .execute()

            if not metrics.data:
                return {'success': False, 'message': 'No metrics found'}

            latest_metrics = metrics.data[0]

            entities_created = 0
            edges_created = 0

            # Create performance metric entities
            performance_metrics = [
                {
                    'name': f"ROAS: {latest_metrics.get('roas', 0):.2f}x",
                    'value': latest_metrics.get('roas', 0),
                    'metric_type': 'roas'
                },
                {
                    'name': f"CTR: {latest_metrics.get('ctr', 0):.2f}%",
                    'value': latest_metrics.get('ctr', 0),
                    'metric_type': 'ctr'
                },
                {
                    'name': f"Conversions: {latest_metrics.get('conversions', 0)}",
                    'value': latest_metrics.get('conversions', 0),
                    'metric_type': 'conversions'
                }
            ]

            for metric in performance_metrics:
                # Create metric entity (using 'topic' type as closest match)
                entity_result = self.supabase.table('ai_kg_entities').insert({
                    'org_id': org_id,
                    'name': metric['name'],
                    'entity_type': 'topic',  # Using topic for metrics
                    'confidence': 0.95,
                    'metadata': {
                        'metric_type': metric['metric_type'],
                        'value': metric['value'],
                        'date': latest_metrics.get('metric_date'),
                        'source': 'ads_performance',
                        'campaign_id': campaign_id
                    }
                }).execute()

                if entity_result.data:
                    metric_entity_id = entity_result.data[0]['id']
                    entities_created += 1

                    # Create relationship: Campaign → achieved_performance → Metric
                    edge_result = self.supabase.table('ai_kg_edges').insert({
                        'org_id': org_id,
                        'from_entity_id': campaign_entity_id,
                        'to_entity_id': metric_entity_id,
                        'relation_type': 'achieved_performance',
                        'confidence': 0.95,
                        'metadata': {
                            'metric_date': latest_metrics.get('metric_date'),
                            'source': 'ads_linker'
                        }
                    }).execute()

                    if edge_result.data:
                        edges_created += 1

                    # Create link record
                    self.supabase.table('ads_kg_links').insert({
                        'org_id': org_id,
                        'campaign_id': campaign_id,
                        'entity_id': metric_entity_id,
                        'link_type': 'performance_metric',
                        'auto_linked': True
                    }).execute()

            logger.info(
                f"Linked performance for campaign {campaign_id}: "
                f"{entities_created} entities, {edges_created} edges"
            )

            return {
                'success': True,
                'entities_created': entities_created,
                'edges_created': edges_created
            }

        except Exception as e:
            logger.error(f"Error linking campaign performance: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def link_optimization_action(
        self,
        optimization_id: str,
        org_id: str
    ) -> Dict[str, Any]:
        """
        Create KG entities and relationships for CIL optimization actions.

        Creates:
        - Optimization action entity
        - optimized_by relationship (Campaign → Optimization)
        - proposed_by relationship (Optimization → CIL)

        Args:
            optimization_id: Optimization history record ID
            org_id: Organization ID

        Returns:
            Linking result
        """
        try:
            # Get optimization details
            optimization = self.supabase.table('cil_ads_optimization_history')\
                .select('*')\
                .eq('id', optimization_id)\
                .single()\
                .execute()

            if not optimization.data:
                return {'success': False, 'message': 'Optimization not found'}

            opt_data = optimization.data
            campaign_id = opt_data.get('campaign_id')

            if not campaign_id:
                return {'success': False, 'message': 'No campaign linked to optimization'}

            # Get campaign entity
            campaign_link = self.supabase.table('ads_kg_links')\
                .select('entity_id')\
                .eq('campaign_id', campaign_id)\
                .eq('link_type', 'campaign_entity')\
                .single()\
                .execute()

            if not campaign_link.data:
                logger.warning(f"Campaign {campaign_id} not linked to KG")
                return {'success': False, 'message': 'Campaign not linked to KG'}

            campaign_entity_id = campaign_link.data['entity_id']

            # Create optimization action entity
            action_name = f"{opt_data['action_taken']} - {opt_data['proposal_type']}"

            entity_result = self.supabase.table('ai_kg_entities').insert({
                'org_id': org_id,
                'name': action_name,
                'entity_type': 'optimization_action',
                'confidence': opt_data.get('confidence_score', 0.8),
                'metadata': {
                    'optimization_id': optimization_id,
                    'proposal_id': opt_data.get('proposal_id'),
                    'action_taken': opt_data['action_taken'],
                    'proposal_type': opt_data['proposal_type'],
                    'expected_gain': opt_data.get('expected_gain'),
                    'actual_gain': opt_data.get('actual_gain'),
                    'success': opt_data.get('success'),
                    'applied_at': opt_data.get('applied_at'),
                    'source': 'cil_optimization'
                }
            }).execute()

            if not entity_result.data:
                return {'success': False, 'message': 'Failed to create optimization entity'}

            optimization_entity_id = entity_result.data[0]['id']

            # Create relationship: Campaign → optimized_by → Optimization
            self.supabase.table('ai_kg_edges').insert({
                'org_id': org_id,
                'from_entity_id': campaign_entity_id,
                'to_entity_id': optimization_entity_id,
                'relation_type': 'optimized_by',
                'confidence': opt_data.get('confidence_score', 0.8),
                'metadata': {
                    'applied_at': opt_data.get('applied_at'),
                    'success': opt_data.get('success')
                }
            }).execute()

            # Create link record
            self.supabase.table('ads_kg_links').insert({
                'org_id': org_id,
                'campaign_id': campaign_id,
                'entity_id': optimization_entity_id,
                'link_type': 'optimization_action',
                'auto_linked': True,
                'confidence': opt_data.get('confidence_score', 0.8)
            }).execute()

            # Find CIL agent entity and create proposed_by relationship
            cil_entity = self.supabase.table('ai_kg_entities')\
                .select('id')\
                .eq('org_id', org_id)\
                .eq('entity_type', 'tool')\
                .ilike('name', '%CIL%')\
                .limit(1)\
                .execute()

            if cil_entity.data:
                cil_entity_id = cil_entity.data[0]['id']

                self.supabase.table('ai_kg_edges').insert({
                    'org_id': org_id,
                    'from_entity_id': optimization_entity_id,
                    'to_entity_id': cil_entity_id,
                    'relation_type': 'proposed_by',
                    'confidence': 0.9,
                    'metadata': {
                        'proposal_id': opt_data.get('proposal_id')
                    }
                }).execute()

            logger.info(f"Linked optimization action {optimization_id} to campaign {campaign_id}")

            return {
                'success': True,
                'entity_id': optimization_entity_id,
                'campaign_entity_id': campaign_entity_id
            }

        except Exception as e:
            logger.error(f"Error linking optimization action: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def link_campaigns_batch(
        self,
        org_id: str,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Batch link recent campaigns to KG.

        Args:
            org_id: Organization ID
            days_back: Days of campaigns to link

        Returns:
            Batch linking results
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            # Get recent campaigns that need linking
            campaigns = self.supabase.table('ad_campaigns')\
                .select('id')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff_date.isoformat())\
                .execute()

            campaign_ids = [c['id'] for c in (campaigns.data or [])]

            total_entities = 0
            total_edges = 0
            campaigns_processed = 0

            for campaign_id in campaign_ids:
                # Link performance
                result = await self.link_campaign_performance(
                    campaign_id=campaign_id,
                    org_id=org_id
                )

                if result.get('success'):
                    total_entities += result.get('entities_created', 0)
                    total_edges += result.get('edges_created', 0)
                    campaigns_processed += 1

                # Small delay to avoid overwhelming DB
                await asyncio.sleep(0.1)

            logger.info(
                f"Batch linked {campaigns_processed} campaigns: "
                f"{total_entities} entities, {total_edges} edges"
            )

            return {
                'success': True,
                'campaigns_processed': campaigns_processed,
                'entities_created': total_entities,
                'edges_created': total_edges
            }

        except Exception as e:
            logger.error(f"Error in batch linking: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def get_campaign_graph(
        self,
        campaign_id: str
    ) -> Dict[str, Any]:
        """
        Get full knowledge graph for a campaign.

        Returns:
        - Campaign entity
        - Related entities (performance, optimizations, platform)
        - All relationships

        Args:
            campaign_id: Campaign ID

        Returns:
            Graph data (nodes and edges)
        """
        try:
            # Get campaign's KG entities
            links = self.supabase.table('ads_kg_links')\
                .select('entity_id, link_type, confidence')\
                .eq('campaign_id', campaign_id)\
                .execute()

            entity_ids = [link['entity_id'] for link in (links.data or [])]

            if not entity_ids:
                return {
                    'success': False,
                    'message': 'Campaign not linked to KG'
                }

            # Get entities
            entities = self.supabase.table('ai_kg_entities')\
                .select('*')\
                .in_('id', entity_ids)\
                .execute()

            # Get edges connecting these entities
            edges = self.supabase.table('ai_kg_edges')\
                .select('*')\
                .or_(
                    f"from_entity_id.in.({','.join(entity_ids)}),"
                    f"to_entity_id.in.({','.join(entity_ids)})"
                )\
                .execute()

            return {
                'success': True,
                'nodes': entities.data or [],
                'edges': edges.data or [],
                'node_count': len(entities.data or []),
                'edge_count': len(edges.data or [])
            }

        except Exception as e:
            logger.error(f"Error getting campaign graph: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def find_campaign_insights(
        self,
        org_id: str,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Discover insights from campaign KG patterns.

        Analyzes:
        - Successful optimization patterns
        - Platform performance patterns
        - Campaign relationships

        Args:
            org_id: Organization ID
            min_confidence: Minimum confidence threshold

        Returns:
            List of discovered insights
        """
        try:
            insights = []

            # Pattern 1: Successful optimization patterns
            successful_optimizations = self.supabase.table('cil_ads_optimization_history')\
                .select('proposal_type, action_taken, actual_gain')\
                .eq('org_id', org_id)\
                .eq('success', True)\
                .gte('actual_gain', 5.0)\
                .execute()

            if successful_optimizations.data and len(successful_optimizations.data) >= 3:
                # Group by action type
                action_counts = {}
                for opt in successful_optimizations.data:
                    action = opt['action_taken']
                    action_counts[action] = action_counts.get(action, 0) + 1

                # Find most common successful actions
                for action, count in action_counts.items():
                    if count >= 2:
                        insights.append({
                            'type': 'optimization_pattern',
                            'insight': f"'{action}' optimization has been successful {count} times",
                            'confidence': min(0.95, 0.7 + (count * 0.05)),
                            'metadata': {
                                'action': action,
                                'success_count': count
                            }
                        })

            # Pattern 2: Platform preferences
            platform_performance = self.supabase.rpc(
                'get_platform_performance_comparison',
                {'p_org_id': org_id, 'p_days': 30}
            ).execute()

            if platform_performance.data and len(platform_performance.data) >= 2:
                platforms = platform_performance.data
                best = max(platforms, key=lambda x: x.get('avg_roas', 0))
                worst = min(platforms, key=lambda x: x.get('avg_roas', 0))

                roas_diff = best['avg_roas'] - worst['avg_roas']

                if roas_diff > 1.0:  # Significant difference
                    insights.append({
                        'type': 'platform_preference',
                        'insight': f"{best['platform']} consistently outperforms {worst['platform']} "
                                   f"by {roas_diff:.1f}x ROAS",
                        'confidence': 0.85,
                        'metadata': {
                            'best_platform': best['platform'],
                            'worst_platform': worst['platform'],
                            'roas_difference': roas_diff
                        }
                    })

            return insights

        except Exception as e:
            logger.error(f"Error finding campaign insights: {e}", exc_info=True)
            return []


# Singleton instance
_kg_ads_linker: Optional[KGAdsLinker] = None


def get_kg_ads_linker() -> KGAdsLinker:
    """Get singleton KG ads linker instance"""
    global _kg_ads_linker
    if _kg_ads_linker is None:
        _kg_ads_linker = KGAdsLinker()
    return _kg_ads_linker


# Add missing import
import asyncio
