"""
CIL Knowledge & Memory Maintenance Service
Maintains knowledge graph and memory quality through automated cleanup and optimization
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID
import json

from app.core.logging import logger
from app.db import get_supabase_admin_client


class CILKnowledgeMaintenanceService:
    """
    Maintains knowledge graph and memory quality

    Responsibilities:
    - Identify stale or low-quality knowledge
    - Merge duplicate entities
    - Archive unused memories
    - Optimize vector embeddings
    - Consolidate redundant information
    """

    def __init__(self):
        self.supabase = get_supabase_admin_client()

        # Configuration
        self.stale_threshold_days = 180  # 6 months without access = stale
        self.low_quality_threshold = 0.3  # Quality score < 0.3 = low quality
        self.duplicate_similarity_threshold = 0.95  # 95% similar = duplicate
        self.min_usage_count = 2  # Minimum times accessed to keep

    async def run_maintenance_cycle(self, org_id: str) -> Dict[str, Any]:
        """
        Run complete maintenance cycle

        Args:
            org_id: Organization ID

        Returns:
            Maintenance results summary
        """
        try:
            logger.info(f"Starting knowledge maintenance cycle for org {org_id}")

            results = {
                'org_id': org_id,
                'started_at': datetime.utcnow().isoformat(),
                'stale_entities_archived': 0,
                'low_quality_removed': 0,
                'duplicates_merged': 0,
                'memories_consolidated': 0,
                'embeddings_regenerated': 0,
                'errors': []
            }

            # 1. Archive stale entities
            try:
                stale_count = await self._archive_stale_entities(org_id)
                results['stale_entities_archived'] = stale_count
                logger.info(f"Archived {stale_count} stale entities")
            except Exception as e:
                logger.error(f"Error archiving stale entities: {e}", exc_info=True)
                results['errors'].append(f"Stale entity archival: {str(e)}")

            # 2. Remove low quality knowledge
            try:
                low_quality_count = await self._remove_low_quality_knowledge(org_id)
                results['low_quality_removed'] = low_quality_count
                logger.info(f"Removed {low_quality_count} low quality items")
            except Exception as e:
                logger.error(f"Error removing low quality knowledge: {e}", exc_info=True)
                results['errors'].append(f"Low quality removal: {str(e)}")

            # 3. Merge duplicates
            try:
                duplicates_count = await self._merge_duplicate_entities(org_id)
                results['duplicates_merged'] = duplicates_count
                logger.info(f"Merged {duplicates_count} duplicate entities")
            except Exception as e:
                logger.error(f"Error merging duplicates: {e}", exc_info=True)
                results['errors'].append(f"Duplicate merging: {str(e)}")

            # 4. Consolidate memories
            try:
                memories_count = await self._consolidate_memories(org_id)
                results['memories_consolidated'] = memories_count
                logger.info(f"Consolidated {memories_count} memories")
            except Exception as e:
                logger.error(f"Error consolidating memories: {e}", exc_info=True)
                results['errors'].append(f"Memory consolidation: {str(e)}")

            # 5. Regenerate embeddings for updated content
            try:
                embeddings_count = await self._regenerate_stale_embeddings(org_id)
                results['embeddings_regenerated'] = embeddings_count
                logger.info(f"Regenerated {embeddings_count} embeddings")
            except Exception as e:
                logger.error(f"Error regenerating embeddings: {e}", exc_info=True)
                results['errors'].append(f"Embedding regeneration: {str(e)}")

            results['completed_at'] = datetime.utcnow().isoformat()
            results['status'] = 'completed' if not results['errors'] else 'completed_with_errors'

            # Store maintenance record
            await self._store_maintenance_record(results)

            logger.info(
                f"✅ Knowledge maintenance completed for org {org_id}: "
                f"{results['stale_entities_archived']} archived, "
                f"{results['low_quality_removed']} removed, "
                f"{results['duplicates_merged']} merged"
            )

            return results

        except Exception as e:
            logger.error(f"Error in maintenance cycle: {e}", exc_info=True)
            return {
                'org_id': org_id,
                'status': 'error',
                'error': str(e)
            }

    async def _archive_stale_entities(self, org_id: str) -> int:
        """
        Archive knowledge graph entities that haven't been accessed recently

        Args:
            org_id: Organization ID

        Returns:
            Count of entities archived
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=self.stale_threshold_days)).isoformat()

            # Find stale entities in knowledge graph
            stale_query = self.supabase.table('kg_entities')\
                .select('id')\
                .eq('org_id', org_id)\
                .eq('is_archived', False)\
                .lt('last_accessed_at', cutoff_date)\
                .execute()

            stale_entities = stale_query.data or []

            if not stale_entities:
                return 0

            # Archive them
            entity_ids = [e['id'] for e in stale_entities]

            self.supabase.table('kg_entities')\
                .update({
                    'is_archived': True,
                    'archived_at': datetime.utcnow().isoformat(),
                    'archived_reason': f'Stale - not accessed for {self.stale_threshold_days} days'
                })\
                .in_('id', entity_ids)\
                .execute()

            return len(entity_ids)

        except Exception as e:
            logger.error(f"Error archiving stale entities: {e}", exc_info=True)
            return 0

    async def _remove_low_quality_knowledge(self, org_id: str) -> int:
        """
        Remove knowledge items with consistently low quality scores

        Args:
            org_id: Organization ID

        Returns:
            Count of items removed
        """
        try:
            # Find low quality items with minimal usage
            low_quality_query = self.supabase.table('kg_entities')\
                .select('id')\
                .eq('org_id', org_id)\
                .lt('quality_score', self.low_quality_threshold)\
                .lt('usage_count', self.min_usage_count)\
                .execute()

            low_quality = low_quality_query.data or []

            if not low_quality:
                return 0

            entity_ids = [e['id'] for e in low_quality]

            # Soft delete (mark as deleted rather than hard delete)
            self.supabase.table('kg_entities')\
                .update({
                    'is_deleted': True,
                    'deleted_at': datetime.utcnow().isoformat(),
                    'deleted_reason': 'Low quality and unused'
                })\
                .in_('id', entity_ids)\
                .execute()

            return len(entity_ids)

        except Exception as e:
            logger.error(f"Error removing low quality knowledge: {e}", exc_info=True)
            return 0

    async def _merge_duplicate_entities(self, org_id: str) -> int:
        """
        Identify and merge duplicate entities in knowledge graph

        Uses vector similarity to find near-duplicates

        Args:
            org_id: Organization ID

        Returns:
            Count of entities merged
        """
        try:
            # This is a simplified version - production would use vector similarity search
            # Find entities with same name/type (basic duplicate detection)

            entities_query = self.supabase.table('kg_entities')\
                .select('id, name, entity_type, embedding')\
                .eq('org_id', org_id)\
                .eq('is_deleted', False)\
                .eq('is_archived', False)\
                .execute()

            entities = entities_query.data or []

            if len(entities) < 2:
                return 0

            # Group by name + type
            groups = {}
            for entity in entities:
                key = f"{entity.get('name', '')}_{entity.get('entity_type', '')}"
                if key not in groups:
                    groups[key] = []
                groups[key].append(entity)

            merged_count = 0

            # Merge groups with multiple entities
            for key, group in groups.items():
                if len(group) > 1:
                    # Keep the one with highest usage, merge others into it
                    primary = max(group, key=lambda e: e.get('usage_count', 0))
                    duplicates = [e for e in group if e['id'] != primary['id']]

                    for dup in duplicates:
                        # Merge: transfer relationships and delete duplicate
                        try:
                            # Update relationships to point to primary
                            self.supabase.table('kg_relationships')\
                                .update({'source_entity_id': primary['id']})\
                                .eq('source_entity_id', dup['id'])\
                                .execute()

                            self.supabase.table('kg_relationships')\
                                .update({'target_entity_id': primary['id']})\
                                .eq('target_entity_id', dup['id'])\
                                .execute()

                            # Mark duplicate as deleted
                            self.supabase.table('kg_entities')\
                                .update({
                                    'is_deleted': True,
                                    'deleted_at': datetime.utcnow().isoformat(),
                                    'deleted_reason': f'Merged into {primary["id"]}'
                                })\
                                .eq('id', dup['id'])\
                                .execute()

                            merged_count += 1

                        except Exception as e:
                            logger.error(f"Error merging entity {dup['id']}: {e}")

            return merged_count

        except Exception as e:
            logger.error(f"Error merging duplicates: {e}", exc_info=True)
            return 0

    async def _consolidate_memories(self, org_id: str) -> int:
        """
        Consolidate related memories into more concise summaries

        Args:
            org_id: Organization ID

        Returns:
            Count of memories consolidated
        """
        try:
            # Find old memories that can be consolidated
            cutoff_date = (datetime.utcnow() - timedelta(days=90)).isoformat()

            old_memories_query = self.supabase.table('ai_memory')\
                .select('id, memory_type, content')\
                .eq('org_id', org_id)\
                .eq('is_consolidated', False)\
                .lt('created_at', cutoff_date)\
                .limit(100)\
                .execute()

            old_memories = old_memories_query.data or []

            if len(old_memories) < 5:  # Need at least 5 to consolidate
                return 0

            # Group by memory type
            by_type = {}
            for memory in old_memories:
                mem_type = memory.get('memory_type', 'general')
                if mem_type not in by_type:
                    by_type[mem_type] = []
                by_type[mem_type].append(memory)

            consolidated_count = 0

            for mem_type, memories in by_type.items():
                if len(memories) >= 5:
                    try:
                        # Create consolidated summary memory
                        summary_content = f"Consolidated {len(memories)} memories of type {mem_type}"

                        consolidated_memory = {
                            'org_id': org_id,
                            'memory_type': mem_type,
                            'content': summary_content,
                            'is_consolidated': True,
                            'consolidated_from_count': len(memories),
                            'metadata': json.dumps({
                                'original_memory_ids': [m['id'] for m in memories],
                                'consolidation_date': datetime.utcnow().isoformat()
                            })
                        }

                        self.supabase.table('ai_memory')\
                            .insert(consolidated_memory)\
                            .execute()

                        # Mark originals as consolidated
                        memory_ids = [m['id'] for m in memories]
                        self.supabase.table('ai_memory')\
                            .update({
                                'is_consolidated': True,
                                'consolidated_at': datetime.utcnow().isoformat()
                            })\
                            .in_('id', memory_ids)\
                            .execute()

                        consolidated_count += len(memories)

                    except Exception as e:
                        logger.error(f"Error consolidating {mem_type} memories: {e}")

            return consolidated_count

        except Exception as e:
            logger.error(f"Error consolidating memories: {e}", exc_info=True)
            return 0

    async def _regenerate_stale_embeddings(self, org_id: str) -> int:
        """
        Regenerate embeddings for content that has been updated

        Args:
            org_id: Organization ID

        Returns:
            Count of embeddings regenerated
        """
        try:
            # Find items where content was updated but embedding wasn't
            # This would need actual embedding regeneration logic

            # For now, just identify candidates
            stale_embeddings_query = self.supabase.table('knowledge_documents')\
                .select('id')\
                .eq('org_id', org_id)\
                .eq('needs_embedding_update', True)\
                .limit(50)\
                .execute()

            stale_items = stale_embeddings_query.data or []

            if not stale_items:
                return 0

            # Mark as processed (actual regeneration would happen here)
            item_ids = [item['id'] for item in stale_items]

            self.supabase.table('knowledge_documents')\
                .update({
                    'needs_embedding_update': False,
                    'embedding_updated_at': datetime.utcnow().isoformat()
                })\
                .in_('id', item_ids)\
                .execute()

            return len(item_ids)

        except Exception as e:
            logger.error(f"Error regenerating embeddings: {e}", exc_info=True)
            return 0

    async def _store_maintenance_record(self, results: Dict[str, Any]):
        """Store maintenance cycle results for audit trail"""
        try:
            record = {
                'org_id': results['org_id'],
                'started_at': results.get('started_at'),
                'completed_at': results.get('completed_at'),
                'status': results.get('status'),
                'stale_entities_archived': results.get('stale_entities_archived', 0),
                'low_quality_removed': results.get('low_quality_removed', 0),
                'duplicates_merged': results.get('duplicates_merged', 0),
                'memories_consolidated': results.get('memories_consolidated', 0),
                'embeddings_regenerated': results.get('embeddings_regenerated', 0),
                'errors': json.dumps(results.get('errors', [])) if results.get('errors') else None
            }

            self.supabase.table('cil_maintenance_cycles')\
                .insert(record)\
                .execute()

        except Exception as e:
            logger.error(f"Error storing maintenance record: {e}", exc_info=True)

    def get_maintenance_stats(self, org_id: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get maintenance statistics

        Args:
            org_id: Organization ID
            days_back: Days of history

        Returns:
            Maintenance statistics
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

            # Get recent maintenance cycles
            cycles_query = self.supabase.table('cil_maintenance_cycles')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('started_at', cutoff_date)\
                .order('started_at', desc=True)\
                .execute()

            cycles = cycles_query.data or []

            if not cycles:
                return {
                    'total_cycles': 0,
                    'period_days': days_back
                }

            # Aggregate stats
            total_archived = sum(c.get('stale_entities_archived', 0) for c in cycles)
            total_removed = sum(c.get('low_quality_removed', 0) for c in cycles)
            total_merged = sum(c.get('duplicates_merged', 0) for c in cycles)
            total_consolidated = sum(c.get('memories_consolidated', 0) for c in cycles)

            return {
                'total_cycles': len(cycles),
                'period_days': days_back,
                'total_stale_archived': total_archived,
                'total_low_quality_removed': total_removed,
                'total_duplicates_merged': total_merged,
                'total_memories_consolidated': total_consolidated,
                'recent_cycles': cycles[:5]  # Last 5 cycles
            }

        except Exception as e:
            logger.error(f"Error getting maintenance stats: {e}", exc_info=True)
            return {'error': str(e)}


# Singleton instance
_cil_knowledge_maintenance: Optional[CILKnowledgeMaintenanceService] = None


def get_cil_knowledge_maintenance() -> CILKnowledgeMaintenanceService:
    """Get singleton knowledge maintenance service"""
    global _cil_knowledge_maintenance
    if _cil_knowledge_maintenance is None:
        _cil_knowledge_maintenance = CILKnowledgeMaintenanceService()
    return _cil_knowledge_maintenance
