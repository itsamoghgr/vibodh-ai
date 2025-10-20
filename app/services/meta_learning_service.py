"""
Meta-Learning Service - Phase 3, Step 4
Enables Vibodh to learn how to learn, detect patterns, and evolve knowledge structures
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from supabase import Client
from app.core.logging import logger
from app.core.config import settings
import json


class MetaLearningService:
    """
    Meta-Learning Service for continuous AI evolution and knowledge discovery.

    Supports:
    - Analyzing reasoning patterns and discovering what works
    - Generating meta-rules for better decision making
    - Detecting trends in organizational data
    - Proposing Knowledge Graph schema evolution
    - Creating model configuration snapshots
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase

        # Initialize LLM client for meta-analysis
        from groq import Groq
        self.llm_client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = "llama-3.3-70b-versatile"

    def analyze_reasoning_patterns(
        self,
        org_id: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze which module combinations yield highest success by intent type.

        Args:
            org_id: Organization ID
            days_back: Number of days to analyze

        Returns:
            Dictionary with pattern analysis results
        """
        try:
            logger.info(f"Analyzing reasoning patterns for org {org_id} (last {days_back} days)")

            # Use the SQL function to get reasoning pattern stats
            result = self.supabase.rpc(
                'get_reasoning_pattern_stats',
                {
                    'org_uuid': org_id,
                    'days_back': days_back
                }
            ).execute()

            patterns = result.data if result.data else []

            # Group by intent
            intent_patterns = {}
            for pattern in patterns:
                intent = pattern['intent']
                if intent not in intent_patterns:
                    intent_patterns[intent] = []
                intent_patterns[intent].append({
                    'modules': pattern['modules_combination'],
                    'usage_count': pattern['usage_count'],
                    'success_rate': pattern['success_rate'],
                    'avg_response_time': pattern['avg_response_time'],
                    'avg_confidence': pattern['avg_confidence']
                })

            # Find best performing combination for each intent
            best_combinations = {}
            for intent, combos in intent_patterns.items():
                if combos:
                    best = max(combos, key=lambda x: (x['success_rate'], -x['avg_response_time']))
                    best_combinations[intent] = best

            logger.info(f"Found {len(best_combinations)} intent-specific patterns")

            return {
                'success': True,
                'patterns': patterns,
                'intent_patterns': intent_patterns,
                'best_combinations': best_combinations,
                'total_patterns': len(patterns)
            }

        except Exception as e:
            logger.error(f"Error analyzing reasoning patterns: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'patterns': [],
                'intent_patterns': {},
                'best_combinations': {},
                'total_patterns': 0
            }

    def discover_reasoning_rules(
        self,
        org_id: str,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate meta-rules based on reasoning pattern analysis.

        Args:
            org_id: Organization ID
            min_confidence: Minimum confidence threshold for rules

        Returns:
            List of discovered meta-rules
        """
        try:
            logger.info(f"Discovering reasoning rules for org {org_id}")

            # Get reasoning patterns
            pattern_analysis = self.analyze_reasoning_patterns(org_id, days_back=30)

            if not pattern_analysis['success']:
                return []

            discovered_rules = []
            best_combos = pattern_analysis['best_combinations']

            # Generate rules for each intent with strong patterns
            for intent, combo in best_combos.items():
                if combo['success_rate'] >= min_confidence and combo['usage_count'] >= 5:
                    rule_text = (
                        f"For '{intent}' queries, use {combo['modules']} "
                        f"(success rate: {combo['success_rate']:.1%}, "
                        f"avg response time: {combo['avg_response_time']:.0f}ms)"
                    )

                    discovered_rules.append({
                        'rule_text': rule_text,
                        'category': 'reasoning',
                        'confidence': combo['success_rate'],
                        'metadata': {
                            'intent': intent,
                            'modules': combo['modules'],
                            'usage_count': combo['usage_count'],
                            'avg_response_time': combo['avg_response_time'],
                            'avg_confidence': combo['avg_confidence']
                        }
                    })

            # Store new rules in database
            for rule in discovered_rules:
                # Check if similar rule already exists
                existing = self.supabase.table('ai_meta_knowledge')\
                    .select('id, application_count, success_rate')\
                    .eq('org_id', org_id)\
                    .eq('rule_text', rule['rule_text'])\
                    .execute()

                if existing.data:
                    # Update existing rule
                    logger.info(f"Updating existing rule: {rule['rule_text'][:50]}...")
                else:
                    # Insert new rule
                    rule['org_id'] = org_id
                    self.supabase.table('ai_meta_knowledge').insert(rule).execute()
                    logger.info(f"Created new rule: {rule['rule_text'][:50]}...")

            logger.info(f"Discovered {len(discovered_rules)} reasoning rules")
            return discovered_rules

        except Exception as e:
            logger.error(f"Error discovering reasoning rules: {e}", exc_info=True)
            return []

    def detect_trends_in_data(
        self,
        org_id: str,
        days_back: int = 30,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Find recurring concepts, new topics, and emerging entities.

        Args:
            org_id: Organization ID
            days_back: Number of days to analyze
            limit: Maximum number of trends to return

        Returns:
            Dictionary with detected trends
        """
        try:
            logger.info(f"Detecting trends for org {org_id} (last {days_back} days)")

            cutoff_date = datetime.now() - timedelta(days=days_back)

            # 1. Get entity growth stats
            entity_stats = self.supabase.rpc(
                'get_entity_discovery_stats',
                {
                    'org_uuid': org_id,
                    'days_back': days_back
                }
            ).execute()

            # 2. Get recent documents for topic analysis
            recent_docs = self.supabase.table('documents')\
                .select('id, title, content, metadata, created_at')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff_date.isoformat())\
                .order('created_at', desc=True)\
                .limit(100)\
                .execute()

            # 3. Get recent entities
            recent_entities = self.supabase.table('kg_entities')\
                .select('name, type, created_at')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff_date.isoformat())\
                .order('created_at', desc=True)\
                .limit(50)\
                .execute()

            # 4. Analyze with LLM to identify trends
            trends_prompt = self._build_trends_analysis_prompt(
                entity_stats.data or [],
                recent_docs.data or [],
                recent_entities.data or []
            )

            llm_response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI that analyzes organizational data patterns and identifies emerging trends, topics, and insights."
                    },
                    {
                        "role": "user",
                        "content": trends_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=800
            )

            trends_analysis = llm_response.choices[0].message.content

            # Store as meta-knowledge
            self.supabase.table('ai_meta_knowledge').insert({
                'org_id': org_id,
                'rule_text': f"Trend Analysis ({datetime.now().strftime('%Y-%m-%d')}): {trends_analysis[:200]}...",
                'category': 'discovery',
                'confidence': 0.75,
                'metadata': {
                    'full_analysis': trends_analysis,
                    'entity_stats': entity_stats.data,
                    'recent_entity_count': len(recent_entities.data or []),
                    'recent_doc_count': len(recent_docs.data or [])
                }
            }).execute()

            logger.info(f"Detected trends and stored analysis")

            return {
                'success': True,
                'trends_analysis': trends_analysis,
                'entity_growth': entity_stats.data or [],
                'recent_entities': recent_entities.data or [],
                'recent_docs_count': len(recent_docs.data or [])
            }

        except Exception as e:
            logger.error(f"Error detecting trends: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'trends_analysis': '',
                'entity_growth': [],
                'recent_entities': []
            }

    def _build_trends_analysis_prompt(
        self,
        entity_stats: List[Dict],
        recent_docs: List[Dict],
        recent_entities: List[Dict]
    ) -> str:
        """Build LLM prompt for trend analysis."""

        prompt = f"""Analyze the following organizational data and identify key trends, emerging topics, and insights.

**Entity Growth (last 30 days):**
{self._format_entity_stats(entity_stats)}

**Recent Entities Discovered:**
{self._format_recent_entities(recent_entities[:20])}

**Recent Documents:**
{self._format_recent_docs(recent_docs[:15])}

**Analysis Task:**
1. Identify 3-5 emerging trends or topics
2. Highlight any significant changes or patterns
3. Note recurring themes or concepts
4. Suggest areas that might need attention

Provide a concise analysis (3-5 sentences per point)."""

        return prompt

    def _format_entity_stats(self, stats: List[Dict]) -> str:
        """Format entity statistics for prompt."""
        if not stats:
            return "No entity data available"

        lines = []
        for stat in stats[:10]:
            lines.append(
                f"- {stat['entity_type']}: {stat['total_count']} total, "
                f"{stat['recent_count']} new ({stat['growth_rate']:.1%} growth)"
            )
        return "\n".join(lines)

    def _format_recent_entities(self, entities: List[Dict]) -> str:
        """Format recent entities for prompt."""
        if not entities:
            return "No recent entities"

        lines = []
        for entity in entities:
            lines.append(f"- {entity['name']} ({entity['type']})")
        return "\n".join(lines)

    def _format_recent_docs(self, docs: List[Dict]) -> str:
        """Format recent documents for prompt."""
        if not docs:
            return "No recent documents"

        lines = []
        for doc in docs:
            title = doc.get('title', 'Untitled')[:60]
            lines.append(f"- {title}")
        return "\n".join(lines)

    def generate_meta_knowledge(
        self,
        org_id: str
    ) -> Dict[str, Any]:
        """
        Run complete meta-learning analysis: patterns, rules, and trends.

        Args:
            org_id: Organization ID

        Returns:
            Combined results from all meta-learning analyses
        """
        try:
            logger.info(f"Generating meta-knowledge for org {org_id}")

            # 1. Analyze reasoning patterns
            patterns = self.analyze_reasoning_patterns(org_id, days_back=30)

            # 2. Discover reasoning rules
            rules = self.discover_reasoning_rules(org_id, min_confidence=0.7)

            # 3. Detect data trends
            trends = self.detect_trends_in_data(org_id, days_back=30)

            return {
                'success': True,
                'patterns': patterns,
                'rules_discovered': len(rules),
                'rules': rules,
                'trends': trends,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating meta-knowledge: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'patterns': {},
                'rules_discovered': 0,
                'rules': [],
                'trends': {}
            }

    def create_model_snapshot(
        self,
        org_id: str,
        snapshot_type: str = 'manual',
        notes: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a snapshot of current model configuration and performance.

        Args:
            org_id: Organization ID
            snapshot_type: Type of snapshot (pre_optimization, post_optimization, manual)
            notes: Optional notes

        Returns:
            Snapshot ID if successful, None otherwise
        """
        try:
            logger.info(f"Creating model snapshot for org {org_id} (type: {snapshot_type})")

            # Get current adaptive config
            config_result = self.supabase.table('ai_adaptive_config')\
                .select('*')\
                .eq('org_id', org_id)\
                .single()\
                .execute()

            if not config_result.data:
                logger.warning(f"No adaptive config found for org {org_id}")
                return None

            config = config_result.data

            # Get recent performance metrics
            metrics_result = self.supabase.rpc(
                'get_feedback_summary',
                {
                    'org_uuid': org_id,
                    'days_back': 7
                }
            ).execute()

            metrics = metrics_result.data[0] if metrics_result.data else {}

            # Create snapshot
            snapshot_data = {
                'org_id': org_id,
                'snapshot_type': snapshot_type,
                'parameters_json': {
                    'rag_weight': config.get('rag_weight'),
                    'kg_weight': config.get('kg_weight'),
                    'memory_weight': config.get('memory_weight'),
                    'insight_weight': config.get('insight_weight'),
                    'llm_temperature': config.get('llm_temperature'),
                    'llm_top_p': config.get('llm_top_p'),
                    'max_context_items': config.get('max_context_items'),
                    'embedding_threshold': config.get('embedding_threshold'),
                    'memory_importance_threshold': config.get('memory_importance_threshold')
                },
                'performance_metrics_json': {
                    'total_interactions': metrics.get('total_interactions', 0),
                    'positive_feedback': metrics.get('positive_feedback', 0),
                    'negative_feedback': metrics.get('negative_feedback', 0),
                    'avg_confidence': metrics.get('avg_confidence', 0),
                    'avg_response_time': metrics.get('avg_response_time', 0),
                    'avg_accuracy': metrics.get('avg_accuracy', 0)
                },
                'optimization_cycle': config.get('optimization_count', 0),
                'notes': notes,
                'created_by': 'system'
            }

            result = self.supabase.table('ai_model_snapshots')\
                .insert(snapshot_data)\
                .execute()

            snapshot_id = result.data[0]['id'] if result.data else None
            logger.info(f"Created snapshot {snapshot_id}")

            return snapshot_id

        except Exception as e:
            logger.error(f"Error creating model snapshot: {e}", exc_info=True)
            return None

    def get_applicable_meta_rules(
        self,
        org_id: str,
        intent: Optional[str] = None,
        min_confidence: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Get meta-rules applicable to current query context.

        Args:
            org_id: Organization ID
            intent: Query intent to filter rules
            min_confidence: Minimum confidence threshold

        Returns:
            List of applicable meta-rules
        """
        try:
            query = self.supabase.table('ai_meta_knowledge')\
                .select('*')\
                .eq('org_id', org_id)\
                .eq('category', 'reasoning')\
                .gte('confidence', min_confidence)\
                .order('success_rate', desc=True)\
                .order('confidence', desc=True)\
                .limit(10)

            result = query.execute()
            rules = result.data or []

            # Filter by intent if provided
            if intent:
                rules = [
                    rule for rule in rules
                    if intent.lower() in rule.get('metadata', {}).get('intent', '').lower()
                ]

            return rules

        except Exception as e:
            logger.error(f"Error getting applicable meta-rules: {e}", exc_info=True)
            return []


# Singleton pattern
_meta_learning_service_instance = None


def get_meta_learning_service(supabase: Client) -> MetaLearningService:
    """Get or create MetaLearningService instance."""
    global _meta_learning_service_instance
    if _meta_learning_service_instance is None:
        _meta_learning_service_instance = MetaLearningService(supabase)
    return _meta_learning_service_instance
