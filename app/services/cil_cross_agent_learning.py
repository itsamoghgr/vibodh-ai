"""
CIL Cross-Agent Learning Analyzer
Analyzes agent collaboration patterns and learns optimal coordination strategies
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from app.core.logging import logger
from app.db import get_supabase_admin_client


class CILCrossAgentLearning:
    """
    Learns from agent collaboration and coordination patterns

    Insights:
    - Which agent combinations work best together
    - Optimal agent handoff sequences
    - Redundant agent invocations
    - Coordination bottlenecks
    - Agent specialization patterns
    """

    def __init__(self):
        self.supabase = get_supabase_admin_client()

    async def analyze_agent_patterns(self, org_id: str, days_back: int = 14) -> Dict[str, Any]:
        """
        Analyze agent collaboration patterns

        Args:
            org_id: Organization ID
            days_back: Days of history to analyze

        Returns:
            Analysis results with recommendations
        """
        try:
            logger.info(f"Analyzing agent patterns for org {org_id}")

            cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

            # Get agent events and telemetry
            events_query = self.supabase.table('ai_agent_events')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff)\
                .execute()

            events = events_query.data or []

            if len(events) < 10:
                return {
                    'message': 'Insufficient agent data for analysis',
                    'event_count': len(events)
                }

            # Analyze patterns
            results = {
                'org_id': org_id,
                'period_days': days_back,
                'total_events': len(events),
                'agent_performance': await self._analyze_agent_performance(org_id, cutoff),
                'collaboration_patterns': self._analyze_collaboration_patterns(events),
                'handoff_sequences': self._analyze_handoff_sequences(events),
                'bottlenecks': self._identify_bottlenecks(events),
                'recommendations': []
            }

            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)

            # Store analysis
            await self._store_analysis(results)

            logger.info(
                f"Agent pattern analysis completed: "
                f"{len(results['recommendations'])} recommendations generated"
            )

            return results

        except Exception as e:
            logger.error(f"Error analyzing agent patterns: {e}", exc_info=True)
            return {'error': str(e)}

    async def _analyze_agent_performance(self, org_id: str, cutoff: str) -> Dict[str, Any]:
        """Analyze individual agent performance metrics"""
        try:
            # Get telemetry for agents
            telemetry_query = self.supabase.table('cil_telemetry')\
                .select('*')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff)\
                .not_.is_('agent_type', 'null')\
                .execute()

            telemetry = telemetry_query.data or []

            if not telemetry:
                return {}

            # Group by agent type
            by_agent = defaultdict(list)
            for record in telemetry:
                agent_type = record.get('agent_type')
                if agent_type:
                    by_agent[agent_type].append(record)

            # Calculate metrics per agent
            agent_metrics = {}
            for agent_type, records in by_agent.items():
                success_count = sum(1 for r in records if r.get('outcome') == 'success')
                success_rate = success_count / len(records)

                response_times = [
                    r.get('response_time_ms')
                    for r in records
                    if r.get('response_time_ms') is not None
                ]
                avg_response_time = (
                    sum(response_times) / len(response_times)
                    if response_times else 0
                )

                quality_scores = [
                    r.get('outcome_quality_score')
                    for r in records
                    if r.get('outcome_quality_score') is not None
                ]
                avg_quality = (
                    sum(quality_scores) / len(quality_scores)
                    if quality_scores else 0
                )

                agent_metrics[agent_type] = {
                    'total_invocations': len(records),
                    'success_rate': round(success_rate, 3),
                    'avg_response_time_ms': round(avg_response_time, 0),
                    'avg_quality_score': round(avg_quality, 3)
                }

            return agent_metrics

        except Exception as e:
            logger.error(f"Error analyzing agent performance: {e}", exc_info=True)
            return {}

    def _analyze_collaboration_patterns(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze which agents work together frequently"""
        try:
            # Group events by session/conversation
            by_session = defaultdict(list)
            for event in events:
                session_id = event.get('conversation_id') or event.get('session_id', 'default')
                by_session[session_id].append(event)

            # Find agent pairs that collaborate
            collaboration_pairs = defaultdict(int)
            collaboration_success = defaultdict(list)

            for session_id, session_events in by_session.items():
                agents_in_session = set(
                    e.get('agent_type')
                    for e in session_events
                    if e.get('agent_type')
                )

                if len(agents_in_session) > 1:
                    # Create pairs
                    agents_list = sorted(agents_in_session)
                    for i, agent1 in enumerate(agents_list):
                        for agent2 in agents_list[i+1:]:
                            pair = f"{agent1}+{agent2}"
                            collaboration_pairs[pair] += 1

                            # Check session success
                            session_outcomes = [
                                e.get('metadata', {}).get('outcome', 'unknown')
                                for e in session_events
                            ]
                            session_success = 'success' in session_outcomes
                            collaboration_success[pair].append(session_success)

            # Calculate collaboration metrics
            collaboration_metrics = {}
            for pair, count in collaboration_pairs.items():
                success_list = collaboration_success[pair]
                success_rate = sum(1 for s in success_list if s) / len(success_list) if success_list else 0

                collaboration_metrics[pair] = {
                    'collaboration_count': count,
                    'success_rate': round(success_rate, 3)
                }

            # Sort by collaboration frequency
            sorted_collaborations = dict(
                sorted(
                    collaboration_metrics.items(),
                    key=lambda x: x[1]['collaboration_count'],
                    reverse=True
                )[:10]  # Top 10
            )

            return {
                'total_collaborations': len(collaboration_pairs),
                'top_collaborations': sorted_collaborations
            }

        except Exception as e:
            logger.error(f"Error analyzing collaborations: {e}", exc_info=True)
            return {}

    def _analyze_handoff_sequences(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze agent handoff sequences (A → B → C)"""
        try:
            # Group by session and sort by time
            by_session = defaultdict(list)
            for event in events:
                session_id = event.get('conversation_id') or event.get('session_id', 'default')
                by_session[session_id].append(event)

            # Analyze sequences
            sequences = defaultdict(int)
            sequence_success = defaultdict(list)

            for session_id, session_events in by_session.items():
                # Sort by timestamp
                sorted_events = sorted(
                    session_events,
                    key=lambda e: e.get('created_at', '')
                )

                # Extract agent sequence
                agent_sequence = [
                    e.get('agent_type')
                    for e in sorted_events
                    if e.get('agent_type')
                ]

                if len(agent_sequence) >= 2:
                    # Create sequence string
                    seq_str = ' → '.join(agent_sequence[:5])  # Limit to 5 agents
                    sequences[seq_str] += 1

                    # Check sequence success
                    final_outcome = sorted_events[-1].get('metadata', {}).get('outcome', 'unknown')
                    sequence_success[seq_str].append(final_outcome == 'success')

            # Calculate sequence metrics
            sequence_metrics = {}
            for seq, count in sequences.items():
                success_list = sequence_success[seq]
                success_rate = sum(1 for s in success_list if s) / len(success_list) if success_list else 0

                sequence_metrics[seq] = {
                    'frequency': count,
                    'success_rate': round(success_rate, 3)
                }

            # Sort by frequency
            sorted_sequences = dict(
                sorted(
                    sequence_metrics.items(),
                    key=lambda x: x[1]['frequency'],
                    reverse=True
                )[:10]  # Top 10
            )

            return {
                'total_unique_sequences': len(sequences),
                'top_sequences': sorted_sequences
            }

        except Exception as e:
            logger.error(f"Error analyzing handoffs: {e}", exc_info=True)
            return {}

    def _identify_bottlenecks(self, events: List[Dict]) -> List[Dict[str, Any]]:
        """Identify coordination bottlenecks"""
        bottlenecks = []

        try:
            # Group by agent type
            by_agent = defaultdict(list)
            for event in events:
                agent_type = event.get('agent_type')
                if agent_type:
                    by_agent[agent_type].append(event)

            # Check for agents with high failure rates
            for agent_type, agent_events in by_agent.items():
                failures = sum(
                    1 for e in agent_events
                    if e.get('metadata', {}).get('outcome') == 'failure'
                )
                failure_rate = failures / len(agent_events) if agent_events else 0

                if failure_rate > 0.3:  # 30% failure rate
                    bottlenecks.append({
                        'type': 'high_failure_rate',
                        'agent': agent_type,
                        'failure_rate': round(failure_rate, 3),
                        'severity': 'high' if failure_rate > 0.5 else 'medium'
                    })

            # Check for agents with high response times
            for agent_type, agent_events in by_agent.items():
                response_times = [
                    e.get('metadata', {}).get('duration_ms', 0)
                    for e in agent_events
                ]

                if response_times:
                    avg_time = sum(response_times) / len(response_times)

                    if avg_time > 5000:  # 5 seconds
                        bottlenecks.append({
                            'type': 'slow_response',
                            'agent': agent_type,
                            'avg_response_ms': round(avg_time, 0),
                            'severity': 'high' if avg_time > 10000 else 'medium'
                        })

        except Exception as e:
            logger.error(f"Error identifying bottlenecks: {e}", exc_info=True)

        return bottlenecks

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from analysis"""
        recommendations = []

        try:
            # Check agent performance
            agent_perf = analysis.get('agent_performance', {})
            for agent, metrics in agent_perf.items():
                if metrics['success_rate'] < 0.7:
                    recommendations.append({
                        'type': 'improve_agent',
                        'priority': 'high',
                        'agent': agent,
                        'reason': f"Low success rate: {metrics['success_rate']:.1%}",
                        'suggestion': f"Review {agent} prompts and logic, consider additional training data"
                    })

                if metrics['avg_response_time_ms'] > 3000:
                    recommendations.append({
                        'type': 'optimize_performance',
                        'priority': 'medium',
                        'agent': agent,
                        'reason': f"Slow response time: {metrics['avg_response_time_ms']:.0f}ms",
                        'suggestion': f"Optimize {agent} execution, consider caching or parallel processing"
                    })

            # Check collaboration patterns
            collaborations = analysis.get('collaboration_patterns', {}).get('top_collaborations', {})
            for pair, metrics in collaborations.items():
                if metrics['success_rate'] > 0.85 and metrics['collaboration_count'] > 5:
                    recommendations.append({
                        'type': 'promote_collaboration',
                        'priority': 'low',
                        'agents': pair,
                        'reason': f"High success rate: {metrics['success_rate']:.1%}",
                        'suggestion': f"Consider {pair} as preferred combination for similar tasks"
                    })

            # Check bottlenecks
            bottlenecks = analysis.get('bottlenecks', [])
            for bottleneck in bottlenecks:
                recommendations.append({
                    'type': 'resolve_bottleneck',
                    'priority': bottleneck['severity'],
                    'agent': bottleneck.get('agent'),
                    'reason': bottleneck.get('type'),
                    'suggestion': f"Investigate and resolve {bottleneck['type']} for {bottleneck.get('agent')}"
                })

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)

        return recommendations

    async def _store_analysis(self, results: Dict[str, Any]):
        """Store analysis results for history"""
        try:
            record = {
                'org_id': results['org_id'],
                'analyzed_at': datetime.utcnow().isoformat(),
                'period_days': results['period_days'],
                'total_events': results['total_events'],
                'agent_performance': json.dumps(results.get('agent_performance', {})),
                'collaboration_patterns': json.dumps(results.get('collaboration_patterns', {})),
                'handoff_sequences': json.dumps(results.get('handoff_sequences', {})),
                'bottlenecks': json.dumps(results.get('bottlenecks', [])),
                'recommendations': json.dumps(results.get('recommendations', []))
            }

            self.supabase.table('cil_agent_analyses')\
                .insert(record)\
                .execute()

        except Exception as e:
            logger.error(f"Error storing analysis: {e}", exc_info=True)

    def get_analysis_history(self, org_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get historical agent analyses"""
        try:
            analyses_query = self.supabase.table('cil_agent_analyses')\
                .select('*')\
                .eq('org_id', org_id)\
                .order('analyzed_at', desc=True)\
                .limit(limit)\
                .execute()

            analyses = analyses_query.data or []

            return {
                'total': len(analyses),
                'analyses': analyses
            }

        except Exception as e:
            logger.error(f"Error getting analysis history: {e}", exc_info=True)
            return {'error': str(e)}


# Singleton instance
_cil_cross_agent_learning: Optional[CILCrossAgentLearning] = None


def get_cil_cross_agent_learning() -> CILCrossAgentLearning:
    """Get singleton cross-agent learning analyzer"""
    global _cil_cross_agent_learning
    if _cil_cross_agent_learning is None:
        _cil_cross_agent_learning = CILCrossAgentLearning()
    return _cil_cross_agent_learning
