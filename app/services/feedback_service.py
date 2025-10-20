"""
Feedback Service - Phase 3, Step 3
Analyzes user feedback and reasoning performance for adaptive optimization
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from supabase import Client
from app.core.logging import logger


class FeedbackService:
    """
    Feedback Service for analyzing AI performance and user satisfaction.

    Supports:
    - Recording feedback with performance metrics
    - Analyzing feedback trends
    - Computing accuracy estimates
    - Generating optimization recommendations
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase

    async def record_feedback(
        self,
        org_id: str,
        query: str,
        intent: str,
        modules_used: List[str],
        user_feedback: Optional[str] = None,
        feedback_comment: Optional[str] = None,
        response_time_ms: int = 0,
        token_usage: Optional[int] = None,
        confidence_score: float = 0.5,
        context_relevance_score: Optional[float] = None,
        context_items_used: Optional[int] = None,
        reasoning_log_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Record feedback with performance metrics.

        Args:
            org_id: Organization ID
            query: User's query
            intent: Classified intent
            modules_used: List of modules that were used
            user_feedback: 'positive', 'negative', or 'neutral'
            feedback_comment: Optional comment from user
            response_time_ms: Response latency in milliseconds
            token_usage: Number of tokens used
            confidence_score: AI's confidence in response (0-1)
            context_relevance_score: Relevance of retrieved context (0-1)
            context_items_used: Number of context items used
            reasoning_log_id: Reference to reasoning_logs table
            user_id: User who provided feedback
            metadata: Additional metadata

        Returns:
            Created feedback metric record
        """
        try:
            logger.info(f"Recording feedback for org {org_id}, intent: {intent}")

            # Calculate accuracy estimate using database function
            accuracy_estimate = None
            if user_feedback:
                # We'll calculate this in Python since the DB function is for SELECT
                feedback_weight = {
                    'positive': 1.0,
                    'negative': 0.0,
                    'neutral': 0.5
                }.get(user_feedback, 0.5)

                accuracy_estimate = (
                    0.4 * feedback_weight +
                    0.3 * confidence_score +
                    0.3 * (context_relevance_score or 0.5)
                )
                accuracy_estimate = max(0.0, min(1.0, accuracy_estimate))

            feedback_data = {
                "org_id": org_id,
                "user_id": user_id,
                "query": query,
                "intent": intent,
                "modules_used": modules_used,
                "user_feedback": user_feedback,
                "feedback_comment": feedback_comment,
                "response_time_ms": response_time_ms,
                "token_usage": token_usage,
                "confidence_score": max(0.0, min(1.0, confidence_score)),
                "accuracy_estimate": accuracy_estimate,
                "context_relevance_score": context_relevance_score,
                "context_items_used": context_items_used,
                "reasoning_log_id": reasoning_log_id,
                "metadata": metadata or {}
            }

            result = self.supabase.table("ai_feedback_metrics").insert(feedback_data).execute()

            if result.data:
                logger.info(f"Feedback recorded successfully: {result.data[0]['id']}")
                return result.data[0]
            else:
                raise Exception("Failed to record feedback")

        except Exception as e:
            logger.error(f"Error recording feedback: {e}", exc_info=True)
            raise

    def analyze_feedback_trends(
        self,
        org_id: str,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze feedback trends for optimization insights.

        Args:
            org_id: Organization ID
            days_back: Number of days to analyze

        Returns:
            Dictionary with trend analysis
        """
        try:
            logger.info(f"Analyzing feedback trends for org {org_id}, past {days_back} days")

            # Get module success rates
            module_stats = self.supabase.rpc(
                "get_module_success_rate",
                {"org_uuid": org_id, "days_back": days_back}
            ).execute()

            # Get response time by intent
            response_times = self.supabase.rpc(
                "get_avg_response_time",
                {"org_uuid": org_id, "days_back": days_back}
            ).execute()

            # Get overall feedback summary
            summary = self.supabase.rpc(
                "get_feedback_summary",
                {"org_uuid": org_id, "days_back": days_back}
            ).execute()

            # Identify underperforming modules
            underperforming = []
            if module_stats.data:
                for module in module_stats.data:
                    if module['success_rate'] < 0.6:  # Less than 60% success
                        underperforming.append({
                            "module": module['module_name'],
                            "success_rate": module['success_rate'],
                            "recommendation": self._generate_module_recommendation(
                                module['module_name'],
                                module['success_rate']
                            )
                        })

            # Identify slow intents
            slow_intents = []
            if response_times.data:
                for intent_data in response_times.data:
                    if intent_data['avg_response_time_ms'] > 5000:  # Slower than 5 seconds
                        slow_intents.append({
                            "intent": intent_data['intent'],
                            "avg_time_ms": intent_data['avg_response_time_ms'],
                            "recommendation": "Consider reducing max_context_items or caching frequent queries"
                        })

            return {
                "module_stats": module_stats.data or [],
                "response_times": response_times.data or [],
                "summary": summary.data[0] if summary.data else {},
                "underperforming_modules": underperforming,
                "slow_intents": slow_intents,
                "recommendations": self._generate_optimization_recommendations(
                    module_stats.data,
                    response_times.data,
                    summary.data[0] if summary.data else {}
                )
            }

        except Exception as e:
            logger.error(f"Error analyzing feedback trends: {e}", exc_info=True)
            return {
                "module_stats": [],
                "response_times": [],
                "summary": {},
                "underperforming_modules": [],
                "slow_intents": [],
                "recommendations": []
            }

    def _generate_module_recommendation(self, module_name: str, success_rate: float) -> str:
        """Generate recommendation for underperforming module."""
        recommendations = {
            "rag": "Consider lowering embedding_threshold to retrieve more context, or improve document quality",
            "kg": "Review entity extraction quality and relationship confidence thresholds",
            "memory": "Increase memory_importance_threshold to focus on high-quality memories",
            "insight": "Regenerate insights with more recent data or adjust insight categories"
        }
        return recommendations.get(module_name.lower(), "Review module configuration and data quality")

    def _generate_optimization_recommendations(
        self,
        module_stats: Optional[List[Dict]],
        response_times: Optional[List[Dict]],
        summary: Dict
    ) -> List[Dict[str, Any]]:
        """Generate actionable optimization recommendations."""
        recommendations = []

        # Check overall success rate
        if summary:
            total = summary.get('total_interactions', 0)
            positive = summary.get('positive_feedback', 0)
            negative = summary.get('negative_feedback', 0)

            if total > 10:  # Need statistically significant data
                success_rate = positive / total if total > 0 else 0

                if success_rate < 0.7:  # Less than 70% success
                    recommendations.append({
                        "type": "accuracy",
                        "priority": "high",
                        "action": "adjust_module_weights",
                        "description": f"Overall success rate is {success_rate:.1%}. Consider rebalancing module weights.",
                        "params": {"target_success_rate": 0.75}
                    })

            # Check response time
            avg_time = summary.get('avg_response_time', 0)
            if avg_time > 4000:  # Slower than 4 seconds
                recommendations.append({
                    "type": "performance",
                    "priority": "medium",
                    "action": "reduce_context_items",
                    "description": f"Average response time is {avg_time}ms. Consider reducing max_context_items.",
                    "params": {"suggested_max_items": 3}
                })

        # Module-specific recommendations
        if module_stats:
            for module in module_stats:
                if module['success_rate'] < 0.5:
                    recommendations.append({
                        "type": "module_optimization",
                        "priority": "high",
                        "action": f"optimize_{module['module_name']}_module",
                        "description": f"{module['module_name']} module has {module['success_rate']:.1%} success rate",
                        "params": {"module": module['module_name'], "current_rate": module['success_rate']}
                    })

        return recommendations

    def get_recent_feedback(
        self,
        org_id: str,
        limit: int = 20,
        include_neutral: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get recent feedback entries.

        Args:
            org_id: Organization ID
            limit: Maximum number of entries
            include_neutral: Whether to include neutral/null feedback

        Returns:
            List of feedback entries
        """
        try:
            query = self.supabase.table("ai_feedback_metrics")\
                .select("*")\
                .eq("org_id", org_id)\
                .order("created_at", desc=True)\
                .limit(limit)

            if not include_neutral:
                query = query.in_("user_feedback", ["positive", "negative"])

            result = query.execute()
            return result.data or []

        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}", exc_info=True)
            return []

    def calculate_confidence_adjustment(
        self,
        org_id: str,
        days_back: int = 7
    ) -> Dict[str, float]:
        """
        Calculate recommended confidence adjustments based on accuracy.

        Args:
            org_id: Organization ID
            days_back: Number of days to analyze

        Returns:
            Dictionary with confidence adjustment recommendations
        """
        try:
            # Get feedback with confidence scores
            result = self.supabase.table("ai_feedback_metrics")\
                .select("confidence_score, accuracy_estimate, user_feedback")\
                .eq("org_id", org_id)\
                .gte("created_at", (datetime.utcnow() - timedelta(days=days_back)).isoformat())\
                .not_.is_("accuracy_estimate", "null")\
                .execute()

            if not result.data or len(result.data) < 5:
                return {"adjustment": 0.0, "reason": "Insufficient data"}

            # Calculate average overconfidence/underconfidence
            total_diff = 0
            count = 0

            for entry in result.data:
                confidence = entry['confidence_score']
                accuracy = entry['accuracy_estimate']

                # Overconfidence: AI thinks it's right but it's wrong
                # Underconfidence: AI thinks it's wrong but it's right
                diff = confidence - accuracy
                total_diff += diff
                count += 1

            avg_diff = total_diff / count if count > 0 else 0

            # Recommend adjustment
            adjustment = -avg_diff * 0.5  # Dampen the correction

            return {
                "adjustment": round(adjustment, 3),
                "reason": f"Average confidence-accuracy gap: {avg_diff:.3f}",
                "sample_size": count
            }

        except Exception as e:
            logger.error(f"Error calculating confidence adjustment: {e}", exc_info=True)
            return {"adjustment": 0.0, "reason": "Error in calculation"}


# Singleton instance
_feedback_service = None

def get_feedback_service(supabase: Client) -> FeedbackService:
    """Get or create FeedbackService instance."""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService(supabase)
    return _feedback_service
