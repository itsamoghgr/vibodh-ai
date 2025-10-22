"""
Adaptive Engine - Phase 3, Step 3
Self-optimization engine for continuous AI improvement
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from supabase import Client
from app.services.feedback_service import get_feedback_service
from app.core.logging import logger


class AdaptiveEngine:
    """
    Adaptive Engine for self-optimization and continuous learning.

    Supports:
    - Evaluating reasoning performance
    - Adjusting module weights dynamically
    - Auto-tuning LLM parameters
    - Logging optimization decisions
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.feedback_service = get_feedback_service(supabase)

    def get_adaptive_config(self, org_id: str) -> Dict[str, Any]:
        """
        Get current adaptive configuration for organization.

        Args:
            org_id: Organization ID

        Returns:
            Adaptive configuration dictionary
        """
        try:
            result = self.supabase.table("ai_adaptive_config")\
                .select("*")\
                .eq("org_id", org_id)\
                .single()\
                .execute()

            if result.data:
                return result.data
            else:
                # Create default config if not exists
                logger.info(f"Creating default adaptive config for org {org_id}")
                return self._create_default_config(org_id)

        except Exception as e:
            logger.error(f"Error getting adaptive config: {e}", exc_info=True)
            return self._create_default_config(org_id)

    def _create_default_config(self, org_id: str) -> Dict[str, Any]:
        """Create default adaptive configuration."""
        try:
            default_config = {
                "org_id": org_id,
                "rag_weight": 1.0,
                "kg_weight": 1.0,
                "memory_weight": 1.0,
                "insight_weight": 1.0,
                "llm_temperature": 0.7,
                "llm_top_p": 1.0,
                "max_context_items": 5,
                "embedding_threshold": 0.3,
                "memory_importance_threshold": 0.3,
                "target_response_time_ms": 5000,
                "target_accuracy": 0.8
            }

            result = self.supabase.table("ai_adaptive_config")\
                .insert(default_config)\
                .execute()

            return result.data[0] if result.data else default_config

        except Exception as e:
            logger.error(f"Error creating default config: {e}", exc_info=True)
            return default_config

    async def evaluate_reasoning_log(
        self,
        org_id: str,
        reasoning_log_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate a specific reasoning log for quality and performance.

        Args:
            org_id: Organization ID
            reasoning_log_id: Reasoning log ID to evaluate

        Returns:
            Evaluation results with recommendations
        """
        try:
            logger.info(f"Evaluating reasoning log {reasoning_log_id}")

            # Get the reasoning log
            log_result = self.supabase.table("reasoning_logs")\
                .select("*")\
                .eq("id", reasoning_log_id)\
                .eq("org_id", org_id)\
                .single()\
                .execute()

            if not log_result.data:
                raise Exception(f"Reasoning log {reasoning_log_id} not found")

            log = log_result.data

            # Get associated feedback if available
            feedback_result = self.supabase.table("ai_feedback_metrics")\
                .select("*")\
                .eq("reasoning_log_id", reasoning_log_id)\
                .execute()

            feedback = feedback_result.data[0] if feedback_result.data else None

            # Evaluate performance
            evaluation = {
                "log_id": reasoning_log_id,
                "intent": log["intent"],
                "modules_used": log["modules_used"],
                "execution_time_ms": log["execution_time_ms"],
                "tokens_used": log.get("tokens_used"),
                "issues": [],
                "strengths": [],
                "recommendations": []
            }

            # Check execution time
            if log["execution_time_ms"] > 5000:
                evaluation["issues"].append("Slow response time")
                evaluation["recommendations"].append({
                    "type": "performance",
                    "action": "reduce_context_items",
                    "reason": f"Response took {log['execution_time_ms']}ms (target: <5000ms)"
                })
            else:
                evaluation["strengths"].append("Fast response time")

            # Check module efficiency
            if len(log["modules_used"]) > 3:
                evaluation["issues"].append("Too many modules queried")
                evaluation["recommendations"].append({
                    "type": "efficiency",
                    "action": "optimize_module_routing",
                    "reason": f"Used {len(log['modules_used'])} modules, consider more selective routing"
                })

            # Check feedback if available
            if feedback:
                if feedback["user_feedback"] == "negative":
                    evaluation["issues"].append("Negative user feedback")

                    # Analyze which module might have failed
                    for module in log["modules_used"]:
                        evaluation["recommendations"].append({
                            "type": "accuracy",
                            "action": f"review_{module}_quality",
                            "reason": f"User gave negative feedback, check {module} module performance"
                        })
                elif feedback["user_feedback"] == "positive":
                    evaluation["strengths"].append("Positive user feedback")

                # Check confidence calibration
                if feedback.get("confidence_score") and feedback.get("accuracy_estimate"):
                    conf_diff = abs(feedback["confidence_score"] - feedback["accuracy_estimate"])
                    if conf_diff > 0.3:
                        evaluation["issues"].append("Poorly calibrated confidence")
                        evaluation["recommendations"].append({
                            "type": "calibration",
                            "action": "adjust_confidence_scoring",
                            "reason": f"Confidence-accuracy gap: {conf_diff:.2f}"
                        })

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating reasoning log: {e}", exc_info=True)
            return {"error": str(e)}

    async def adjust_module_weights(
        self,
        org_id: str,
        days_back: int = 7,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Automatically adjust module weights based on performance data.

        Args:
            org_id: Organization ID
            days_back: Number of days to analyze
            dry_run: If True, only return recommendations without applying

        Returns:
            Dictionary with adjustment results
        """
        try:
            logger.info(f"Adjusting module weights for org {org_id} (dry_run={dry_run})")

            # Get current config
            config = self.get_adaptive_config(org_id)

            # Get module success rates
            module_stats_result = self.supabase.rpc(
                "get_module_success_rate",
                {"org_uuid": org_id, "days_back": days_back}
            ).execute()

            if not module_stats_result.data or len(module_stats_result.data) == 0:
                return {
                    "adjusted": False,
                    "reason": "Insufficient data for adjustment",
                    "recommendation": "Need at least 10 feedback entries per module"
                }

            # Calculate new weights based on success rates
            adjustments = {}
            for module_stat in module_stats_result.data:
                module = module_stat["module_name"].lower()
                success_rate = module_stat["success_rate"]
                total_uses = module_stat["total_uses"]

                # Need minimum sample size
                if total_uses < 5:
                    continue

                weight_key = f"{module}_weight"
                if weight_key in config:
                    current_weight = config[weight_key]

                    # Adjust weight based on success rate
                    # High success (>80%): increase weight by 10%
                    # Medium success (60-80%): keep current
                    # Low success (<60%): decrease weight by 15%
                    if success_rate > 0.8:
                        new_weight = min(2.0, current_weight * 1.1)
                    elif success_rate < 0.6:
                        new_weight = max(0.5, current_weight * 0.85)
                    else:
                        new_weight = current_weight

                    # Only adjust if change is significant (>5%)
                    if abs(new_weight - current_weight) / current_weight > 0.05:
                        adjustments[weight_key] = {
                            "old_value": current_weight,
                            "new_value": round(new_weight, 2),
                            "reason": f"Success rate: {success_rate:.1%} over {total_uses} uses"
                        }

            if not adjustments:
                return {
                    "adjusted": False,
                    "reason": "No significant adjustments needed",
                    "current_weights": {k: v for k, v in config.items() if k.endswith("_weight")}
                }

            # Apply adjustments if not dry_run
            if not dry_run:
                update_data = {k: v["new_value"] for k, v in adjustments.items()}
                update_data["last_optimized_at"] = datetime.utcnow().isoformat()
                update_data["optimization_count"] = config.get("optimization_count", 0) + 1

                self.supabase.table("ai_adaptive_config")\
                    .update(update_data)\
                    .eq("org_id", org_id)\
                    .execute()

                # Log adjustments
                for param, change in adjustments.items():
                    self._log_optimization(
                        org_id=org_id,
                        optimization_type="module_weight",
                        parameter_name=param,
                        old_value=change["old_value"],
                        new_value=change["new_value"],
                        reason=change["reason"],
                        trigger_event="automated_analysis"
                    )

                logger.info(f"Applied {len(adjustments)} module weight adjustments")

            return {
                "adjusted": not dry_run,
                "adjustments": adjustments,
                "dry_run": dry_run
            }

        except Exception as e:
            logger.error(f"Error adjusting module weights: {e}", exc_info=True)
            return {"adjusted": False, "error": str(e)}

    async def auto_tune_prompt_parameters(
        self,
        org_id: str,
        days_back: int = 7,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Automatically tune LLM prompt parameters based on feedback.

        Args:
            org_id: Organization ID
            days_back: Number of days to analyze
            dry_run: If True, only return recommendations

        Returns:
            Dictionary with tuning results
        """
        try:
            logger.info(f"Auto-tuning prompt parameters for org {org_id}")

            # Get current config
            config = self.get_adaptive_config(org_id)

            # Get feedback summary
            summary_result = self.supabase.rpc(
                "get_feedback_summary",
                {"org_uuid": org_id, "days_back": days_back}
            ).execute()

            if not summary_result.data or summary_result.data[0]["total_interactions"] < 10:
                return {
                    "tuned": False,
                    "reason": "Insufficient feedback data (need at least 10 interactions)"
                }

            summary = summary_result.data[0]
            adjustments = {}

            # Adjust temperature based on accuracy
            avg_accuracy = summary.get("avg_accuracy") or 0.5
            current_temp = config["llm_temperature"]

            if avg_accuracy < 0.6:
                # Low accuracy: lower temperature for more focused responses
                new_temp = max(0.3, current_temp - 0.1)
                if abs(new_temp - current_temp) > 0.05:
                    adjustments["llm_temperature"] = {
                        "old_value": current_temp,
                        "new_value": round(new_temp, 2),
                        "reason": f"Low accuracy ({avg_accuracy:.1%}), reducing temperature for more focused responses"
                    }
            elif avg_accuracy > 0.85:
                # High accuracy: can increase temperature slightly for more creative responses
                new_temp = min(0.9, current_temp + 0.05)
                if abs(new_temp - current_temp) > 0.05:
                    adjustments["llm_temperature"] = {
                        "old_value": current_temp,
                        "new_value": round(new_temp, 2),
                        "reason": f"High accuracy ({avg_accuracy:.1%}), can increase temperature slightly"
                    }

            # Adjust max_context_items based on response time
            avg_time = summary.get("avg_response_time") or 3000
            current_max = config["max_context_items"]

            if avg_time > 5000:
                # Slow responses: reduce context items
                new_max = max(3, current_max - 1)
                if new_max != current_max:
                    adjustments["max_context_items"] = {
                        "old_value": current_max,
                        "new_value": new_max,
                        "reason": f"Slow average response time ({avg_time}ms), reducing context items"
                    }
            elif avg_time < 2000 and avg_accuracy < 0.75:
                # Fast but inaccurate: add more context
                new_max = min(10, current_max + 1)
                if new_max != current_max:
                    adjustments["max_context_items"] = {
                        "old_value": current_max,
                        "new_value": new_max,
                        "reason": f"Fast responses ({avg_time}ms) but low accuracy, adding more context"
                    }

            if not adjustments:
                return {
                    "tuned": False,
                    "reason": "Parameters are already optimal",
                    "current_params": {
                        "llm_temperature": config["llm_temperature"],
                        "max_context_items": config["max_context_items"]
                    }
                }

            # Apply adjustments if not dry_run
            if not dry_run:
                update_data = {k: v["new_value"] for k, v in adjustments.items()}
                update_data["last_optimized_at"] = datetime.utcnow().isoformat()

                self.supabase.table("ai_adaptive_config")\
                    .update(update_data)\
                    .eq("org_id", org_id)\
                    .execute()

                # Log adjustments
                for param, change in adjustments.items():
                    self._log_optimization(
                        org_id=org_id,
                        optimization_type="llm_parameter",
                        parameter_name=param,
                        old_value=change["old_value"],
                        new_value=change["new_value"],
                        reason=change["reason"],
                        trigger_event="automated_tuning"
                    )

                logger.info(f"Applied {len(adjustments)} parameter adjustments")

            return {
                "tuned": not dry_run,
                "adjustments": adjustments,
                "dry_run": dry_run
            }

        except Exception as e:
            logger.error(f"Error auto-tuning parameters: {e}", exc_info=True)
            return {"tuned": False, "error": str(e)}

    def _log_optimization(
        self,
        org_id: str,
        optimization_type: str,
        parameter_name: str,
        old_value: float,
        new_value: float,
        reason: str,
        trigger_event: str,
        performed_by: str = "system",
        metrics_before: Optional[Dict] = None,
        metrics_after: Optional[Dict] = None
    ):
        """Log an optimization adjustment for auditability."""
        try:
            log_data = {
                "org_id": org_id,
                "optimization_type": optimization_type,
                "parameter_name": parameter_name,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
                "trigger_event": trigger_event,
                "performed_by": performed_by,
                "metrics_before": metrics_before or {},
                "metrics_after": metrics_after or {}
            }

            self.supabase.table("ai_optimization_log").insert(log_data).execute()
            logger.info(f"Logged optimization: {parameter_name} {old_value} -> {new_value}")

        except Exception as e:
            logger.error(f"Error logging optimization: {e}", exc_info=True)

    async def run_full_optimization(
        self,
        org_id: str,
        days_back: int = 7,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run full optimization cycle: adjust weights and tune parameters.

        Args:
            org_id: Organization ID
            days_back: Number of days to analyze
            dry_run: If True, only return recommendations

        Returns:
            Complete optimization results
        """
        try:
            logger.info(f"Running full optimization for org {org_id}")

            # Adjust module weights
            weight_results = await self.adjust_module_weights(org_id, days_back, dry_run)

            # Tune LLM parameters
            param_results = await self.auto_tune_prompt_parameters(org_id, days_back, dry_run)

            # Get feedback trends for context
            trends = self.feedback_service.analyze_feedback_trends(org_id, days_back)

            return {
                "org_id": org_id,
                "optimized": not dry_run,
                "module_weights": weight_results,
                "llm_parameters": param_results,
                "trends": trends,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error running full optimization: {e}", exc_info=True)
            return {"optimized": False, "error": str(e)}

    async def ingest_pending_reflections(
        self,
        org_id: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Ingest pending agent reflections and apply learning.

        Args:
            org_id: Organization ID
            limit: Maximum number of reflections to process

        Returns:
            Ingestion results with actions taken
        """
        try:
            logger.info(f"[ADAPTIVE_ENGINE] Ingesting pending reflections for org {org_id}")

            # Fetch pending reflections
            result = self.supabase.table("ai_reflections")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("ingested_by_adaptive_engine", False)\
                .order("created_at", desc=False)\
                .limit(limit)\
                .execute()

            reflections = result.data if result.data else []

            if not reflections:
                logger.info("[ADAPTIVE_ENGINE] No pending reflections to ingest")
                return {
                    "reflections_processed": 0,
                    "actions_taken": []
                }

            logger.info(f"[ADAPTIVE_ENGINE] Processing {len(reflections)} reflections")

            actions_taken = []

            for reflection in reflections:
                reflection_actions = await self._process_reflection(reflection)
                actions_taken.extend(reflection_actions)

                # Mark reflection as ingested
                self.supabase.table("ai_reflections")\
                    .update({
                        "ingested_by_adaptive_engine": True,
                        "ingested_at": datetime.utcnow().isoformat(),
                        "adaptive_actions_taken": reflection_actions,
                        "updated_at": datetime.utcnow().isoformat()
                    })\
                    .eq("id", reflection["id"])\
                    .execute()

            logger.info(
                f"[ADAPTIVE_ENGINE] Ingested {len(reflections)} reflections, "
                f"took {len(actions_taken)} actions"
            )

            return {
                "reflections_processed": len(reflections),
                "actions_taken": actions_taken,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"[ADAPTIVE_ENGINE] Error ingesting reflections: {e}", exc_info=True)
            return {
                "reflections_processed": 0,
                "actions_taken": [],
                "error": str(e)
            }

    async def _process_reflection(
        self,
        reflection: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process a single reflection and determine actions to take.

        Args:
            reflection: Reflection data

        Returns:
            List of actions taken
        """
        actions = []

        try:
            agent_type = reflection.get("agent_type")
            overall_success = reflection.get("overall_success")
            performance_metrics = reflection.get("performance_metrics", {})
            learning_points = reflection.get("learning_points", [])
            improvements_suggested = reflection.get("improvements_suggested", [])

            # Action 1: Update agent success rate tracking
            if overall_success is not None:
                actions.append({
                    "type": "success_rate_tracked",
                    "agent_type": agent_type,
                    "success": overall_success
                })

            # Action 2: Analyze performance metrics for optimization opportunities
            if performance_metrics:
                execution_time = performance_metrics.get("total_execution_time_ms", 0)
                success_rate = performance_metrics.get("success_rate", 1.0)

                # If success rate is low, consider adjusting parameters
                if success_rate < 0.7:
                    actions.append({
                        "type": "low_success_rate_flagged",
                        "agent_type": agent_type,
                        "success_rate": success_rate,
                        "recommendation": "Consider adjusting agent parameters or providing additional context"
                    })

                # If execution time is high, flag for optimization
                if execution_time > 10000:  # > 10 seconds
                    actions.append({
                        "type": "high_execution_time_flagged",
                        "agent_type": agent_type,
                        "execution_time_ms": execution_time,
                        "recommendation": "Consider optimizing action execution or parallelizing steps"
                    })

            # Action 3: Extract and store learning points
            if learning_points:
                for point in learning_points:
                    actions.append({
                        "type": "learning_point_extracted",
                        "agent_type": agent_type,
                        "learning_point": point
                    })

            # Action 4: Process improvement suggestions
            if improvements_suggested:
                for improvement in improvements_suggested:
                    actions.append({
                        "type": "improvement_suggested",
                        "agent_type": agent_type,
                        "suggestion": improvement
                    })

            return actions

        except Exception as e:
            logger.error(f"[ADAPTIVE_ENGINE] Error processing reflection: {e}", exc_info=True)
            return []

    def get_optimization_history(
        self,
        org_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent optimization history."""
        try:
            result = self.supabase.table("ai_optimization_log")\
                .select("*")\
                .eq("org_id", org_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error getting optimization history: {e}", exc_info=True)
            return []


# Singleton instance
_adaptive_engine = None

def get_adaptive_engine(supabase: Client) -> AdaptiveEngine:
    """Get or create AdaptiveEngine instance."""
    global _adaptive_engine
    if _adaptive_engine is None:
        _adaptive_engine = AdaptiveEngine(supabase)
    return _adaptive_engine
