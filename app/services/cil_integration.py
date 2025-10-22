"""
CIL Integration Helpers
Provides easy integration points for CDE and agents to consume CIL services
"""

from typing import Dict, Optional, Any
from datetime import datetime

from app.core.logging import logger
from app.services.cil_policy_service import get_cil_policy_service
from app.services.cil_telemetry_service import get_cil_telemetry_service
from app.services.cil_prompt_optimizer import get_cil_prompt_optimizer


class CILIntegration:
    """
    Integration layer for consuming CIL services

    Makes it easy for CDE and agents to:
    - Get active policy configurations
    - Record telemetry events
    - Fetch optimized prompts
    - Track usage outcomes
    """

    def __init__(self):
        self.policy_service = get_cil_policy_service()
        self.telemetry_service = get_cil_telemetry_service()
        self.prompt_optimizer = get_cil_prompt_optimizer()

    # =========================================================================
    # Policy Integration
    # =========================================================================

    def get_confidence_threshold(self, org_id: str, intent_type: str) -> float:
        """
        Get confidence threshold for an intent type

        Args:
            org_id: Organization ID
            intent_type: Intent type (question, execute, task, summary, insight, risk)

        Returns:
            Confidence threshold (0.0-1.0)
        """
        try:
            policy = self.policy_service.get_active_policy(org_id)

            if not policy:
                logger.warning(f"No policy found for org {org_id}, using default")
                policy = {'policy_config': self.policy_service.get_default_policy()}

            config = policy['policy_config']
            thresholds = config.get('confidence_thresholds', {})

            return thresholds.get(intent_type, 0.7)  # Default 0.7

        except Exception as e:
            logger.error(f"Error getting confidence threshold: {e}")
            return 0.7  # Fallback

    def get_module_weights(self, org_id: str, intent_type: str) -> Dict[str, float]:
        """
        Get module weights for an intent type

        Args:
            org_id: Organization ID
            intent_type: Intent type

        Returns:
            Dict of module weights {rag: 0.8, kg: 0.5, ...}
        """
        try:
            policy = self.policy_service.get_active_policy(org_id)

            if not policy:
                policy = {'policy_config': self.policy_service.get_default_policy()}

            config = policy['policy_config']
            module_weights = config.get('module_weights', {})

            return module_weights.get(intent_type, {})

        except Exception as e:
            logger.error(f"Error getting module weights: {e}")
            return {}

    def get_risk_sensitivity(self, org_id: str, risk_level: str) -> float:
        """
        Get risk sensitivity threshold

        Args:
            org_id: Organization ID
            risk_level: Risk level (low, medium, high, critical)

        Returns:
            Risk sensitivity (0.0-1.0)
        """
        try:
            policy = self.policy_service.get_active_policy(org_id)

            if not policy:
                policy = {'policy_config': self.policy_service.get_default_policy()}

            config = policy['policy_config']
            risk_sensitivity = config.get('risk_sensitivity', {})

            return risk_sensitivity.get(risk_level, 0.6)  # Default 0.6

        except Exception as e:
            logger.error(f"Error getting risk sensitivity: {e}")
            return 0.6

    def get_agent_config(self, org_id: str, agent_type: str) -> Dict[str, Any]:
        """
        Get agent configuration from policy

        Args:
            org_id: Organization ID
            agent_type: Agent type (communication, marketing, etc.)

        Returns:
            Agent configuration dict
        """
        try:
            policy = self.policy_service.get_active_policy(org_id)

            if not policy:
                policy = {'policy_config': self.policy_service.get_default_policy()}

            config = policy['policy_config']
            agent_preferences = config.get('agent_preferences', {})

            return agent_preferences.get(agent_type, {
                'priority': 5,
                'max_retries': 2,
                'timeout_seconds': 300
            })

        except Exception as e:
            logger.error(f"Error getting agent config: {e}")
            return {'priority': 5, 'max_retries': 2, 'timeout_seconds': 300}

    # =========================================================================
    # Telemetry Integration
    # =========================================================================

    def record_query_outcome(
        self,
        org_id: str,
        query_text: str,
        intent_type: str,
        confidence_score: float,
        outcome: str,  # 'success' or 'failure'
        modules_used: list = None,
        response_time_ms: int = None,
        quality_score: float = None,
        metadata: Dict = None
    ):
        """
        Record a query outcome to telemetry

        This is the primary integration point for CDE to report outcomes

        Args:
            org_id: Organization ID
            query_text: The user's query
            intent_type: Detected intent
            confidence_score: Intent confidence
            outcome: 'success' or 'failure'
            modules_used: List of modules (rag, kg, memory, insight)
            response_time_ms: Response time in milliseconds
            quality_score: Quality score (0.0-1.0)
            metadata: Additional metadata
        """
        try:
            # This would normally create a telemetry record directly
            # But since telemetry ingests from other tables, we just log for now
            logger.info(
                f"Query outcome recorded: org={org_id}, intent={intent_type}, "
                f"confidence={confidence_score:.2f}, outcome={outcome}"
            )

            # In production, this could write to a temporary staging table
            # that telemetry service picks up, or trigger immediate ingestion

        except Exception as e:
            logger.error(f"Error recording query outcome: {e}", exc_info=True)

    def record_agent_outcome(
        self,
        org_id: str,
        agent_type: str,
        outcome: str,
        response_time_ms: int = None,
        quality_score: float = None,
        metadata: Dict = None
    ):
        """
        Record agent execution outcome

        Args:
            org_id: Organization ID
            agent_type: Agent type
            outcome: 'success' or 'failure'
            response_time_ms: Response time
            quality_score: Quality score
            metadata: Additional context
        """
        try:
            logger.info(
                f"Agent outcome recorded: org={org_id}, agent={agent_type}, outcome={outcome}"
            )

        except Exception as e:
            logger.error(f"Error recording agent outcome: {e}", exc_info=True)

    # =========================================================================
    # Prompt Integration
    # =========================================================================

    def get_prompt(
        self,
        org_id: str,
        template_name: str,
        variables: Dict[str, str] = None,
        user_id: str = None
    ) -> Optional[str]:
        """
        Get optimized prompt template with variable substitution

        Automatically handles A/B testing selection

        Args:
            org_id: Organization ID
            template_name: Template name (e.g., 'cde_intent_detection')
            variables: Dict of variables to substitute {name: value}
            user_id: User ID for consistent A/B assignment

        Returns:
            Rendered prompt text or None
        """
        try:
            # Get template (handles A/B test selection)
            template = self.prompt_optimizer.get_prompt_for_use(
                org_id=org_id,
                template_name=template_name,
                user_id=user_id
            )

            if not template:
                logger.warning(f"Template not found: {template_name}")
                return None

            prompt_text = template['prompt_text']

            # Substitute variables
            if variables:
                for var_name, var_value in variables.items():
                    placeholder = f"{{{var_name}}}"
                    prompt_text = prompt_text.replace(placeholder, str(var_value))

            # Return template_id for usage tracking
            return {
                'template_id': template['id'],
                'prompt_text': prompt_text,
                'version': template['version']
            }

        except Exception as e:
            logger.error(f"Error getting prompt: {e}", exc_info=True)
            return None

    def record_prompt_outcome(
        self,
        template_id: str,
        outcome: str,
        response_time_ms: int = None,
        quality_score: float = None
    ):
        """
        Record prompt usage outcome for A/B testing

        Args:
            template_id: Template ID (from get_prompt response)
            outcome: 'success' or 'failure'
            response_time_ms: Response time
            quality_score: Quality score
        """
        try:
            self.prompt_optimizer.record_prompt_usage(
                template_id=template_id,
                outcome=outcome,
                response_time_ms=response_time_ms,
                quality_score=quality_score
            )

        except Exception as e:
            logger.error(f"Error recording prompt outcome: {e}", exc_info=True)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def should_route_to_module(
        self,
        org_id: str,
        intent_type: str,
        module_name: str,
        threshold: float = 0.5
    ) -> bool:
        """
        Check if a module should be used for an intent based on policy weights

        Args:
            org_id: Organization ID
            intent_type: Intent type
            module_name: Module name (rag, kg, memory, insight)
            threshold: Minimum weight to use module

        Returns:
            True if module weight >= threshold
        """
        weights = self.get_module_weights(org_id, intent_type)
        weight = weights.get(module_name, 0)
        return weight >= threshold

    def requires_approval(
        self,
        org_id: str,
        risk_level: str,
        confidence_score: float
    ) -> bool:
        """
        Determine if an action requires human approval

        Args:
            org_id: Organization ID
            risk_level: Risk level (low, medium, high, critical)
            confidence_score: Action confidence

        Returns:
            True if approval required
        """
        # High/critical risk always requires approval
        if risk_level in ['high', 'critical']:
            return True

        # Low confidence requires approval
        risk_sensitivity = self.get_risk_sensitivity(org_id, risk_level)
        if confidence_score < risk_sensitivity:
            return True

        return False


# Singleton instance
_cil_integration: Optional[CILIntegration] = None


def get_cil_integration() -> CILIntegration:
    """Get singleton CIL integration instance"""
    global _cil_integration
    if _cil_integration is None:
        _cil_integration = CILIntegration()
    return _cil_integration


# Convenience functions for direct import
def get_confidence_threshold(org_id: str, intent_type: str) -> float:
    """Get confidence threshold for intent type"""
    return get_cil_integration().get_confidence_threshold(org_id, intent_type)


def get_module_weights(org_id: str, intent_type: str) -> Dict[str, float]:
    """Get module weights for intent type"""
    return get_cil_integration().get_module_weights(org_id, intent_type)


def record_query_outcome(org_id: str, query_text: str, intent_type: str,
                        confidence_score: float, outcome: str, **kwargs):
    """Record query outcome"""
    return get_cil_integration().record_query_outcome(
        org_id, query_text, intent_type, confidence_score, outcome, **kwargs
    )


def get_optimized_prompt(org_id: str, template_name: str,
                        variables: Dict[str, str] = None, user_id: str = None):
    """Get optimized prompt with A/B testing"""
    return get_cil_integration().get_prompt(org_id, template_name, variables, user_id)
