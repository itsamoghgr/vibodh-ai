"""
Safety Service - Phase 4, Step 1
Implements safety checks and approval mechanisms for agent actions
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from supabase import Client

from app.agents.base_agent import ActionStep, ActionPlan
from app.models.agent import RiskLevel, ActionStatus
from app.core.logging import logger


class SafetyService:
    """
    Service for implementing safety checks and approval mechanisms.

    Responsibilities:
    - Risk assessment for actions
    - Determine approval requirements
    - Validate actions against safety rules
    - Manage organization-specific safety policies
    """

    def __init__(self, supabase: Client):
        """
        Initialize safety service.

        Args:
            supabase: Supabase client
        """
        self.supabase = supabase

        # Default safety rules
        self.default_rules = {
            "high_risk_keywords": [
                "delete", "remove", "terminate", "cancel", "refund",
                "payment", "transfer", "credential", "password", "api_key"
            ],
            "critical_risk_keywords": [
                "production", "customer_data", "financial", "legal",
                "compliance", "security", "private", "confidential"
            ],
            "auto_approve_integrations": ["slack"],  # Low risk integrations
            "require_approval_integrations": ["hubspot", "stripe", "aws"],  # High risk
            "max_auto_approve_cost": 100.0,  # Maximum cost for auto-approval
            "max_auto_approve_users": 10  # Maximum affected users for auto-approval
        }

        logger.info("[SAFETY_SERVICE] Service initialized")

    async def assess_risk_level(
        self,
        action: ActionStep,
        org_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[RiskLevel, List[str]]:
        """
        Assess the risk level of an action.

        Args:
            action: Action step to assess
            org_id: Organization ID
            context: Additional context for assessment

        Returns:
            Tuple of (risk_level, reasons)
        """
        try:
            logger.info(f"[SAFETY_SERVICE] Assessing risk for action: {action.action_name}")

            risk_factors = []
            risk_score = 0

            # Check action description for risk keywords
            description_lower = (action.description or "").lower()
            action_name_lower = action.action_name.lower()
            combined_text = f"{description_lower} {action_name_lower}"

            # For Slack messages, remove channel names to avoid false positives
            # (e.g., "#private-ch" shouldn't trigger "private" keyword)
            if action.action_type == "send_message" and action.target_integration == "slack":
                # Remove channel patterns like #channel-name or "private-ch channel"
                import re
                combined_text = re.sub(r'#[a-zA-Z0-9-_]+', '', combined_text)
                combined_text = re.sub(r'[a-zA-Z0-9-_]+\s+channel', '', combined_text)

            # Check for critical risk keywords
            for keyword in self.default_rules["critical_risk_keywords"]:
                if keyword in combined_text:
                    risk_factors.append(f"Contains critical keyword: {keyword}")
                    risk_score += 30

            # Check for high risk keywords
            for keyword in self.default_rules["high_risk_keywords"]:
                if keyword in combined_text:
                    risk_factors.append(f"Contains high-risk keyword: {keyword}")
                    risk_score += 20

            # Check integration risk
            if action.target_integration:
                if action.target_integration in self.default_rules["require_approval_integrations"]:
                    risk_factors.append(f"High-risk integration: {action.target_integration}")
                    risk_score += 25
                elif action.target_integration not in self.default_rules["auto_approve_integrations"]:
                    risk_factors.append(f"Unknown integration: {action.target_integration}")
                    risk_score += 15

            # Check action type risk
            # Exclude simple messaging actions from Slack
            if action.action_type == "send_message" and action.target_integration == "slack":
                # Simple Slack messaging is low risk - don't add points
                pass
            else:
                risky_action_types = ["delete", "update", "create", "send", "post", "execute"]
                for risky_type in risky_action_types:
                    if risky_type in action.action_type.lower():
                        risk_factors.append(f"Risky action type: {action.action_type}")
                        risk_score += 10
                        break

            # Check target resource scope
            if action.target_resource:
                # Check if affecting multiple resources
                resource_count = action.target_resource.get("count", 1)
                if resource_count > 10:
                    risk_factors.append(f"Affects {resource_count} resources")
                    risk_score += 20
                elif resource_count > 1:
                    risk_factors.append(f"Affects multiple resources: {resource_count}")
                    risk_score += 10

                # Check if affecting users
                affected_users = action.target_resource.get("affected_users", 0)
                if affected_users > self.default_rules["max_auto_approve_users"]:
                    risk_factors.append(f"Affects {affected_users} users")
                    risk_score += 25

            # Check parameters for sensitive data
            if action.parameters:
                sensitive_params = ["password", "token", "key", "secret", "credential"]
                for param_key in action.parameters.keys():
                    if any(sensitive in param_key.lower() for sensitive in sensitive_params):
                        risk_factors.append(f"Contains sensitive parameter: {param_key}")
                        risk_score += 30

                # Check for cost implications
                cost = action.parameters.get("cost", 0) or action.parameters.get("amount", 0)
                if cost > self.default_rules["max_auto_approve_cost"]:
                    risk_factors.append(f"High cost: ${cost}")
                    risk_score += 25

            # Apply context-based adjustments
            if context:
                # Check if user has approval history
                user_trust_level = context.get("user_trust_level", "normal")
                if user_trust_level == "high":
                    risk_score -= 10
                elif user_trust_level == "low":
                    risk_score += 10

                # Check time of day (actions at unusual hours are riskier)
                current_hour = datetime.utcnow().hour
                if current_hour < 6 or current_hour > 22:
                    risk_factors.append("Action requested outside business hours")
                    risk_score += 5

            # Get organization-specific risk adjustments
            org_rules = await self._get_org_safety_rules(org_id)
            if org_rules:
                risk_score = self._apply_org_rules(risk_score, action, org_rules)

            # Determine final risk level based on score
            if risk_score >= 60:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 40:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 20:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            logger.info(f"[SAFETY_SERVICE] Risk assessment: {risk_level.value} (score: {risk_score})")
            logger.info(f"[SAFETY_SERVICE] Risk factors: {risk_factors}")

            return risk_level, risk_factors

        except Exception as e:
            logger.error(f"[SAFETY_SERVICE] Risk assessment failed: {e}")
            # Default to high risk on error
            return RiskLevel.HIGH, [f"Risk assessment error: {str(e)}"]

    async def requires_approval(
        self,
        action: ActionStep,
        risk_level: RiskLevel,
        org_id: str
    ) -> bool:
        """
        Determine if an action requires human approval.

        Args:
            action: Action step
            risk_level: Assessed risk level
            org_id: Organization ID

        Returns:
            True if approval required
        """
        try:
            # Always require approval for critical risk
            if risk_level == RiskLevel.CRITICAL:
                return True

            # Check if action explicitly requires approval
            if action.requires_approval:
                return True

            # Get organization approval thresholds
            thresholds = await self._get_approval_thresholds(org_id)

            # Check risk level against threshold
            risk_values = {
                RiskLevel.LOW: 0,
                RiskLevel.MEDIUM: 1,
                RiskLevel.HIGH: 2,
                RiskLevel.CRITICAL: 3
            }

            threshold_value = risk_values.get(thresholds.get("risk_threshold", RiskLevel.MEDIUM), 1)
            action_risk_value = risk_values[risk_level]

            requires_approval = action_risk_value >= threshold_value

            logger.info(f"[SAFETY_SERVICE] Approval required: {requires_approval} (risk: {risk_level.value})")

            return requires_approval

        except Exception as e:
            logger.error(f"[SAFETY_SERVICE] Approval check failed: {e}")
            # Default to requiring approval on error
            return True

    async def validate_action(
        self,
        action: ActionStep,
        org_id: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate an action against safety rules.

        Args:
            action: Action to validate
            org_id: Organization ID

        Returns:
            Tuple of (is_valid, validation_errors)
        """
        try:
            logger.info(f"[SAFETY_SERVICE] Validating action: {action.action_name}")

            validation_errors = []

            # Check for required fields
            if not action.action_type:
                validation_errors.append("Action type is required")

            if not action.action_name:
                validation_errors.append("Action name is required")

            # Check for blocked actions
            blocked_actions = await self._get_blocked_actions(org_id)
            if action.action_type in blocked_actions:
                validation_errors.append(f"Action type '{action.action_type}' is blocked")

            # Check integration permissions
            if action.target_integration:
                allowed_integrations = await self._get_allowed_integrations(org_id)
                if allowed_integrations and action.target_integration not in allowed_integrations:
                    validation_errors.append(f"Integration '{action.target_integration}' not allowed")

            # Validate parameters
            if action.parameters:
                # Check for required parameters based on action type
                required_params = self._get_required_parameters(action.action_type)
                for param in required_params:
                    if param not in action.parameters:
                        validation_errors.append(f"Required parameter missing: {param}")

            # Check resource access permissions
            if action.target_resource:
                resource_type = action.target_resource.get("type")
                if resource_type:
                    allowed_resources = await self._get_allowed_resources(org_id)
                    if allowed_resources and resource_type not in allowed_resources:
                        validation_errors.append(f"Resource type '{resource_type}' not allowed")

            is_valid = len(validation_errors) == 0

            logger.info(f"[SAFETY_SERVICE] Validation result: {is_valid}")
            if validation_errors:
                logger.warning(f"[SAFETY_SERVICE] Validation errors: {validation_errors}")

            return is_valid, validation_errors

        except Exception as e:
            logger.error(f"[SAFETY_SERVICE] Validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]

    async def validate_plan(
        self,
        plan: ActionPlan,
        org_id: str
    ) -> Tuple[bool, RiskLevel, List[str]]:
        """
        Validate an entire action plan.

        Args:
            plan: Action plan to validate
            org_id: Organization ID

        Returns:
            Tuple of (is_valid, overall_risk_level, issues)
        """
        try:
            logger.info(f"[SAFETY_SERVICE] Validating plan for goal: {plan.goal}")

            issues = []
            max_risk_level = RiskLevel.LOW
            requires_approval = False

            # Validate each step
            for step in plan.steps:
                # Validate action
                is_valid, validation_errors = await self.validate_action(step, org_id)
                if not is_valid:
                    issues.extend([f"Step {step.step_index}: {error}" for error in validation_errors])

                # Assess risk
                risk_level, risk_factors = await self.assess_risk_level(step, org_id)

                # Track maximum risk
                if self._compare_risk_levels(risk_level, max_risk_level) > 0:
                    max_risk_level = risk_level

                # Check approval requirements
                if await self.requires_approval(step, risk_level, org_id):
                    requires_approval = True
                    step.requires_approval = True

            # Update plan risk level and approval requirement
            plan.risk_level = max_risk_level.value
            plan.requires_approval = requires_approval

            is_valid = len(issues) == 0

            logger.info(f"[SAFETY_SERVICE] Plan validation: valid={is_valid}, risk={max_risk_level.value}")

            return is_valid, max_risk_level, issues

        except Exception as e:
            logger.error(f"[SAFETY_SERVICE] Plan validation failed: {e}")
            return False, RiskLevel.HIGH, [f"Plan validation error: {str(e)}"]

    async def get_approval_rules(self, org_id: str) -> Dict[str, Any]:
        """
        Get organization-specific approval rules.

        Args:
            org_id: Organization ID

        Returns:
            Approval rules dictionary
        """
        try:
            # Try to get org-specific rules from database
            result = self.supabase.table("organizations")\
                .select("settings")\
                .eq("id", org_id)\
                .single()\
                .execute()

            if result.data and result.data.get("settings"):
                safety_settings = result.data["settings"].get("safety", {})
                if safety_settings.get("approval_rules"):
                    return safety_settings["approval_rules"]

            # Return default rules
            return {
                "risk_threshold": RiskLevel.MEDIUM.value,
                "auto_approve_low_risk": True,
                "require_approval_for_production": True,
                "require_approval_for_financial": True,
                "max_auto_approve_cost": self.default_rules["max_auto_approve_cost"],
                "max_auto_approve_users": self.default_rules["max_auto_approve_users"],
                "approval_timeout_minutes": 60,
                "escalation_enabled": False
            }

        except Exception as e:
            logger.error(f"[SAFETY_SERVICE] Failed to get approval rules: {e}")
            return {}

    async def _get_org_safety_rules(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get organization-specific safety rules."""
        try:
            result = self.supabase.table("organizations")\
                .select("settings")\
                .eq("id", org_id)\
                .single()\
                .execute()

            if result.data and result.data.get("settings"):
                return result.data["settings"].get("safety", {})

            return None

        except Exception as e:
            logger.error(f"[SAFETY_SERVICE] Failed to get org safety rules: {e}")
            return None

    def _apply_org_rules(
        self,
        risk_score: int,
        action: ActionStep,
        org_rules: Dict[str, Any]
    ) -> int:
        """Apply organization-specific rules to risk score."""
        # Apply custom risk adjustments
        if "risk_adjustments" in org_rules:
            for adjustment in org_rules["risk_adjustments"]:
                if adjustment["condition"] in action.action_type.lower():
                    risk_score += adjustment["score_change"]

        # Apply integration-specific rules
        if action.target_integration and "integration_risks" in org_rules:
            if action.target_integration in org_rules["integration_risks"]:
                risk_score += org_rules["integration_risks"][action.target_integration]

        return max(0, min(100, risk_score))  # Clamp between 0 and 100

    async def _get_approval_thresholds(self, org_id: str) -> Dict[str, Any]:
        """Get approval thresholds for an organization."""
        rules = await self.get_approval_rules(org_id)
        return {
            "risk_threshold": RiskLevel(rules.get("risk_threshold", RiskLevel.MEDIUM.value))
        }

    async def _get_blocked_actions(self, org_id: str) -> List[str]:
        """Get list of blocked action types for an organization."""
        rules = await self._get_org_safety_rules(org_id)
        if rules and "blocked_actions" in rules:
            return rules["blocked_actions"]
        return []

    async def _get_allowed_integrations(self, org_id: str) -> Optional[List[str]]:
        """Get list of allowed integrations for an organization."""
        rules = await self._get_org_safety_rules(org_id)
        if rules and "allowed_integrations" in rules:
            return rules["allowed_integrations"]
        return None  # None means all are allowed

    async def _get_allowed_resources(self, org_id: str) -> Optional[List[str]]:
        """Get list of allowed resource types for an organization."""
        rules = await self._get_org_safety_rules(org_id)
        if rules and "allowed_resources" in rules:
            return rules["allowed_resources"]
        return None  # None means all are allowed

    def _get_required_parameters(self, action_type: str) -> List[str]:
        """Get required parameters for an action type."""
        # Define required parameters for common action types
        required_params_map = {
            "send_message": ["channel", "message"],
            "create_task": ["title", "description"],
            "update_task": ["task_id", "updates"],
            "send_email": ["to", "subject", "body"],
            "create_campaign": ["name", "type"],
            "post_social": ["platform", "content"],
        }

        return required_params_map.get(action_type.lower(), [])

    def _compare_risk_levels(self, level1: RiskLevel, level2: RiskLevel) -> int:
        """
        Compare two risk levels.

        Returns:
            -1 if level1 < level2, 0 if equal, 1 if level1 > level2
        """
        risk_values = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3
        }

        val1 = risk_values[level1]
        val2 = risk_values[level2]

        if val1 < val2:
            return -1
        elif val1 > val2:
            return 1
        else:
            return 0


# Singleton instance
_safety_service = None


def get_safety_service(supabase: Client) -> SafetyService:
    """Get or create SafetyService instance."""
    global _safety_service
    if _safety_service is None:
        _safety_service = SafetyService(supabase)
    return _safety_service