"""
Agent Registry Service - Phase 4, Step 1
Manages agent registration, discovery, and routing
"""

from typing import Dict, List, Optional, Any, Type
from uuid import UUID
from datetime import datetime
from supabase import Client

from app.agents.base_agent import BaseAgent, AgentCapability
from app.models.agent import (
    AgentConfig, AgentRegistryEntry, AgentStatus,
    AgentStatusResponse, AgentStatistics
)
from app.core.logging import logger


class AgentRegistry:
    """
    Central registry for managing decision agents.

    Responsibilities:
    - Register and manage agent types
    - Route goals to appropriate agents
    - Track agent availability and capabilities
    - Provide agent discovery
    """

    def __init__(self, supabase: Client):
        """
        Initialize agent registry.

        Args:
            supabase: Supabase client
        """
        self.supabase = supabase
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}
        self._agent_instances: Dict[str, BaseAgent] = {}

        logger.info("[AGENT_REGISTRY] Initialized")

    def register_agent_class(self, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register an agent class for instantiation.

        Args:
            agent_type: Unique agent type identifier
            agent_class: Agent class that extends BaseAgent
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class must extend BaseAgent")

        self._agent_classes[agent_type] = agent_class
        logger.info(f"[AGENT_REGISTRY] Registered agent class: {agent_type}")

    async def register_agent(self, org_id: str, config: AgentConfig) -> AgentRegistryEntry:
        """
        Register a new agent in the database.

        Args:
            org_id: Organization ID
            config: Agent configuration

        Returns:
            Created registry entry
        """
        try:
            # Check if agent type exists
            existing = self.supabase.table("ai_agent_registry")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("agent_type", config.agent_type)\
                .execute()

            if existing.data:
                # Update existing
                result = self.supabase.table("ai_agent_registry")\
                    .update({
                        "agent_name": config.agent_name,
                        "description": config.description,
                        "status": config.status,
                        "capabilities": config.capabilities,
                        "required_permissions": config.required_permissions,
                        "supported_integrations": config.supported_integrations,
                        "config": config.config,
                        "default_risk_threshold": config.default_risk_threshold,
                        "version": config.version,
                        "updated_at": datetime.utcnow().isoformat()
                    })\
                    .eq("id", existing.data[0]["id"])\
                    .execute()

                logger.info(f"[AGENT_REGISTRY] Updated agent: {config.agent_type} for org {org_id}")
            else:
                # Create new
                result = self.supabase.table("ai_agent_registry")\
                    .insert({
                        "org_id": org_id,
                        "agent_type": config.agent_type,
                        "agent_name": config.agent_name,
                        "description": config.description,
                        "status": config.status,
                        "capabilities": config.capabilities,
                        "required_permissions": config.required_permissions,
                        "supported_integrations": config.supported_integrations,
                        "config": config.config,
                        "default_risk_threshold": config.default_risk_threshold,
                        "version": config.version
                    })\
                    .execute()

                logger.info(f"[AGENT_REGISTRY] Registered new agent: {config.agent_type} for org {org_id}")

            return AgentRegistryEntry(**result.data[0])

        except Exception as e:
            logger.error(f"[AGENT_REGISTRY] Failed to register agent: {e}")
            raise

    def get_agent(self, org_id: str, agent_type: str) -> BaseAgent:
        """
        Get or create an agent instance.

        Args:
            org_id: Organization ID
            agent_type: Agent type

        Returns:
            Agent instance

        Raises:
            ValueError: If agent type not registered
        """
        # Check if agent class is registered
        if agent_type not in self._agent_classes:
            raise ValueError(f"Agent type '{agent_type}' not registered")

        # Check for existing instance
        instance_key = f"{org_id}_{agent_type}"
        if instance_key not in self._agent_instances:
            # Create new instance
            agent_class = self._agent_classes[agent_type]
            self._agent_instances[instance_key] = agent_class(org_id, agent_type)
            logger.info(f"[AGENT_REGISTRY] Created agent instance: {agent_type} for org {org_id}")

        return self._agent_instances[instance_key]

    async def list_available_agents(self, org_id: str, status: Optional[AgentStatus] = None) -> List[AgentRegistryEntry]:
        """
        List all available agents for an organization.

        Args:
            org_id: Organization ID
            status: Optional status filter

        Returns:
            List of agent registry entries
        """
        try:
            query = self.supabase.table("ai_agent_registry")\
                .select("*")\
                .eq("org_id", org_id)

            if status:
                query = query.eq("status", status.value)

            result = query.execute()

            return [AgentRegistryEntry(**entry) for entry in result.data]

        except Exception as e:
            logger.error(f"[AGENT_REGISTRY] Failed to list agents: {e}")
            raise

    async def route_to_agent(
        self,
        org_id: str,
        goal: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Determine which agent should handle a goal.

        Args:
            org_id: Organization ID
            goal: Goal to achieve
            context: Additional context

        Returns:
            Agent type that should handle the goal, or None
        """
        try:
            # Get active agents
            agents = await self.list_available_agents(org_id, AgentStatus.ACTIVE)

            if not agents:
                logger.warning(f"[AGENT_REGISTRY] No active agents for org {org_id}")
                return None

            # Simple routing based on keywords - can be enhanced with ML
            goal_lower = goal.lower()

            # PRIORITY 1: Communication agent - simple messaging (VERB-based intent)
            # Strong communication phrases that indicate simple message sending
            strong_communication_triggers = [
                "send a message", "send message", "send a slack message",
                "send slack message", "post a message", "notify", "send to"
            ]

            # Weaker communication triggers
            simple_communication_triggers = [
                "tell", "post to", "message", "let them know", "inform",
                "announce to", "alert", "ping"
            ]

            # Check if query STARTS with a STRONG communication verb
            # This takes priority over campaign content mentions
            query_start = goal_lower[:100]
            has_strong_communication_verb = any(
                query_start.startswith(trigger) or f" {trigger}" in query_start[:50]
                for trigger in strong_communication_triggers
            )

            # If strong communication verb at start, route to CommunicationAgent immediately
            if has_strong_communication_verb:
                if any(a.agent_type == "communication_agent" for a in agents):
                    logger.info(f"[AGENT_REGISTRY] Routing to CommunicationAgent (explicit message sending)")
                    return "communication_agent"

            # Check for weaker communication triggers
            has_communication_verb = any(
                trigger in query_start
                for trigger in simple_communication_triggers
            )

            # Check for campaign CREATION verbs NEAR campaign nouns (within 10 chars)
            # This prevents "message about starting the campaign" from being detected as creation
            creation_patterns = [
                "create a campaign", "create campaign", "launch a campaign", "launch campaign",
                "build a campaign", "build campaign", "start a campaign", "start campaign",
                "set up a campaign", "set up campaign", "run a campaign", "run campaign",
                "design a campaign", "design campaign", "new campaign"
            ]

            is_creating_campaign = any(
                pattern in goal_lower
                for pattern in creation_patterns
            )

            # Route to CommunicationAgent if:
            # 1. Has communication verb (tell, inform, etc.)
            # 2. NOT explicitly creating a campaign
            if has_communication_verb and not is_creating_campaign:
                if any(a.agent_type == "communication_agent" for a in agents):
                    logger.info(f"[AGENT_REGISTRY] Routing to CommunicationAgent (verb-based intent)")
                    return "communication_agent"

            # PRIORITY 2: Marketing agent keywords (complex campaigns)
            marketing_keywords = [
                "campaign", "marketing", "content", "social", "email",
                "audience", "engagement", "promotion", "brand", "advertisement"
            ]
            if any(keyword in goal_lower for keyword in marketing_keywords):
                if any(a.agent_type == "marketing_agent" for a in agents):
                    logger.info(f"[AGENT_REGISTRY] Routing to MarketingAgent (campaign)")
                    return "marketing_agent"

            # Operations agent keywords
            ops_keywords = [
                "task", "workflow", "process", "automate", "operation",
                "efficiency", "optimize", "schedule", "assign", "resource"
            ]
            if any(keyword in goal_lower for keyword in ops_keywords):
                if any(a.agent_type == "ops_agent" for a in agents):
                    return "ops_agent"

            # HR agent keywords (future)
            hr_keywords = [
                "hire", "recruit", "employee", "onboard", "performance",
                "training", "team", "culture", "compensation", "benefits"
            ]
            if any(keyword in goal_lower for keyword in hr_keywords):
                if any(a.agent_type == "hr_agent" for a in agents):
                    return "hr_agent"

            # Finance agent keywords (future)
            finance_keywords = [
                "budget", "finance", "expense", "revenue", "invoice",
                "payment", "forecast", "financial", "accounting", "profit"
            ]
            if any(keyword in goal_lower for keyword in finance_keywords):
                if any(a.agent_type == "finance_agent" for a in agents):
                    return "finance_agent"

            # If no specific match, return first available agent (for testing)
            if agents:
                logger.info(f"[AGENT_REGISTRY] No specific match, using default: {agents[0].agent_type}")
                return agents[0].agent_type

            return None

        except Exception as e:
            logger.error(f"[AGENT_REGISTRY] Failed to route goal: {e}")
            return None

    async def get_agent_status(self, org_id: str, agent_type: str) -> AgentStatusResponse:
        """
        Get current status of an agent.

        Args:
            org_id: Organization ID
            agent_type: Agent type

        Returns:
            Agent status
        """
        try:
            # Get registry entry
            registry = self.supabase.table("ai_agent_registry")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("agent_type", agent_type)\
                .single()\
                .execute()

            if not registry.data:
                raise ValueError(f"Agent {agent_type} not found")

            # Get agent instance status if available
            instance_key = f"{org_id}_{agent_type}"
            if instance_key in self._agent_instances:
                agent = self._agent_instances[instance_key]
                instance_status = agent.get_status()
            else:
                instance_status = {
                    "state": "idle",
                    "has_current_plan": False,
                    "execution_history_count": 0
                }

            # Count active plans
            active_plans = self.supabase.table("ai_action_plans")\
                .select("id", count="exact")\
                .eq("org_id", org_id)\
                .eq("agent_type", agent_type)\
                .in_("status", ["pending", "approved", "executing"])\
                .execute()

            # Count pending actions
            pending_actions = self.supabase.table("ai_actions_pending")\
                .select("id", count="exact")\
                .eq("org_id", org_id)\
                .eq("status", "pending")\
                .execute()

            return AgentStatusResponse(
                agent_type=agent_type,
                agent_name=registry.data["agent_name"],
                status=AgentStatus(registry.data["status"]),
                state=instance_status["state"],
                has_current_plan=instance_status["has_current_plan"],
                execution_history_count=instance_status["execution_history_count"],
                capabilities=registry.data["capabilities"],
                supported_integrations=registry.data["supported_integrations"],
                active_plans_count=active_plans.count if active_plans.count else 0,
                pending_actions_count=pending_actions.count if pending_actions.count else 0
            )

        except Exception as e:
            logger.error(f"[AGENT_REGISTRY] Failed to get agent status: {e}")
            raise

    async def get_agent_statistics(self, org_id: str, agent_type: str) -> AgentStatistics:
        """
        Get performance statistics for an agent.

        Args:
            org_id: Organization ID
            agent_type: Agent type

        Returns:
            Agent statistics
        """
        try:
            # Get all plans for this agent
            plans = self.supabase.table("ai_action_plans")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("agent_type", agent_type)\
                .execute()

            total_plans = len(plans.data)
            completed_plans = sum(1 for p in plans.data if p["status"] == "completed")
            failed_plans = sum(1 for p in plans.data if p["status"] == "failed")
            success_rate = completed_plans / total_plans if total_plans > 0 else 0

            # Calculate execution times
            execution_times = [
                p["execution_time_ms"] for p in plans.data
                if p["execution_time_ms"] is not None
            ]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

            # Get executed actions
            actions = self.supabase.table("ai_actions_executed")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("agent_type", agent_type)\
                .execute()

            total_actions = len(actions.data)

            # Count risk levels
            high_risk_actions = sum(
                1 for p in plans.data
                if p["risk_level"] in ["high", "critical"]
            )

            # Count approvals
            approvals_required = sum(1 for p in plans.data if p["requires_approval"])
            approvals_granted = sum(
                1 for p in plans.data
                if p["approval_status"] == "approved"
            )
            approvals_rejected = sum(
                1 for p in plans.data
                if p["approval_status"] == "rejected"
            )

            # Get last execution
            last_plan = max(
                (p for p in plans.data if p["completed_at"]),
                key=lambda x: x["completed_at"],
                default=None
            )
            last_execution = datetime.fromisoformat(last_plan["completed_at"]) if last_plan else None

            return AgentStatistics(
                agent_type=agent_type,
                total_plans=total_plans,
                completed_plans=completed_plans,
                failed_plans=failed_plans,
                success_rate=success_rate,
                average_execution_time_ms=int(avg_execution_time),
                total_actions_executed=total_actions,
                high_risk_actions=high_risk_actions,
                approvals_required=approvals_required,
                approvals_granted=approvals_granted,
                approvals_rejected=approvals_rejected,
                last_execution=last_execution
            )

        except Exception as e:
            logger.error(f"[AGENT_REGISTRY] Failed to get agent statistics: {e}")
            raise

    async def deactivate_agent(self, org_id: str, agent_type: str) -> None:
        """
        Deactivate an agent.

        Args:
            org_id: Organization ID
            agent_type: Agent type
        """
        try:
            result = self.supabase.table("ai_agent_registry")\
                .update({"status": AgentStatus.INACTIVE.value})\
                .eq("org_id", org_id)\
                .eq("agent_type", agent_type)\
                .execute()

            if result.data:
                logger.info(f"[AGENT_REGISTRY] Deactivated agent: {agent_type} for org {org_id}")

                # Remove instance if exists
                instance_key = f"{org_id}_{agent_type}"
                if instance_key in self._agent_instances:
                    del self._agent_instances[instance_key]

        except Exception as e:
            logger.error(f"[AGENT_REGISTRY] Failed to deactivate agent: {e}")
            raise

    async def activate_agent(self, org_id: str, agent_type: str) -> None:
        """
        Activate an agent.

        Args:
            org_id: Organization ID
            agent_type: Agent type
        """
        try:
            result = self.supabase.table("ai_agent_registry")\
                .update({"status": AgentStatus.ACTIVE.value})\
                .eq("org_id", org_id)\
                .eq("agent_type", agent_type)\
                .execute()

            if result.data:
                logger.info(f"[AGENT_REGISTRY] Activated agent: {agent_type} for org {org_id}")

        except Exception as e:
            logger.error(f"[AGENT_REGISTRY] Failed to activate agent: {e}")
            raise


# Singleton instance
_agent_registry = None


def get_agent_registry(supabase: Client) -> AgentRegistry:
    """Get or create AgentRegistry instance."""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry(supabase)
    return _agent_registry