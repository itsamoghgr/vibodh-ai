"""
Communication Agent - Phase 4
Autonomous agent for simple communication tasks (messages, notifications)
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from app.agents.base_agent import (
    BaseAgent, AgentCapability, ActionStep, ActionPlan,
    ObservationContext, ExecutionResult, ReflectionInsight
)
from app.services.slack_service import SlackService
from app.core.config import settings
from app.core.logging import logger
import re


class CommunicationAgent(BaseAgent):
    """
    Communication Agent implementation for simple messaging tasks.

    Capabilities:
    - Send Slack messages
    - Send notifications
    - Simple announcements

    Designed for single-step, low-risk communication actions
    that auto-execute without approval.
    """

    def _initialize_agent(self) -> None:
        """Initialize communication-specific configuration."""
        self.slack_service = SlackService(
            settings.SLACK_CLIENT_ID,
            settings.SLACK_CLIENT_SECRET
        )
        logger.info(f"[COMMUNICATION_AGENT] Initialized")

    @property
    def capabilities(self) -> List[AgentCapability]:
        """Return list of capabilities this agent supports."""
        return [
            AgentCapability.MESSAGE_SENDING,
            AgentCapability.CONTENT_GENERATION
        ]

    @property
    def required_permissions(self) -> List[str]:
        """Return list of permissions required for this agent to function."""
        return [
            "slack:write",
            "slack:read"
        ]

    @property
    def supported_integrations(self) -> List[str]:
        """Return list of integrations this agent can work with."""
        return ["slack", "email"]

    @property
    def description(self) -> str:
        """Description of what this agent does."""
        return "Simple communication agent for sending messages and notifications without campaign overhead"

    async def _observe_impl(self, context: ObservationContext) -> Tuple[bool, Optional[str]]:
        """
        Communication-specific observation implementation.

        Detects simple message sending requests:
        - "send a message to X"
        - "notify the team about Y"
        - "tell #channel that Z"

        Args:
            context: Observation context with query and metadata

        Returns:
            Tuple of (should_act, reason)
        """
        logger.info(f"[COMMUNICATION_AGENT] Observing for communication opportunities")

        if context.query:
            query_lower = context.query.lower()

            # Simple communication triggers
            communication_triggers = [
                "send a message", "send message", "notify", "tell",
                "announce", "post to", "send to", "message",
                "let them know", "inform", "alert", "ping"
            ]

            # Check for simple message sending
            # Trust the routing layer to have sent us here correctly
            # We handle ALL message sending, regardless of content
            if any(trigger in query_lower for trigger in communication_triggers):
                return True, "Simple message sending requested"

        logger.info(f"[COMMUNICATION_AGENT] No simple communication action needed")
        return False, None

    async def _plan_impl(self, goal: str, context: Dict[str, Any]) -> ActionPlan:
        """
        Generate a simple communication action plan.

        Creates single-step plan for message sending.

        Args:
            goal: Communication goal
            context: Planning context

        Returns:
            Simple single-step action plan
        """
        logger.info(f"[COMMUNICATION_AGENT] Planning for goal: {goal}")

        # Parse the goal to extract channel and message
        channel, message = self._parse_message_request(goal)

        # Create single action step
        step = ActionStep(
            step_index=0,
            action_type="send_message",
            action_name="Send Message",
            description=f"Send message to {channel}",
            target_integration="slack",
            target_resource={"channel": channel},
            parameters={
                "channel": channel,
                "message": message,
                "goal": goal
            },
            risk_level="low",
            requires_approval=False,  # Auto-execute
            depends_on=[],
            estimated_duration_ms=2000  # 2 seconds
        )

        # Create simple action plan
        plan = ActionPlan(
            goal=goal,
            description=f"Send message to {channel}",
            steps=[step],
            total_steps=1,
            risk_level="low",
            requires_approval=False,  # Auto-execute simple messages
            context={
                "channel": channel,
                "message_preview": message[:100] if message else goal[:100]
            },
            confidence_score=0.95,
            estimated_total_duration_ms=2000
        )

        logger.info(f"[COMMUNICATION_AGENT] Created simple 1-step plan")
        return plan

    def _parse_message_request(self, goal: str) -> Tuple[str, str]:
        """
        Parse the goal to extract channel and message.

        Args:
            goal: User's request

        Returns:
            Tuple of (channel, message)
        """
        # Try to find channel mention with various patterns
        # Pattern 1: "to [the] #channel-name [channel] that"
        # Pattern 2: "in [the] #channel-name [channel]"
        channel_patterns = [
            r'(?:to|in)\s+(?:the\s+)?#([a-zA-Z0-9-_]+)',  # "to #channel" or "to the #channel"
            r'(?:to|in)\s+(?:the\s+)?([a-zA-Z0-9-_]+)\s+channel',  # "to the private-ch channel"
            r'channel\s+#([a-zA-Z0-9-_]+)',  # "channel #name"
            r'channel\s+([a-zA-Z0-9-_]+)',  # "channel name"
        ]

        channel = None
        for pattern in channel_patterns:
            channel_match = re.search(pattern, goal, re.IGNORECASE)
            if channel_match:
                channel = channel_match.group(1)
                break

        # Default to general if no channel found
        if not channel:
            channel = "general"

        # Ensure channel has # prefix
        if not channel.startswith('#'):
            channel = f"#{channel}"

        # Try to extract the message content
        # Look for "that" followed by the message
        message_patterns = [
            r'(?:channel|#[a-zA-Z0-9-_]+)\s+that\s+(.+)$',  # "channel that MESSAGE"
            r'(?:that|:)\s+(.+)$',  # "that MESSAGE" or ": MESSAGE"
            r'"(.+)"',  # Quoted message
        ]

        message = None
        for pattern in message_patterns:
            message_match = re.search(pattern, goal, re.IGNORECASE)
            if message_match:
                message = message_match.group(1).strip()
                break

        # If no message found, use the whole goal
        if not message:
            message = goal

        return channel, message

    async def _execute_impl(self, action: ActionStep) -> ExecutionResult:
        """
        Execute a communication action step.

        Sends actual Slack messages using SlackService.

        Args:
            action: Action to execute

        Returns:
            Execution result
        """
        logger.info(f"[COMMUNICATION_AGENT] Executing: {action.action_name}")

        try:
            if action.action_type == "send_message":
                # Send actual Slack message
                result_data = await self._send_slack_message(action)

                return ExecutionResult(
                    success=True,
                    action_id=f"comm_{action.step_index}",
                    result=result_data,
                    execution_time_ms=result_data.get("execution_time_ms", 1000),
                    side_effects=[{
                        "type": "message_sent",
                        "description": f"Sent message to {action.parameters.get('channel')}",
                        "channel": action.parameters.get('channel'),
                        "timestamp": result_data.get("timestamp")
                    }]
                )
            else:
                # Unknown action type
                raise ValueError(f"Unknown action type: {action.action_type}")

        except Exception as e:
            logger.error(f"[COMMUNICATION_AGENT] Execution failed: {e}")
            return ExecutionResult(
                success=False,
                action_id=f"comm_{action.step_index}",
                error_message=str(e),
                execution_time_ms=1000
            )

    async def _send_slack_message(self, action: ActionStep) -> Dict[str, Any]:
        """
        Send actual Slack message using SlackService.

        Args:
            action: Action step with message parameters

        Returns:
            Result with Slack response
        """
        import time
        start_time = time.time()

        channel = action.parameters.get("channel", "#general")
        message = action.parameters.get("message", "")

        # Remove # prefix for API call
        channel_id = channel.lstrip('#')

        logger.info(f"[COMMUNICATION_AGENT] Sending message to {channel}")

        try:
            # Get Slack access token from database
            from app.db import supabase

            creds = supabase.table("connections")\
                .select("access_token")\
                .eq("org_id", self.org_id)\
                .eq("source_type", "slack")\
                .single()\
                .execute()

            if not creds.data:
                raise Exception("Slack not connected for this organization")

            access_token = creds.data["access_token"]

            # Send message via Slack API
            slack_response = self.slack_service.post_message(
                access_token=access_token,
                channel=channel_id,
                text=message
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            logger.info(f"[COMMUNICATION_AGENT] Message sent successfully to {channel}")

            return {
                "status": "sent",
                "channel": channel,
                "message": message,
                "timestamp": slack_response.get("ts"),
                "slack_channel_id": slack_response.get("channel"),
                "execution_time_ms": execution_time_ms
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[COMMUNICATION_AGENT] Failed to send Slack message: {error_msg}")

            # Provide helpful error messages
            if "missing_scope" in error_msg.lower():
                error_msg = "Missing Slack permission. Please add 'chat:write' scope and reinstall the app."
            elif "channel_not_found" in error_msg.lower():
                error_msg = f"Channel '{channel}' not found. Please check the channel name."
            elif "not_in_channel" in error_msg.lower():
                error_msg = f"Bot is not in channel '{channel}'. Please invite the bot to the channel."

            raise Exception(error_msg)

    async def _reflect_impl(self, result: ExecutionResult, context: Dict[str, Any]) -> ReflectionInsight:
        """
        Reflect on communication action execution.

        Args:
            result: Execution result
            context: Execution context

        Returns:
            Reflection insights
        """
        if result.success:
            insight = "Message sent successfully. Communication clear and direct."
            confidence = 0.9
        else:
            insight = f"Message failed: {result.error_message}. May need to check permissions or channel access."
            confidence = 0.7

        return ReflectionInsight(
            observation=f"Communication action {'succeeded' if result.success else 'failed'}",
            learning=insight,
            confidence_score=confidence,
            should_adjust_strategy=not result.success,
            suggested_improvements=[
                "Verify Slack permissions include chat:write scope",
                "Ensure bot is invited to target channels",
                "Validate channel names before sending"
            ] if not result.success else []
        )
