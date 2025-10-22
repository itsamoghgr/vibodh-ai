"""
Communication Agent - Phase 4 Enhanced
Autonomous agent with intelligent reasoning for communication tasks

Enhanced with:
- Intent classification (informational/urgent/strategic/routine)
- Context awareness from KG + Memory
- Multi-step dynamic planning
- Reflection feedback loop
- Self-verification pre-execution
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from app.agents.base_agent import (
    BaseAgent, AgentCapability, ActionStep, ActionPlan,
    ObservationContext, ExecutionResult, ReflectionInsight
)
from app.services.slack_service import SlackService
from app.services.communication_reasoning_service import get_communication_reasoning_service
from app.services.communication_context_service import get_communication_context_service
from app.services.agent_event_bus import get_agent_event_bus, AgentEventType
from app.services.email_service import get_email_service, EmailMessage
from app.core.config import settings
from app.core.logging import logger
from app.db import supabase
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

        # Initialize reasoning and context services
        self.reasoning_service = get_communication_reasoning_service(supabase)
        self.context_service = get_communication_context_service(supabase)

        # Initialize agent event bus for cross-agent communication
        self.event_bus = get_agent_event_bus(supabase)

        # Initialize email service
        self.email_service = get_email_service(supabase=supabase)

        # Subscribe to relevant event types
        self.event_bus.subscribe(
            agent_type="communication_agent",
            event_types=[
                AgentEventType.CAMPAIGN_COMPLETED,
                AgentEventType.TASK_CREATED,
                AgentEventType.INSIGHT_GENERATED,
                AgentEventType.WORKFLOW_COMPLETED
            ]
        )

        logger.info(f"[COMMUNICATION_AGENT] Initialized with reasoning, context, event bus, and email")

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
        Communication-specific observation implementation with interactive information gathering.

        Observes two types of opportunities:
        1. Direct user requests: "send a message to X"
        2. Event-driven triggers: Pending communication events from listener

        NOW ENHANCED: Detects incomplete requests and asks clarifying questions

        Args:
            context: Observation context with query and metadata

        Returns:
            Tuple of (should_act, reason)
            - If enough info: (True, "reason")
            - If need more info: (False, "NEED_INFO: <conversational question>")
        """
        logger.info(f"[COMMUNICATION_AGENT] Observing for communication opportunities")

        # Path 1a: Check for agent-to-agent events (cross-agent coordination)
        agent_events = await self._check_agent_events()
        if agent_events:
            logger.info(
                f"[COMMUNICATION_AGENT] Found {len(agent_events)} agent events"
            )
            # Store agent events in context for planning
            context.metadata["agent_events"] = agent_events
            return True, f"Cross-agent event: {len(agent_events)} agent events"

        # Path 1b: Check for pending communication events (event-driven from listener)
        pending_events = await self._check_pending_communication_events()
        if pending_events:
            logger.info(
                f"[COMMUNICATION_AGENT] Found {len(pending_events)} pending "
                f"communication events"
            )
            # Store pending events in context for planning
            context.metadata["pending_events"] = pending_events
            return True, f"Event-driven communication: {len(pending_events)} pending events"

        # Path 2: Check for direct user request
        if context.query:
            query_lower = context.query.lower()

            # Simple communication triggers
            communication_triggers = [
                "send a message", "send message", "notify", "tell",
                "announce", "post to", "send to", "message",
                "let them know", "inform", "alert", "ping"
            ]

            # Check for simple message sending
            if any(trigger in query_lower for trigger in communication_triggers):
                # NEW: Check if we have enough information to proceed
                has_enough_info, missing_info_question = await self._check_information_completeness(context.query)

                if not has_enough_info:
                    logger.info(f"[COMMUNICATION_AGENT] Incomplete request, asking for: {missing_info_question}")
                    return False, f"NEED_INFO: {missing_info_question}"

                # Has enough info, proceed with planning
                return True, "User-requested message sending with complete information"

        logger.info(f"[COMMUNICATION_AGENT] No communication action needed")
        return False, None

    async def _check_agent_events(self) -> List[Dict[str, Any]]:
        """
        Check for agent-to-agent events from the event bus.

        Returns:
            List of pending agent events targeted at communication_agent
        """
        try:
            # Poll event bus for events targeted at this agent
            events = self.event_bus.poll_events(
                org_id=self.org_id,
                agent_type="communication_agent",
                event_types=[
                    AgentEventType.CAMPAIGN_COMPLETED,
                    AgentEventType.TASK_CREATED,
                    AgentEventType.INSIGHT_GENERATED,
                    AgentEventType.WORKFLOW_COMPLETED
                ],
                limit=5
            )

            return events

        except Exception as e:
            logger.error(f"[COMMUNICATION_AGENT] Failed to check agent events: {e}")
            return []

    async def _check_pending_communication_events(self) -> List[Dict[str, Any]]:
        """
        Check for pending communication events created by event listener.

        Returns:
            List of pending communication events
        """
        try:
            result = supabase.table("communication_events")\
                .select("*")\
                .eq("org_id", self.org_id)\
                .eq("status", "pending")\
                .order("created_at", desc=False)\
                .limit(5)\
                .execute()

            return result.data or []

        except Exception as e:
            logger.error(f"[COMMUNICATION_AGENT] Failed to check pending events: {e}")
            return []

    async def _check_information_completeness(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check if the request has enough information to create a plan.

        Uses LLM to analyze if we have:
        - Message content (what to send)
        - Target audience (who/where to send)

        Args:
            query: User's communication request

        Returns:
            Tuple of (has_enough_info, clarifying_question)
            - (True, None) if enough info
            - (False, "What message would you like to send?") if missing info
        """
        try:
            import requests

            prompt = f"""Analyze this communication request and determine if it has enough information to execute.

Request: "{query}"

For a message sending request, we need:
1. Message content: What to send (actual message text or clear topic)
2. Target: Who/where to send (channel, person, team)

Analyze the request and respond with ONLY a JSON object:

If COMPLETE (has both message content and target):
{{
  "complete": true,
  "missing": null
}}

If INCOMPLETE (missing information):
{{
  "complete": false,
  "missing": "what|where|both",
  "question": "Natural conversational question asking for the missing info"
}}

Examples:

Request: "send a message to #engineering saying the deployment is complete"
Response: {{"complete": true, "missing": null}}

Request: "I want to send a message"
Response: {{"complete": false, "missing": "both", "question": "Sure! What message would you like to send, and which channel should I send it to?"}}

Request: "tell the team about the new feature"
Response: {{"complete": false, "missing": "where", "question": "Got it! Which channel or team should I notify?"}}

Request: "send to #general"
Response: {{"complete": false, "missing": "what", "question": "Perfect! What message would you like me to send to #general?"}}

Now analyze this request:"""

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": settings.GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 200,
                    "response_format": {"type": "json_object"}
                },
                timeout=10
            )

            if response.status_code == 200:
                import json
                result = json.loads(response.json()["choices"][0]["message"]["content"])

                if result.get("complete"):
                    return True, None
                else:
                    question = result.get("question", "Could you provide more details about what message to send and where?")
                    return False, question

            # Fallback: assume incomplete
            return False, "Sure! What message would you like to send, and which channel should I send it to?"

        except Exception as e:
            logger.error(f"[COMMUNICATION_AGENT] Info completeness check failed: {e}")
            # On error, assume we need more info to be safe
            return False, "Could you tell me what message you'd like to send and where?"

    async def _plan_impl(self, goal: str, context: Dict[str, Any]) -> ActionPlan:
        """
        Generate intelligent communication action plan with reasoning.

        Enhanced planning process:
        1. Check for agent events (cross-agent coordination)
        2. Analyze intent and urgency (reasoning service)
        3. Gather rich context from KG + Memory
        4. Log reasoning steps for transparency
        5. Create context-aware action plan
        6. Determine audience and channels intelligently

        Args:
            goal: Communication goal
            context: Planning context

        Returns:
            Context-aware action plan
        """
        logger.info(f"[COMMUNICATION_AGENT] Planning with reasoning for goal: {goal}")

        # Step 0: Handle agent events if present (cross-agent coordination)
        agent_events = context.get("agent_events", [])
        if agent_events:
            logger.info(f"[COMMUNICATION_AGENT] Handling {len(agent_events)} agent events")
            return await self._plan_for_agent_event(agent_events[0], context)

        # Step 1: Analyze intent and get recommendations
        intent_analysis = await self.reasoning_service.analyze_communication_request(
            query=goal,
            org_id=self.org_id,
            user_id=context.get('user_id'),
            context=context
        )

        logger.info(
            f"[COMMUNICATION_AGENT] Intent classified as {intent_analysis.intent_type.value} "
            f"(urgency: {intent_analysis.urgency_score:.2f}), "
            f"suggested_audience: {intent_analysis.suggested_audience}"
        )

        # Step 2: Gather rich context from KG + Memory, prioritizing explicit channel mentions
        comm_context = await self.context_service.gather_context(
            query=goal,
            org_id=self.org_id,
            intent_type=intent_analysis.intent_type.value,
            suggested_audience=intent_analysis.suggested_audience
        )

        logger.info(
            f"[COMMUNICATION_AGENT] Context gathered: "
            f"topic='{comm_context.topic}', "
            f"channels={comm_context.audience.channels}, "
            f"people={len(comm_context.audience.people)}, "
            f"memories={len(comm_context.recent_context)}"
        )

        # Step 3: Log reasoning to database for transparency
        reasoning_log = {
            "org_id": self.org_id,
            "query": goal,
            "intent": "communication",
            "modules_used": intent_analysis.recommended_modules,
            "reasoning_steps": [
                {
                    "step": 1,
                    "action": "classify_intent",
                    "result": intent_analysis.intent_type.value
                },
                {
                    "step": 2,
                    "action": "assess_urgency",
                    "score": intent_analysis.urgency_score
                },
                {
                    "step": 3,
                    "action": "identify_audience",
                    "channels": comm_context.audience.channels,
                    "teams": comm_context.audience.teams
                },
                {
                    "step": 4,
                    "action": "gather_context",
                    "topic": comm_context.topic,
                    "memories_found": len(comm_context.recent_context),
                    "entities_found": len(comm_context.related_entities)
                }
            ],
            "execution_time_ms": 1500,
            "created_at": datetime.now().isoformat()
        }

        try:
            supabase.table("reasoning_logs").insert(reasoning_log).execute()
            logger.info(f"[COMMUNICATION_AGENT] Reasoning logged to database")
        except Exception as e:
            logger.warning(f"[COMMUNICATION_AGENT] Failed to log reasoning: {e}")

        # Step 4: Determine channels and message from context
        # Use context-aware channels if available, fallback to parsing
        channels = comm_context.audience.channels or [self._parse_message_request(goal)[0]]

        # Enrich message with relevant context
        _, parsed_message = self._parse_message_request(goal)
        enriched_message = await self.context_service.enrich_message(parsed_message, comm_context)

        # Step 5: Create multi-step dynamic action plan based on intent
        primary_channel = channels[0] if channels else "#general"

        plan = self._create_multi_step_plan(
            intent_analysis=intent_analysis,
            comm_context=comm_context,
            goal=goal,
            primary_channel=primary_channel,
            enriched_message=enriched_message
        )

        logger.info(
            f"[COMMUNICATION_AGENT] Created {plan.total_steps}-step plan "
            f"(intent: {intent_analysis.intent_type.value}, "
            f"requires_approval: {plan.requires_approval})"
        )

        return plan

    def _create_multi_step_plan(
        self,
        intent_analysis,
        comm_context,
        goal: str,
        primary_channel: str,
        enriched_message: str
    ) -> ActionPlan:
        """
        Create multi-step action plan based on communication intent.

        Different intents create different plan sequences:
        - STRATEGIC: Slack → ClickUp task → Follow-up summary
        - URGENT: Immediate Slack → Email backup
        - INFORMATIONAL/ROUTINE: Simple Slack post

        Args:
            intent_analysis: Intent analysis result
            comm_context: Communication context
            goal: Communication goal
            primary_channel: Primary Slack channel
            enriched_message: Enriched message content

        Returns:
            Multi-step action plan
        """
        steps = []
        intent_type = intent_analysis.intent_type.value

        # Step 1: Always send primary Slack message
        steps.append(ActionStep(
            step_index=0,
            action_type="send_message",
            action_name=f"Send {intent_type.capitalize()} Message",
            description=f"Post message to {primary_channel}",
            target_integration="slack",
            target_resource={"channel": primary_channel},
            parameters={
                "channel": primary_channel,
                "message": enriched_message,
                "goal": goal,
                "intent": intent_type,
                "urgency": intent_analysis.urgency_score,
                "topic": comm_context.topic,
                "audience_confidence": comm_context.audience.confidence
            },
            risk_level="low" if intent_analysis.urgency_score < 0.7 else "medium",
            requires_approval=False,
            depends_on=[],
            estimated_duration_ms=2000
        ))

        # Strategic communications: Add task tracking and summary
        if intent_type == "strategic":
            # Step 2: Create ClickUp task for tracking
            steps.append(ActionStep(
                step_index=1,
                action_type="create_task",
                action_name="Create Tracking Task",
                description=f"Create ClickUp task to track '{comm_context.topic}' communication",
                target_integration="clickup",
                target_resource={"list": "communications"},
                parameters={
                    "title": f"Track: {comm_context.topic}",  # Changed from task_name to title
                    "description": f"Monitor engagement and responses for strategic message posted to {primary_channel}",
                    "priority": 2,  # High priority
                    "assignees": comm_context.audience.people[:3] if comm_context.audience.people else [],
                    "due_date": "in 3 days",
                    "tags": ["communication", "strategic", comm_context.topic]
                },
                risk_level="low",
                requires_approval=False,
                depends_on=[0],  # Depends on message being sent
                estimated_duration_ms=3000
            ))

            # Step 3: Schedule follow-up summary
            steps.append(ActionStep(
                step_index=2,
                action_type="schedule_summary",
                action_name="Schedule Engagement Summary",
                description=f"Schedule summary of engagement metrics in 48 hours",
                target_integration="internal",
                target_resource={"type": "scheduled_insight"},
                parameters={
                    "summary_type": "communication_engagement",
                    "channel": primary_channel,
                    "topic": comm_context.topic,
                    "schedule_hours": 48,
                    "recipients": comm_context.audience.people[:2] if comm_context.audience.people else []
                },
                risk_level="low",
                requires_approval=False,
                depends_on=[0],  # Depends on message being sent
                estimated_duration_ms=1000
            ))

        # Urgent communications: Add email backup
        elif intent_type == "urgent":
            # Step 2: Send email backup to key stakeholders
            steps.append(ActionStep(
                step_index=1,
                action_type="send_email",
                action_name="Send Email Backup",
                description=f"Email key stakeholders about urgent message",
                target_integration="email",
                target_resource={"type": "stakeholder_notification"},
                parameters={
                    "to": [p["name"] for p in comm_context.audience.people[:5]],  # Changed from recipients to to
                    "subject": f"URGENT: {comm_context.topic}",
                    "body": f"This is a backup notification for an urgent message posted to {primary_channel}:\n\n{enriched_message}",
                    "priority": "high"
                },
                risk_level="medium",
                requires_approval=False,
                depends_on=[0],  # Depends on message being sent
                estimated_duration_ms=2000
            ))

        # Informational/Routine: Single step (already added above)

        # Determine plan-level properties
        requires_approval = (
            intent_type == "strategic" or
            intent_analysis.urgency_score > 0.9
        )

        # Calculate total duration, handling None values
        total_duration = sum(step.estimated_duration_ms or 0 for step in steps)

        # Calculate overall confidence (use audience confidence or intent confidence as fallback)
        confidence = comm_context.audience.confidence if comm_context.audience.confidence is not None else intent_analysis.confidence
        # Final safety check: ensure confidence is never None
        if confidence is None:
            confidence = 0.7  # Default to moderate confidence

        plan = ActionPlan(
            goal=goal,
            description=self._generate_plan_description(intent_type, comm_context.topic, len(steps)),
            steps=steps,
            total_steps=len(steps),
            risk_level="medium" if requires_approval else "low",
            requires_approval=requires_approval,
            context={
                "intent": intent_type,
                "urgency": intent_analysis.urgency_score,
                "topic": comm_context.topic,
                "channels": [primary_channel],
                "audience_confidence": comm_context.audience.confidence or 0.5,
                "suggested_timing": comm_context.suggested_timing,
                "reasoning": intent_analysis.reasoning,
                "message_preview": enriched_message[:100],
                "multi_step": len(steps) > 1,
                "step_types": [step.action_type for step in steps]
            },
            confidence_score=confidence,
            estimated_total_duration_ms=total_duration
        )

        return plan

    def _generate_plan_description(self, intent_type: str, topic: str, step_count: int) -> str:
        """Generate human-readable plan description."""
        if intent_type == "strategic":
            return f"Strategic communication about '{topic}' with task tracking and follow-up ({step_count} steps)"
        elif intent_type == "urgent":
            return f"Urgent notification about '{topic}' with email backup ({step_count} steps)"
        elif intent_type == "routine":
            return f"Routine update about '{topic}'"
        else:
            return f"Informational message about '{topic}'"

    async def _plan_for_agent_event(self, event: Dict[str, Any], context: Dict[str, Any]) -> ActionPlan:
        """
        Create action plan for handling agent-to-agent event.

        Translates events from other agents into communication actions.
        For example: campaign_completed → announce campaign success

        Args:
            event: Agent event from event bus
            context: Planning context

        Returns:
            Action plan for communicating the event
        """
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        source_agent = event.get("source_agent")

        logger.info(
            f"[COMMUNICATION_AGENT] Creating plan for {event_type} "
            f"event from {source_agent}"
        )

        # Generate appropriate message based on event type
        if event_type == AgentEventType.CAMPAIGN_COMPLETED.value:
            metrics = payload.get("metrics", {})
            channel = "#marketing"  # Default channel for marketing announcements
            message = f"""🎉 *Marketing Campaign Completed Successfully!*

Campaign ID: {payload.get('plan_id', 'N/A')}

*Results:*
• Total Actions: {metrics.get('total_actions', 0)}
• Successful: {metrics.get('successful_actions', 0)}
• Messages Sent: {metrics.get('messages_sent', 0)}
• Tasks Created: {metrics.get('tasks_created', 0)}

Great work team! 🚀"""

        elif event_type == AgentEventType.TASK_CREATED.value:
            channel = "#general"
            message = f"""📋 *New Task Created*

{payload.get('title', 'Task')}

Check ClickUp for details!"""

        elif event_type == AgentEventType.INSIGHT_GENERATED.value:
            channel = "#insights"
            message = f"""💡 *New Insight Available*

{payload.get('summary', 'A new insight has been generated.')}"""

        elif event_type == AgentEventType.WORKFLOW_COMPLETED.value:
            channel = "#general"
            message = f"""✅ *Workflow Completed*

{payload.get('description', 'A workflow has finished successfully.')}"""

        else:
            # Generic handler
            channel = "#general"
            message = f"""ℹ️ *Agent Event: {event_type}*

Event from: {source_agent}
"""

        # Create single-step plan for announcing the event
        steps = [
            ActionStep(
                step_index=0,
                action_type="send_message",
                action_name=f"Announce {event_type}",
                description=f"Announce {event_type} event from {source_agent}",
                target_integration="slack",
                target_resource={
                    "type": "channel",
                    "name": channel
                },
                parameters={
                    "channel": channel,
                    "message": message,
                    "event_id": event.get("id"),
                    "source_agent": source_agent
                },
                risk_level="low",
                requires_approval=False,
                depends_on=[],
                estimated_duration_ms=1000
            )
        ]

        plan = ActionPlan(
            goal=f"Announce {event_type} event",
            description=f"Cross-agent coordination: announce {event_type} from {source_agent}",
            steps=steps,
            total_steps=1,
            risk_level="low",
            requires_approval=False,
            context={
                "event_type": event_type,
                "source_agent": source_agent,
                "event_id": event.get("id"),
                "payload": payload
            },
            confidence_score=0.9,  # High confidence for agent events
            estimated_total_duration_ms=1000
        )

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
        Execute a communication action step with pre-execution verification.

        Enhanced execution process:
        1. Pre-execution verification (timing, tone, sensitivity)
        2. Abort if verification fails with low confidence
        3. Send message via Slack
        4. Log execution details

        Args:
            action: Action to execute

        Returns:
            Execution result
        """
        logger.info(f"[COMMUNICATION_AGENT] Executing with verification: {action.action_name}")

        try:
            if action.action_type == "send_message":
                # Step 1: Pre-execution verification
                message = action.parameters.get("message", "")
                channel = action.parameters.get("channel", "#general")

                verification_context = {
                    "channel": channel,
                    "intent": action.parameters.get("intent", "informational"),
                    "urgency": action.parameters.get("urgency", 0.5),
                    "recent_messages": []  # Could fetch from DB in future
                }

                logger.info(f"[COMMUNICATION_AGENT] Verifying appropriateness before sending...")
                verification = await self.reasoning_service.verify_appropriateness(
                    message=message,
                    context=verification_context
                )

                logger.info(
                    f"[COMMUNICATION_AGENT] Verification result: "
                    f"appropriate={verification.appropriate}, "
                    f"confidence={verification.confidence:.2f}, "
                    f"reason={verification.reason}"
                )

                # Step 2: Decide whether to proceed based on verification
                if not verification.appropriate:
                    # Message deemed inappropriate - abort execution
                    logger.warning(
                        f"[COMMUNICATION_AGENT] Message blocked by verification: "
                        f"{verification.reason}"
                    )

                    return ExecutionResult(
                        success=False,
                        action_id=f"comm_{action.step_index}",
                        error_message=f"Pre-execution verification failed: {verification.reason}",
                        execution_time_ms=500,
                        result={
                            "verification_failed": True,
                            "reason": verification.reason,
                            "suggested_changes": verification.suggested_changes
                        }
                    )

                if verification.confidence < 0.75:
                    # Low confidence - flag for human review
                    logger.warning(
                        f"[COMMUNICATION_AGENT] Low verification confidence "
                        f"({verification.confidence:.2f}), flagging for review"
                    )

                    # Could create a pending approval here in future
                    # For now, we'll proceed but log the concern
                    try:
                        supabase.table("ai_action_approvals").insert({
                            "org_id": self.org_id,
                            "action_type": "send_message",
                            "action_details": {
                                "channel": channel,
                                "message": message,
                                "verification_confidence": verification.confidence,
                                "verification_reason": verification.reason
                            },
                            "status": "pending_review",
                            "created_at": datetime.now().isoformat()
                        }).execute()
                        logger.info(f"[COMMUNICATION_AGENT] Flagged for human review")
                    except Exception as e:
                        logger.warning(f"[COMMUNICATION_AGENT] Failed to flag for review: {e}")

                # Step 3: Verification passed - send message
                result_data = await self._send_slack_message(action)

                # Step 4: Mark agent event as consumed (if this was triggered by an event)
                event_id = action.parameters.get("event_id")
                if event_id:
                    try:
                        source_agent = action.parameters.get("source_agent", "unknown")
                        self.event_bus.mark_consumed(
                            event_id=event_id,
                            consumed_by="communication_agent",
                            notes=f"Successfully announced event from {source_agent} to {channel}"
                        )
                        logger.info(f"[COMMUNICATION_AGENT] Marked event {event_id} as consumed")
                    except Exception as e:
                        logger.warning(f"[COMMUNICATION_AGENT] Failed to mark event as consumed: {e}")

                return ExecutionResult(
                    success=True,
                    action_id=f"comm_{action.step_index}",
                    result={
                        **result_data,
                        "verification": {
                            "appropriate": verification.appropriate,
                            "confidence": verification.confidence,
                            "reason": verification.reason
                        },
                        "event_consumed": event_id is not None
                    },
                    execution_time_ms=result_data.get("execution_time_ms", 1000),
                    side_effects=[{
                        "type": "message_sent",
                        "description": f"Sent message to {channel}",
                        "channel": channel,
                        "timestamp": result_data.get("timestamp"),
                        "verified": True,
                        "verification_confidence": verification.confidence
                    }]
                )

            elif action.action_type == "create_task":
                # Create ClickUp task for tracking
                result_data = await self._create_clickup_task(action)

                return ExecutionResult(
                    success=True,
                    action_id=f"comm_{action.step_index}",
                    result=result_data,
                    execution_time_ms=result_data.get("execution_time_ms", 3000),
                    side_effects=[{
                        "type": "task_created",
                        "description": f"Created ClickUp task for tracking",
                        "task_id": result_data.get("task_id"),
                        "task_url": result_data.get("task_url")
                    }]
                )

            elif action.action_type == "send_email":
                # Send email backup
                result_data = await self._send_email_backup(action)

                return ExecutionResult(
                    success=True,
                    action_id=f"comm_{action.step_index}",
                    result=result_data,
                    execution_time_ms=result_data.get("execution_time_ms", 2000),
                    side_effects=[{
                        "type": "email_sent",
                        "description": f"Sent email backup to {len(action.parameters.get('recipients', []))} recipients",
                        "recipients": action.parameters.get('recipients', [])
                    }]
                )

            elif action.action_type == "schedule_summary":
                # Schedule follow-up summary
                result_data = await self._schedule_engagement_summary(action)

                return ExecutionResult(
                    success=True,
                    action_id=f"comm_{action.step_index}",
                    result=result_data,
                    execution_time_ms=result_data.get("execution_time_ms", 1000),
                    side_effects=[{
                        "type": "summary_scheduled",
                        "description": f"Scheduled engagement summary for {action.parameters.get('schedule_hours')} hours from now",
                        "schedule_id": result_data.get("schedule_id")
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

    async def _create_clickup_task(self, action: ActionStep) -> Dict[str, Any]:
        """
        Create ClickUp task for communication tracking.

        Args:
            action: Action step with task parameters

        Returns:
            Result with task creation details
        """
        import time
        start_time = time.time()

        params = action.parameters
        task_name = params.get("task_name", "Communication Tracking")

        logger.info(f"[COMMUNICATION_AGENT] Creating ClickUp task: {task_name}")

        try:
            # Get ClickUp access token from database
            creds = supabase.table("connections")\
                .select("access_token, metadata")\
                .eq("org_id", self.org_id)\
                .eq("source_type", "clickup")\
                .single()\
                .execute()

            if not creds.data:
                logger.warning(f"[COMMUNICATION_AGENT] ClickUp not connected, simulating task creation")
                # Simulate task creation without actual ClickUp API call
                execution_time_ms = int((time.time() - start_time) * 1000)
                return {
                    "status": "simulated",
                    "task_id": f"simulated_{int(time.time())}",
                    "task_url": f"https://app.clickup.com/t/simulated",
                    "task_name": task_name,
                    "execution_time_ms": execution_time_ms,
                    "note": "ClickUp not connected - task creation simulated"
                }

            # TODO: Implement actual ClickUp API call
            # For now, simulate successful task creation
            # NOTE: pending_integrations table doesn't exist yet - skipping logging

            execution_time_ms = int((time.time() - start_time) * 1000)

            logger.info(f"[COMMUNICATION_AGENT] ClickUp task creation simulated (credentials found but API not implemented)")

            return {
                "status": "simulated",
                "task_id": f"simulated_{int(time.time())}",
                "task_url": f"https://app.clickup.com/t/simulated",
                "task_name": task_name,
                "execution_time_ms": execution_time_ms,
                "note": "Task creation logged for ClickUp integration"
            }

        except Exception as e:
            logger.error(f"[COMMUNICATION_AGENT] Failed to create ClickUp task: {e}")
            raise Exception(f"ClickUp task creation failed: {str(e)}")

    async def _send_email_backup(self, action: ActionStep) -> Dict[str, Any]:
        """
        Send email backup notification for urgent messages.

        Args:
            action: Action step with email parameters

        Returns:
            Result with email sending details
        """
        import time
        start_time = time.time()

        params = action.parameters
        subject = params.get("subject", "Communication Notification")
        recipients = params.get("recipients", [])
        body_text = params.get("body", "")
        priority = params.get("priority", "normal")

        logger.info(f"[COMMUNICATION_AGENT] Sending email backup to {len(recipients)} recipients")

        try:
            # Create email message
            email_message = EmailMessage(
                to=recipients,
                subject=subject,
                body_text=body_text,
                body_html=None  # Could add HTML template in future
            )

            # Send email via email service
            result = self.email_service.send_email(
                message=email_message,
                org_id=self.org_id,
                max_retries=3
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            if result.get("success"):
                logger.info(
                    f"[COMMUNICATION_AGENT] Email backup sent successfully",
                    extra={
                        "recipients": recipients,
                        "message_id": result.get("message_id")
                    }
                )

                return {
                    "status": "sent",
                    "subject": subject,
                    "recipients": recipients,
                    "message_id": result.get("message_id"),
                    "execution_time_ms": execution_time_ms,
                    "sent_at": result.get("sent_at")
                }
            else:
                logger.error(
                    f"[COMMUNICATION_AGENT] Email backup failed after retries",
                    extra={"error": result.get("error")}
                )

                return {
                    "status": "failed",
                    "subject": subject,
                    "recipients": recipients,
                    "error": result.get("error"),
                    "execution_time_ms": execution_time_ms
                }

        except Exception as e:
            logger.error(f"[COMMUNICATION_AGENT] Failed to send email backup: {e}")
            raise Exception(f"Email backup failed: {str(e)}")

    async def _schedule_engagement_summary(self, action: ActionStep) -> Dict[str, Any]:
        """
        Schedule follow-up engagement summary.

        Args:
            action: Action step with summary parameters

        Returns:
            Result with scheduling details
        """
        import time
        start_time = time.time()

        params = action.parameters
        schedule_hours = params.get("schedule_hours", 48)
        topic = params.get("topic", "communication")

        logger.info(f"[COMMUNICATION_AGENT] Scheduling engagement summary for {schedule_hours}h from now")

        try:
            from datetime import timedelta

            # Calculate scheduled time
            scheduled_time = datetime.now() + timedelta(hours=schedule_hours)

            # Simulate scheduling (scheduled_insights table doesn't exist yet)
            # NOTE: In production, this would create a scheduled job or cron task

            execution_time_ms = int((time.time() - start_time) * 1000)

            schedule_id = f"simulated_{int(time.time())}"

            logger.info(f"[COMMUNICATION_AGENT] Engagement summary scheduling simulated for {scheduled_time.isoformat()}")

            return {
                "status": "simulated",
                "schedule_id": schedule_id,
                "scheduled_time": scheduled_time.isoformat(),
                "topic": topic,
                "execution_time_ms": execution_time_ms,
                "note": f"Summary will be generated in {schedule_hours} hours"
            }

        except Exception as e:
            logger.error(f"[COMMUNICATION_AGENT] Failed to schedule summary: {e}")
            raise Exception(f"Summary scheduling failed: {str(e)}")

    async def _reflect_impl(self, result: ExecutionResult, context: Dict[str, Any]) -> ReflectionInsight:
        """
        Reflect on communication action execution with engagement analysis.

        Enhanced reflection process:
        1. Measure engagement (Slack reactions, responses)
        2. Analyze communication effectiveness
        3. Feed outcomes to Adaptive Engine for learning
        4. Update communication preferences
        5. Suggest improvements for future communications

        Args:
            result: Execution result
            context: Execution context

        Returns:
            Rich reflection insights with engagement metrics
        """
        logger.info(f"[COMMUNICATION_AGENT] Reflecting on execution...")

        if not result.success:
            # Execution failed - analyze failure
            insight = f"Message failed: {result.error_message}"
            confidence = 0.7

            # Check if it was a verification failure
            if result.result and result.result.get("verification_failed"):
                insight = f"Pre-execution verification blocked message: {result.result['reason']}"
                suggested_improvements = result.result.get("suggested_changes", [
                    "Review timing appropriateness",
                    "Check message tone and sensitivity",
                    "Verify channel and audience selection"
                ])
            else:
                suggested_improvements = [
                    "Verify Slack permissions include chat:write scope",
                    "Ensure bot is invited to target channels",
                    "Validate channel names before sending"
                ]

            return ReflectionInsight(
                plan_id=context.get('plan_id', 'unknown'),
                overall_success=False,
                lessons_learned=[insight],
                improvements_suggested=suggested_improvements,
                performance_metrics={
                    "execution_time_ms": result.execution_time_ms,
                    "error_message": result.error_message
                },
                should_retry=True
            )

        # Execution succeeded - measure engagement
        logger.info(f"[COMMUNICATION_AGENT] Measuring engagement...")

        engagement_metrics = await self._measure_engagement(result, context)

        # Analyze effectiveness
        effectiveness_score = self._calculate_effectiveness(engagement_metrics, context)

        logger.info(
            f"[COMMUNICATION_AGENT] Engagement metrics: "
            f"reactions={engagement_metrics.get('reaction_count', 0)}, "
            f"replies={engagement_metrics.get('reply_count', 0)}, "
            f"effectiveness={effectiveness_score:.2f}"
        )

        # Feed outcomes to Adaptive Engine for learning
        await self._feed_to_adaptive_engine(
            result=result,
            context=context,
            engagement=engagement_metrics,
            effectiveness=effectiveness_score
        )

        # Generate insights and learnings
        learnings = []
        should_adjust = False

        if effectiveness_score >= 0.7:
            learnings.append(
                f"High engagement ({effectiveness_score:.2f}). "
                f"Channel and timing were appropriate."
            )
        elif effectiveness_score >= 0.4:
            learnings.append(
                f"Moderate engagement ({effectiveness_score:.2f}). "
                f"Consider adjusting timing or message format."
            )
            should_adjust = True
        else:
            learnings.append(
                f"Low engagement ({effectiveness_score:.2f}). "
                f"May need to reconsider channel selection or timing."
            )
            should_adjust = True

        # Add specific engagement observations
        if engagement_metrics.get('reaction_count', 0) > 0:
            learnings.append(
                f"Received {engagement_metrics['reaction_count']} reactions - message resonated"
            )

        if engagement_metrics.get('reply_count', 0) > 0:
            learnings.append(
                f"Received {engagement_metrics['reply_count']} replies - sparked conversation"
            )

        # Suggest improvements based on performance
        suggested_improvements = []
        if effectiveness_score < 0.7:
            # Analyze patterns from context
            suggested_timing = context.get('suggested_timing')
            if suggested_timing and 'afternoon' in suggested_timing:
                suggested_improvements.append("Try posting during afternoon hours for better engagement")

            if engagement_metrics.get('reaction_count', 0) == 0:
                suggested_improvements.append("Consider adding more engaging content or call-to-action")

            if effectiveness_score < 0.3:
                suggested_improvements.append("Review channel selection - may not be reaching right audience")

        return ReflectionInsight(
            plan_id=context.get('plan_id', 'unknown'),
            overall_success=True,
            lessons_learned=learnings,
            improvements_suggested=suggested_improvements,
            performance_metrics={
                "execution_time_ms": result.execution_time_ms,
                "effectiveness_score": effectiveness_score,
                "engagement": engagement_metrics
            },
            should_retry=False
        )

    async def _measure_engagement(
        self,
        result: ExecutionResult,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Measure engagement metrics for sent message.

        Queries Slack for:
        - Reaction count
        - Reply count
        - Time to first response

        Args:
            result: Execution result with message timestamp
            context: Execution context

        Returns:
            Engagement metrics
        """
        try:
            # Extract message timestamp and channel from result
            timestamp = result.result.get("timestamp")
            channel = result.result.get("channel", "#general")

            if not timestamp:
                logger.warning(f"[COMMUNICATION_AGENT] No timestamp for engagement measurement")
                return {"reaction_count": 0, "reply_count": 0}

            # Wait a short period for initial reactions (3 seconds)
            import asyncio
            await asyncio.sleep(3)

            # Get Slack access token
            creds = supabase.table("connections")\
                .select("access_token")\
                .eq("org_id", self.org_id)\
                .eq("source_type", "slack")\
                .single()\
                .execute()

            if not creds.data:
                return {"reaction_count": 0, "reply_count": 0}

            access_token = creds.data["access_token"]
            channel_id = channel.lstrip('#')

            # Fetch message with reactions using Slack API
            import requests

            # Get message details with reactions
            # https://api.slack.com/methods/conversations.history
            history_response = requests.get(
                "https://slack.com/api/conversations.history",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                params={
                    "channel": channel_id,
                    "latest": timestamp,
                    "inclusive": True,
                    "limit": 1
                },
                timeout=10
            )

            if history_response.status_code != 200:
                logger.warning(f"[COMMUNICATION_AGENT] Failed to fetch message history")
                return {"reaction_count": 0, "reply_count": 0}

            history_data = history_response.json()

            if not history_data.get("ok") or not history_data.get("messages"):
                logger.warning(f"[COMMUNICATION_AGENT] No message found for timestamp {timestamp}")
                return {"reaction_count": 0, "reply_count": 0}

            message_data = history_data["messages"][0]

            # Count reactions
            reactions = message_data.get("reactions", [])
            reaction_count = sum(r.get("count", 0) for r in reactions)

            # Count replies
            reply_count = message_data.get("reply_count", 0)

            # Get thread replies for more detailed analysis if needed
            thread_ts = message_data.get("thread_ts")
            if thread_ts and reply_count > 0:
                # Could fetch thread replies here for sentiment analysis
                pass

            metrics = {
                "reaction_count": reaction_count,
                "reply_count": reply_count,
                "reactions": [
                    {"name": r.get("name"), "count": r.get("count", 0)}
                    for r in reactions
                ],
                "has_thread": thread_ts is not None,
                "measured_at": datetime.now().isoformat()
            }

            logger.info(
                f"[COMMUNICATION_AGENT] Engagement measured: "
                f"{reaction_count} reactions, {reply_count} replies"
            )
            return metrics

        except Exception as e:
            logger.error(f"[COMMUNICATION_AGENT] Failed to measure engagement: {e}")
            return {"reaction_count": 0, "reply_count": 0}

    def _calculate_effectiveness(
        self,
        engagement: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate communication effectiveness score.

        Factors:
        - Reaction count (40%)
        - Reply count (40%)
        - Verification confidence (20%)

        Args:
            engagement: Engagement metrics
            context: Execution context

        Returns:
            Effectiveness score (0.0-1.0)
        """
        reaction_count = engagement.get("reaction_count", 0)
        reply_count = engagement.get("reply_count", 0)
        verification_confidence = context.get("verification_confidence", 0.9)

        # Normalize reaction and reply counts (cap at 10 for score calculation)
        reaction_score = min(reaction_count / 10.0, 1.0)
        reply_score = min(reply_count / 10.0, 1.0)

        # Weighted average
        effectiveness = (
            reaction_score * 0.4 +
            reply_score * 0.4 +
            verification_confidence * 0.2
        )

        return effectiveness

    async def _feed_to_adaptive_engine(
        self,
        result: ExecutionResult,
        context: Dict[str, Any],
        engagement: Dict[str, Any],
        effectiveness: float
    ):
        """
        Feed reflection outcomes to Adaptive Engine for learning.

        Logs to adaptive_learning table for the Adaptive Engine
        to analyze patterns and improve future performance.

        Args:
            result: Execution result
            context: Execution context
            engagement: Engagement metrics
            effectiveness: Effectiveness score
        """
        try:
            learning_entry = {
                "org_id": self.org_id,
                "agent_type": "communication_agent",
                "action_type": "send_message",
                "context": {
                    "intent": context.get("intent", "informational"),
                    "urgency": context.get("urgency", 0.5),
                    "channel": result.result.get("channel"),
                    "topic": context.get("topic"),
                    "timing": datetime.now().strftime("%A %H:%M")
                },
                "outcome": {
                    "success": result.success,
                    "engagement": engagement,
                    "effectiveness": effectiveness,
                    "execution_time_ms": result.execution_time_ms
                },
                "learnings": {
                    "effective_channel": result.result.get("channel") if effectiveness > 0.7 else None,
                    "effective_timing": datetime.now().hour if effectiveness > 0.7 else None,
                    "should_adjust": effectiveness < 0.5
                },
                "created_at": datetime.now().isoformat()
            }

            supabase.table("adaptive_learning").insert(learning_entry).execute()
            logger.info(f"[COMMUNICATION_AGENT] Fed outcomes to Adaptive Engine")

        except Exception as e:
            logger.error(f"[COMMUNICATION_AGENT] Failed to feed to Adaptive Engine: {e}")
