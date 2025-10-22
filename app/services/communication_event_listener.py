"""
Communication Event Listener - Phase 4 Enhanced
Autonomous event-driven communication behavior

Listens to system events and triggers proactive communications:
- Insight generation events → Notify relevant stakeholders
- Action plan completion → Send summary to team
- Critical errors → Alert on-call team
- Milestone achievements → Announce to organization
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from supabase import Client
from app.core.logging import logger
from app.core.config import settings
import asyncio


class CommunicationEventListener:
    """
    Event listener for autonomous communication triggering.

    Monitors database tables for events and triggers appropriate
    communications based on event type, importance, and context.
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.running = False
        self.last_check_times = {
            "events": datetime.now(),
            "ai_insights": datetime.now(),
            "ai_action_plans": datetime.now(),
            "adaptive_learning": datetime.now()
        }

    async def start_listening(self, org_id: str, poll_interval: int = 30):
        """
        Start listening for events.

        Args:
            org_id: Organization ID to monitor
            poll_interval: Seconds between polls (default: 30)
        """
        self.running = True
        logger.info(f"[COMM_EVENT_LISTENER] Started listening for org {org_id}")

        while self.running:
            try:
                # Check all event sources
                await self._check_insight_events(org_id)
                await self._check_action_plan_events(org_id)
                await self._check_system_events(org_id)
                await self._check_learning_events(org_id)

                # Sleep before next poll
                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.error(f"[COMM_EVENT_LISTENER] Error in event loop: {e}")
                await asyncio.sleep(poll_interval)

    def stop_listening(self):
        """Stop the event listener."""
        self.running = False
        logger.info(f"[COMM_EVENT_LISTENER] Stopped listening")

    async def _check_insight_events(self, org_id: str):
        """
        Check for new AI insights that should be communicated.

        Triggers communications for:
        - High-importance insights (>0.8)
        - Strategic insights
        - Actionable insights
        """
        try:
            # Query for new insights since last check
            since_time = self.last_check_times["ai_insights"].isoformat()

            result = self.supabase.table("ai_insights")\
                .select("*")\
                .eq("org_id", org_id)\
                .gte("created_at", since_time)\
                .order("created_at", desc=False)\
                .execute()

            if not result.data:
                return

            logger.info(f"[COMM_EVENT_LISTENER] Found {len(result.data)} new insights")

            for insight in result.data:
                await self._handle_insight_event(insight, org_id)

            # Update last check time
            self.last_check_times["ai_insights"] = datetime.now()

        except Exception as e:
            logger.error(f"[COMM_EVENT_LISTENER] Failed to check insight events: {e}")

    async def _handle_insight_event(self, insight: Dict[str, Any], org_id: str):
        """
        Handle individual insight event.

        Determines if insight should trigger communication based on:
        - Importance score
        - Insight type
        - Tags
        - Previous communications
        """
        insight_id = insight.get("id")
        importance = insight.get("importance", 0.5)
        insight_type = insight.get("insight_type", "general")
        title = insight.get("title", "New Insight")

        logger.info(
            f"[COMM_EVENT_LISTENER] Processing insight: {title} "
            f"(importance: {importance:.2f})"
        )

        # Only communicate high-importance insights
        if importance < 0.8:
            logger.debug(f"[COMM_EVENT_LISTENER] Skipping low-importance insight")
            return

        # Check if we've already communicated this insight
        existing = self.supabase.table("communication_events")\
            .select("id")\
            .eq("org_id", org_id)\
            .eq("source_type", "insight")\
            .eq("source_id", insight_id)\
            .execute()

        if existing.data:
            logger.debug(f"[COMM_EVENT_LISTENER] Insight already communicated")
            return

        # Determine communication strategy based on insight type
        if insight_type == "strategic":
            # Strategic insights → #general + stakeholders
            await self._trigger_communication(
                org_id=org_id,
                event_type="insight_strategic",
                title=f"Strategic Insight: {title}",
                message=self._format_insight_message(insight),
                channels=["#general", "#strategy"],
                urgency="high",
                source_type="insight",
                source_id=insight_id
            )

        elif insight_type == "actionable":
            # Actionable insights → relevant team channel
            await self._trigger_communication(
                org_id=org_id,
                event_type="insight_actionable",
                title=f"Actionable Insight: {title}",
                message=self._format_insight_message(insight),
                channels=["#insights", "#general"],
                urgency="medium",
                source_type="insight",
                source_id=insight_id
            )

        else:
            # General high-importance insights → #insights
            await self._trigger_communication(
                org_id=org_id,
                event_type="insight_general",
                title=f"New Insight: {title}",
                message=self._format_insight_message(insight),
                channels=["#insights"],
                urgency="low",
                source_type="insight",
                source_id=insight_id
            )

    async def _check_action_plan_events(self, org_id: str):
        """
        Check for completed action plans that should be communicated.

        Triggers communications for:
        - Completed strategic plans
        - Failed critical plans
        - Long-running plans reaching milestones
        """
        try:
            since_time = self.last_check_times["ai_action_plans"].isoformat()

            # Query for completed plans since last check
            result = self.supabase.table("ai_action_plans")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("status", "completed")\
                .gte("updated_at", since_time)\
                .execute()

            if not result.data:
                return

            logger.info(f"[COMM_EVENT_LISTENER] Found {len(result.data)} completed plans")

            for plan in result.data:
                await self._handle_plan_completion_event(plan, org_id)

            self.last_check_times["ai_action_plans"] = datetime.now()

        except Exception as e:
            logger.error(f"[COMM_EVENT_LISTENER] Failed to check action plan events: {e}")

    async def _handle_plan_completion_event(self, plan: Dict[str, Any], org_id: str):
        """Handle action plan completion event."""
        plan_id = plan.get("id")
        goal = plan.get("goal", "Action plan")
        total_steps = plan.get("total_steps", 0)
        risk_level = plan.get("risk_level", "low")

        logger.info(f"[COMM_EVENT_LISTENER] Processing plan completion: {goal}")

        # Check if already communicated
        existing = self.supabase.table("communication_events")\
            .select("id")\
            .eq("org_id", org_id)\
            .eq("source_type", "action_plan")\
            .eq("source_id", plan_id)\
            .execute()

        if existing.data:
            return

        # Communicate strategic or high-risk plan completions
        if risk_level in ["medium", "high"] or total_steps >= 3:
            await self._trigger_communication(
                org_id=org_id,
                event_type="plan_completed",
                title=f"Action Plan Completed: {goal}",
                message=self._format_plan_completion_message(plan),
                channels=["#general", "#ai-updates"],
                urgency="medium" if risk_level == "high" else "low",
                source_type="action_plan",
                source_id=plan_id
            )

    async def _check_system_events(self, org_id: str):
        """
        Check for system events requiring communication.

        Monitors:
        - Critical errors
        - Integration failures
        - Performance degradation
        """
        try:
            since_time = self.last_check_times["events"].isoformat()

            # Query for critical events
            result = self.supabase.table("events")\
                .select("*")\
                .eq("org_id", org_id)\
                .in_("event_type", ["error", "critical", "integration_failure"])\
                .gte("happened_at", since_time)\
                .execute()

            if not result.data:
                return

            logger.info(f"[COMM_EVENT_LISTENER] Found {len(result.data)} system events")

            for event in result.data:
                await self._handle_system_event(event, org_id)

            self.last_check_times["events"] = datetime.now()

        except Exception as e:
            logger.error(f"[COMM_EVENT_LISTENER] Failed to check system events: {e}")

    async def _handle_system_event(self, event: Dict[str, Any], org_id: str):
        """Handle critical system event."""
        event_type = event.get("event_type")
        source = event.get("source", "system")
        payload = event.get("payload", {})

        logger.warning(f"[COMM_EVENT_LISTENER] Processing critical event: {event_type}")

        # Only communicate unique errors (don't spam)
        error_signature = f"{event_type}_{source}_{payload.get('error_code', 'unknown')}"

        # Check if similar error communicated in last hour
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        recent = self.supabase.table("communication_events")\
            .select("id")\
            .eq("org_id", org_id)\
            .eq("event_type", "system_error")\
            .contains("metadata", {"error_signature": error_signature})\
            .gte("created_at", one_hour_ago)\
            .execute()

        if recent.data:
            logger.debug(f"[COMM_EVENT_LISTENER] Similar error already reported")
            return

        # Trigger urgent communication
        await self._trigger_communication(
            org_id=org_id,
            event_type="system_error",
            title=f"System Alert: {event_type}",
            message=self._format_system_event_message(event),
            channels=["#tech-alerts", "#general"],
            urgency="urgent",
            source_type="system_event",
            source_id=event.get("id"),
            metadata={"error_signature": error_signature}
        )

    async def _check_learning_events(self, org_id: str):
        """
        Check for learning milestones worth communicating.

        Examples:
        - Agent performance improvements
        - New patterns discovered
        - Significant efficiency gains
        """
        try:
            since_time = self.last_check_times["adaptive_learning"].isoformat()

            # Query for significant learning events
            result = self.supabase.table("adaptive_learning")\
                .select("*")\
                .eq("org_id", org_id)\
                .gte("created_at", since_time)\
                .execute()

            if not result.data:
                return

            # Aggregate learning events to find patterns
            high_effectiveness = [
                l for l in result.data
                if l.get("outcome", {}).get("effectiveness", 0) >= 0.9
            ]

            # Communicate if we have significant improvements
            if len(high_effectiveness) >= 5:
                logger.info(
                    f"[COMM_EVENT_LISTENER] Found {len(high_effectiveness)} "
                    f"high-effectiveness learning events"
                )

                await self._trigger_communication(
                    org_id=org_id,
                    event_type="learning_milestone",
                    title="AI Performance Milestone Reached",
                    message=self._format_learning_milestone_message(high_effectiveness),
                    channels=["#ai-updates"],
                    urgency="low",
                    source_type="learning",
                    source_id="milestone"
                )

            self.last_check_times["adaptive_learning"] = datetime.now()

        except Exception as e:
            logger.error(f"[COMM_EVENT_LISTENER] Failed to check learning events: {e}")

    async def _trigger_communication(
        self,
        org_id: str,
        event_type: str,
        title: str,
        message: str,
        channels: List[str],
        urgency: str,
        source_type: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Trigger a communication by creating a communication event.

        This will be picked up by the CommunicationAgent for execution.
        """
        try:
            # Create communication event
            comm_event = {
                "org_id": org_id,
                "event_type": event_type,
                "title": title,
                "message": message,
                "channels": channels,
                "urgency": urgency,
                "source_type": source_type,
                "source_id": source_id,
                "metadata": metadata or {},
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }

            result = self.supabase.table("communication_events").insert(comm_event).execute()

            logger.info(
                f"[COMM_EVENT_LISTENER] Triggered communication: {title} "
                f"(urgency: {urgency}, channels: {channels})"
            )

            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"[COMM_EVENT_LISTENER] Failed to trigger communication: {e}")
            return None

    def _format_insight_message(self, insight: Dict[str, Any]) -> str:
        """Format insight into communication message."""
        title = insight.get("title", "Insight")
        content = insight.get("content", "")
        importance = insight.get("importance", 0.5)

        message = f"**{title}**\n\n"
        message += f"{content[:500]}"  # First 500 chars

        if len(content) > 500:
            message += "...\n\n_[View full insight in dashboard]_"

        message += f"\n\n_Importance: {importance:.2f}_"

        return message

    def _format_plan_completion_message(self, plan: Dict[str, Any]) -> str:
        """Format action plan completion into communication message."""
        goal = plan.get("goal", "Action plan")
        total_steps = plan.get("total_steps", 0)
        description = plan.get("description", "")

        message = f"**Completed: {goal}**\n\n"
        message += f"{description}\n\n"
        message += f"✅ Successfully completed {total_steps} steps\n"
        message += f"\n_[View details in AI Brain dashboard]_"

        return message

    def _format_system_event_message(self, event: Dict[str, Any]) -> str:
        """Format system event into communication message."""
        event_type = event.get("event_type")
        source = event.get("source", "system")
        payload = event.get("payload", {})

        message = f"🚨 **System Alert**\n\n"
        message += f"**Type:** {event_type}\n"
        message += f"**Source:** {source}\n"

        if payload.get("error_message"):
            message += f"**Error:** {payload['error_message'][:200]}\n"

        message += f"\n_Please check system logs for details_"

        return message

    def _format_learning_milestone_message(self, learning_events: List[Dict[str, Any]]) -> str:
        """Format learning milestone into communication message."""
        count = len(learning_events)
        avg_effectiveness = sum(
            l.get("outcome", {}).get("effectiveness", 0)
            for l in learning_events
        ) / count if count > 0 else 0

        message = f"🎯 **AI Performance Milestone**\n\n"
        message += f"The AI has achieved {count} high-effectiveness communications "
        message += f"(avg effectiveness: {avg_effectiveness:.2f})\n\n"
        message += "The system is continuously learning and improving!\n"
        message += f"\n_[View AI Performance metrics in dashboard]_"

        return message


def get_communication_event_listener(supabase: Client) -> CommunicationEventListener:
    """Factory function to get CommunicationEventListener instance."""
    return CommunicationEventListener(supabase)
