"""
Agent Event Bus Service

Enables cross-agent communication and coordination through a pub/sub event system.
Agents can publish events when they complete actions, and other agents can
subscribe to and consume these events to trigger coordinated workflows.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum

from app.core.logging import logger, log_error


class AgentEventType(str, Enum):
    """Types of events that can be published between agents"""
    CAMPAIGN_COMPLETED = "campaign_completed"
    TASK_CREATED = "task_created"
    MESSAGE_SENT = "message_sent"
    INSIGHT_GENERATED = "insight_generated"
    WORKFLOW_COMPLETED = "workflow_completed"
    ACTION_REQUIRES_FOLLOWUP = "action_requires_followup"
    APPROVAL_NEEDED = "approval_needed"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"


class AgentEventBus:
    """
    Pub/Sub event bus for inter-agent communication.

    Agents publish events when they complete significant actions.
    Other agents subscribe to event types and poll for new events.
    """

    def __init__(self, supabase):
        """
        Initialize the agent event bus.

        Args:
            supabase: Supabase client for database operations
        """
        self.supabase = supabase
        logger.info("[AGENT_EVENT_BUS] Initialized")

    def publish(
        self,
        org_id: str,
        event_type: AgentEventType,
        source_agent: str,
        payload: Dict[str, Any],
        target_agent: Optional[str] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Publish an event to the event bus.

        Args:
            org_id: Organization ID
            event_type: Type of event (from AgentEventType enum)
            source_agent: Agent that published the event
            payload: Event data (must be JSON-serializable)
            target_agent: Specific agent to receive event (None = broadcast)
            priority: Event priority (low/normal/high/critical)

        Returns:
            Dict containing the created event record
        """
        try:
            event_id = str(uuid.uuid4())

            event_data = {
                "id": event_id,
                "org_id": org_id,
                "event_type": event_type.value,
                "source_agent": source_agent,
                "target_agent": target_agent,
                "payload": payload,
                "priority": priority,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat()
            }

            result = self.supabase.table("agent_events")\
                .insert(event_data)\
                .execute()

            logger.info(
                f"[AGENT_EVENT_BUS] Published event",
                extra={
                    "event_id": event_id,
                    "event_type": event_type.value,
                    "source": source_agent,
                    "target": target_agent or "broadcast",
                    "org_id": org_id
                }
            )

            return result.data[0] if result.data else event_data

        except Exception as e:
            log_error(e, context="AgentEventBus.publish")
            raise

    def subscribe(
        self,
        agent_type: str,
        event_types: List[AgentEventType]
    ) -> None:
        """
        Register an agent's subscription to specific event types.

        This is metadata only - actual event consumption happens via poll_events().
        Useful for discovery and monitoring.

        Args:
            agent_type: Type of subscribing agent
            event_types: List of event types to subscribe to
        """
        try:
            subscription_data = {
                "agent_type": agent_type,
                "event_types": [et.value for et in event_types],
                "subscribed_at": datetime.utcnow().isoformat()
            }

            # Store in agent_subscriptions table (if it exists)
            # For now, just log it
            logger.info(
                f"[AGENT_EVENT_BUS] Agent subscribed",
                extra={
                    "agent": agent_type,
                    "event_types": [et.value for et in event_types]
                }
            )

        except Exception as e:
            log_error(e, context="AgentEventBus.subscribe")

    def poll_events(
        self,
        org_id: str,
        agent_type: Optional[str] = None,
        event_types: Optional[List[AgentEventType]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Poll for pending events from the event bus.

        Args:
            org_id: Organization ID
            agent_type: Filter for events targeted at this agent (None = all)
            event_types: Filter for specific event types (None = all)
            limit: Maximum number of events to retrieve

        Returns:
            List of pending events
        """
        try:
            query = self.supabase.table("agent_events")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("status", "pending")\
                .order("created_at", desc=False)\
                .limit(limit)

            # Filter by target agent (broadcast or specific)
            if agent_type:
                query = query.or_(f"target_agent.is.null,target_agent.eq.{agent_type}")

            # Filter by event types
            if event_types:
                event_type_values = [et.value for et in event_types]
                query = query.in_("event_type", event_type_values)

            result = query.execute()

            events = result.data if result.data else []

            if events:
                logger.info(
                    f"[AGENT_EVENT_BUS] Polled {len(events)} events",
                    extra={
                        "org_id": org_id,
                        "agent": agent_type,
                        "count": len(events)
                    }
                )

            return events

        except Exception as e:
            log_error(e, context="AgentEventBus.poll_events")
            return []

    def mark_consumed(
        self,
        event_id: str,
        consumed_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Mark an event as consumed by an agent.

        Args:
            event_id: ID of the event
            consumed_by: Agent that consumed the event
            notes: Optional notes about how the event was handled

        Returns:
            True if successfully marked, False otherwise
        """
        try:
            update_data = {
                "status": "consumed",
                "consumed_by": consumed_by,
                "consumed_at": datetime.utcnow().isoformat(),
                "notes": notes
            }

            result = self.supabase.table("agent_events")\
                .update(update_data)\
                .eq("id", event_id)\
                .execute()

            logger.info(
                f"[AGENT_EVENT_BUS] Event consumed",
                extra={
                    "event_id": event_id,
                    "consumer": consumed_by
                }
            )

            return True

        except Exception as e:
            log_error(e, context="AgentEventBus.mark_consumed")
            return False

    def get_event_history(
        self,
        org_id: str,
        agent_type: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get historical events for monitoring and debugging.

        Args:
            org_id: Organization ID
            agent_type: Filter by source or target agent
            hours_back: How many hours of history to retrieve
            limit: Maximum number of events

        Returns:
            List of historical events
        """
        try:
            cutoff_time = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()

            query = self.supabase.table("agent_events")\
                .select("*")\
                .eq("org_id", org_id)\
                .gte("created_at", cutoff_time)\
                .order("created_at", desc=True)\
                .limit(limit)

            if agent_type:
                query = query.or_(f"source_agent.eq.{agent_type},target_agent.eq.{agent_type}")

            result = query.execute()

            return result.data if result.data else []

        except Exception as e:
            log_error(e, context="AgentEventBus.get_event_history")
            return []

    def cleanup_old_events(
        self,
        org_id: str,
        days_old: int = 7
    ) -> int:
        """
        Clean up consumed events older than specified days.

        Args:
            org_id: Organization ID
            days_old: Delete consumed events older than this many days

        Returns:
            Number of events deleted
        """
        try:
            cutoff_time = (datetime.utcnow() - timedelta(days=days_old)).isoformat()

            result = self.supabase.table("agent_events")\
                .delete()\
                .eq("org_id", org_id)\
                .eq("status", "consumed")\
                .lt("created_at", cutoff_time)\
                .execute()

            deleted_count = len(result.data) if result.data else 0

            logger.info(
                f"[AGENT_EVENT_BUS] Cleaned up {deleted_count} old events",
                extra={"org_id": org_id, "days_old": days_old}
            )

            return deleted_count

        except Exception as e:
            log_error(e, context="AgentEventBus.cleanup_old_events")
            return 0


# Global instance (lazy initialization)
_agent_event_bus_instance = None


def get_agent_event_bus(supabase) -> AgentEventBus:
    """
    Get or create the global AgentEventBus instance.

    Args:
        supabase: Supabase client

    Returns:
        AgentEventBus instance
    """
    global _agent_event_bus_instance

    if _agent_event_bus_instance is None:
        _agent_event_bus_instance = AgentEventBus(supabase)

    return _agent_event_bus_instance
