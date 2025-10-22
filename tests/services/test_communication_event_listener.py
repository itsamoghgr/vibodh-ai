"""
Tests for CommunicationEventListener
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from app.services.communication_event_listener import CommunicationEventListener


@pytest.fixture
def mock_supabase():
    """Create mock Supabase client."""
    mock = Mock()
    mock.table.return_value = mock
    return mock


@pytest.fixture
def event_listener(mock_supabase):
    """Create CommunicationEventListener instance."""
    return CommunicationEventListener(mock_supabase)


class TestInsightEvents:
    """Tests for insight event handling."""

    @pytest.mark.asyncio
    async def test_check_insight_events_high_importance(self, event_listener, mock_supabase):
        """Test handling of high-importance insights."""
        mock_result = Mock()
        mock_result.data = [
            {
                "id": "insight_1",
                "title": "Critical Market Trend",
                "importance": 0.9,
                "insight_type": "strategic",
                "created_at": datetime.now().isoformat()
            }
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value = mock_result

        # Mock communication event check (no duplicates)
        mock_check = Mock()
        mock_check.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_check

        # Mock trigger communication
        with patch.object(event_listener, '_trigger_communication', new_callable=AsyncMock) as mock_trigger:
            await event_listener._check_insight_events("test_org")

            # Should trigger communication for high-importance insight
            assert mock_trigger.called
            call_args = mock_trigger.call_args[1]
            assert call_args["event_type"] == "insight_strategic"
            assert "Strategic Insight" in call_args["title"]

    @pytest.mark.asyncio
    async def test_check_insight_events_low_importance_skipped(self, event_listener, mock_supabase):
        """Test that low-importance insights are skipped."""
        mock_result = Mock()
        mock_result.data = [
            {
                "id": "insight_1",
                "title": "Minor Update",
                "importance": 0.5,
                "insight_type": "general",
                "created_at": datetime.now().isoformat()
            }
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value = mock_result

        with patch.object(event_listener, '_trigger_communication', new_callable=AsyncMock) as mock_trigger:
            await event_listener._check_insight_events("test_org")

            # Should NOT trigger communication for low-importance
            assert not mock_trigger.called

    @pytest.mark.asyncio
    async def test_check_insight_events_duplicate_skipped(self, event_listener, mock_supabase):
        """Test that already-communicated insights are skipped."""
        mock_result = Mock()
        mock_result.data = [
            {
                "id": "insight_1",
                "title": "Important Insight",
                "importance": 0.9,
                "insight_type": "strategic",
                "created_at": datetime.now().isoformat()
            }
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value = mock_result

        # Mock duplicate check (already exists)
        mock_check = Mock()
        mock_check.data = [{"id": "existing_comm"}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_check

        with patch.object(event_listener, '_trigger_communication', new_callable=AsyncMock) as mock_trigger:
            await event_listener._check_insight_events("test_org")

            # Should NOT trigger duplicate communication
            assert not mock_trigger.called

    @pytest.mark.asyncio
    async def test_handle_actionable_insight(self, event_listener):
        """Test handling of actionable insights."""
        insight = {
            "id": "insight_1",
            "title": "Revenue Opportunity",
            "importance": 0.85,
            "insight_type": "actionable",
            "content": "Detected 20% increase in conversion potential"
        }

        with patch.object(event_listener, '_trigger_communication', new_callable=AsyncMock) as mock_trigger:
            # Mock duplicate check
            event_listener.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value.data = []

            await event_listener._handle_insight_event(insight, "test_org")

            assert mock_trigger.called
            call_args = mock_trigger.call_args[1]
            assert call_args["event_type"] == "insight_actionable"
            assert "#insights" in call_args["channels"]


class TestActionPlanEvents:
    """Tests for action plan completion events."""

    @pytest.mark.asyncio
    async def test_check_plan_completion_strategic(self, event_listener, mock_supabase):
        """Test handling of strategic plan completion."""
        mock_result = Mock()
        mock_result.data = [
            {
                "id": "plan_1",
                "goal": "Launch new product feature",
                "status": "completed",
                "total_steps": 5,
                "risk_level": "medium",
                "updated_at": datetime.now().isoformat()
            }
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.execute.return_value = mock_result

        # Mock duplicate check
        mock_check = Mock()
        mock_check.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_check

        with patch.object(event_listener, '_trigger_communication', new_callable=AsyncMock) as mock_trigger:
            await event_listener._check_action_plan_events("test_org")

            assert mock_trigger.called
            call_args = mock_trigger.call_args[1]
            assert call_args["event_type"] == "plan_completed"
            assert "Action Plan Completed" in call_args["title"]

    @pytest.mark.asyncio
    async def test_check_plan_completion_low_risk_skipped(self, event_listener, mock_supabase):
        """Test that simple low-risk plans are skipped."""
        mock_result = Mock()
        mock_result.data = [
            {
                "id": "plan_1",
                "goal": "Simple task",
                "status": "completed",
                "total_steps": 1,
                "risk_level": "low",
                "updated_at": datetime.now().isoformat()
            }
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.execute.return_value = mock_result

        # Mock duplicate check
        mock_check = Mock()
        mock_check.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_check

        with patch.object(event_listener, '_trigger_communication', new_callable=AsyncMock) as mock_trigger:
            await event_listener._check_action_plan_events("test_org")

            # Simple low-risk plan should not trigger communication
            assert not mock_trigger.called


class TestSystemEvents:
    """Tests for system event handling."""

    @pytest.mark.asyncio
    async def test_check_system_events_critical_error(self, event_listener, mock_supabase):
        """Test handling of critical system errors."""
        mock_result = Mock()
        mock_result.data = [
            {
                "id": "event_1",
                "event_type": "critical",
                "source": "database",
                "payload": {
                    "error_message": "Connection timeout",
                    "error_code": "DB_TIMEOUT"
                },
                "happened_at": datetime.now().isoformat()
            }
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.in_.return_value.gte.return_value.execute.return_value = mock_result

        # Mock recent error check (no duplicates)
        mock_check = Mock()
        mock_check.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.contains.return_value.gte.return_value.execute.return_value = mock_check

        with patch.object(event_listener, '_trigger_communication', new_callable=AsyncMock) as mock_trigger:
            await event_listener._check_system_events("test_org")

            assert mock_trigger.called
            call_args = mock_trigger.call_args[1]
            assert call_args["event_type"] == "system_error"
            assert call_args["urgency"] == "urgent"
            assert "#tech-alerts" in call_args["channels"]

    @pytest.mark.asyncio
    async def test_check_system_events_duplicate_suppression(self, event_listener, mock_supabase):
        """Test that duplicate errors within 1 hour are suppressed."""
        mock_result = Mock()
        mock_result.data = [
            {
                "id": "event_1",
                "event_type": "error",
                "source": "api",
                "payload": {"error_code": "API_500"},
                "happened_at": datetime.now().isoformat()
            }
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.in_.return_value.gte.return_value.execute.return_value = mock_result

        # Mock recent similar error exists
        mock_check = Mock()
        mock_check.data = [{"id": "recent_comm"}]
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.contains.return_value.gte.return_value.execute.return_value = mock_check

        with patch.object(event_listener, '_trigger_communication', new_callable=AsyncMock) as mock_trigger:
            await event_listener._check_system_events("test_org")

            # Should NOT trigger duplicate alert
            assert not mock_trigger.called


class TestLearningEvents:
    """Tests for learning milestone events."""

    @pytest.mark.asyncio
    async def test_check_learning_milestone(self, event_listener, mock_supabase):
        """Test detection of learning milestones."""
        # Create 5 high-effectiveness learning events
        learning_events = [
            {
                "id": f"learning_{i}",
                "org_id": "test_org",
                "outcome": {"effectiveness": 0.95},
                "created_at": datetime.now().isoformat()
            }
            for i in range(6)
        ]

        mock_result = Mock()
        mock_result.data = learning_events

        mock_supabase.table.return_value.select.return_value.eq.return_value.gte.return_value.execute.return_value = mock_result

        with patch.object(event_listener, '_trigger_communication', new_callable=AsyncMock) as mock_trigger:
            await event_listener._check_learning_events("test_org")

            # Should trigger milestone communication
            assert mock_trigger.called
            call_args = mock_trigger.call_args[1]
            assert call_args["event_type"] == "learning_milestone"
            assert "Performance Milestone" in call_args["title"]

    @pytest.mark.asyncio
    async def test_check_learning_no_milestone(self, event_listener, mock_supabase):
        """Test no communication when milestone not reached."""
        # Only 2 high-effectiveness events (need 5+)
        learning_events = [
            {
                "id": f"learning_{i}",
                "outcome": {"effectiveness": 0.95},
                "created_at": datetime.now().isoformat()
            }
            for i in range(2)
        ]

        mock_result = Mock()
        mock_result.data = learning_events

        mock_supabase.table.return_value.select.return_value.eq.return_value.gte.return_value.execute.return_value = mock_result

        with patch.object(event_listener, '_trigger_communication', new_callable=AsyncMock) as mock_trigger:
            await event_listener._check_learning_events("test_org")

            # Should NOT trigger communication
            assert not mock_trigger.called


class TestCommunicationTriggering:
    """Tests for communication event triggering."""

    @pytest.mark.asyncio
    async def test_trigger_communication_success(self, event_listener, mock_supabase):
        """Test successful communication triggering."""
        mock_result = Mock()
        mock_result.data = [
            {
                "id": "comm_1",
                "org_id": "test_org",
                "event_type": "test_event",
                "status": "pending"
            }
        ]

        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_result

        result = await event_listener._trigger_communication(
            org_id="test_org",
            event_type="test_event",
            title="Test Communication",
            message="Test message",
            channels=["#general"],
            urgency="medium",
            source_type="test",
            source_id="test_1"
        )

        assert result is not None
        assert result["id"] == "comm_1"

    @pytest.mark.asyncio
    async def test_trigger_communication_with_metadata(self, event_listener, mock_supabase):
        """Test communication triggering with metadata."""
        mock_result = Mock()
        mock_result.data = [{"id": "comm_1"}]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_result

        result = await event_listener._trigger_communication(
            org_id="test_org",
            event_type="test_event",
            title="Test",
            message="Message",
            channels=["#test"],
            urgency="low",
            source_type="test",
            source_id="test_1",
            metadata={"custom": "data"}
        )

        # Verify insert was called with metadata
        insert_call = mock_supabase.table.return_value.insert.call_args[0][0]
        assert insert_call["metadata"]["custom"] == "data"


class TestMessageFormatting:
    """Tests for message formatting functions."""

    def test_format_insight_message(self, event_listener):
        """Test insight message formatting."""
        insight = {
            "title": "Market Trend",
            "content": "Detailed analysis of market conditions and trends...",
            "importance": 0.85
        }

        message = event_listener._format_insight_message(insight)

        assert "Market Trend" in message
        assert "0.85" in message
        assert len(message) > 0

    def test_format_insight_message_long_content(self, event_listener):
        """Test insight formatting truncates long content."""
        long_content = "A" * 1000
        insight = {
            "title": "Test",
            "content": long_content,
            "importance": 0.8
        }

        message = event_listener._format_insight_message(insight)

        # Should be truncated to ~500 chars + metadata
        assert len(message) < 700
        assert "View full insight" in message

    def test_format_plan_completion_message(self, event_listener):
        """Test plan completion message formatting."""
        plan = {
            "goal": "Complete project",
            "description": "Successfully completed all project phases",
            "total_steps": 5
        }

        message = event_listener._format_plan_completion_message(plan)

        assert "Complete project" in message
        assert "5 steps" in message
        assert "✅" in message

    def test_format_system_event_message(self, event_listener):
        """Test system event message formatting."""
        event = {
            "event_type": "critical_error",
            "source": "database",
            "payload": {
                "error_message": "Connection timeout after 30s"
            }
        }

        message = event_listener._format_system_event_message(event)

        assert "🚨" in message
        assert "critical_error" in message
        assert "database" in message
        assert "Connection timeout" in message

    def test_format_learning_milestone_message(self, event_listener):
        """Test learning milestone message formatting."""
        learning_events = [
            {"outcome": {"effectiveness": 0.9}},
            {"outcome": {"effectiveness": 0.95}},
            {"outcome": {"effectiveness": 0.92}},
        ]

        message = event_listener._format_learning_milestone_message(learning_events)

        assert "🎯" in message
        assert "3" in message
        assert "0.9" in message  # Average should be ~0.92


class TestEventLoopControl:
    """Tests for event loop control."""

    def test_stop_listening(self, event_listener):
        """Test stopping the event listener."""
        event_listener.running = True

        event_listener.stop_listening()

        assert event_listener.running is False

    @pytest.mark.asyncio
    async def test_start_listening_polls_all_sources(self, event_listener, mock_supabase):
        """Test that start_listening checks all event sources."""
        # Mock all data sources to return empty
        mock_result = Mock()
        mock_result.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value = mock_result
        mock_supabase.table.return_value.select.return_value.eq.return_value.in_.return_value.gte.return_value.execute.return_value = mock_result

        with patch.object(event_listener, '_check_insight_events', new_callable=AsyncMock) as mock_insights:
            with patch.object(event_listener, '_check_action_plan_events', new_callable=AsyncMock) as mock_plans:
                with patch.object(event_listener, '_check_system_events', new_callable=AsyncMock) as mock_system:
                    with patch.object(event_listener, '_check_learning_events', new_callable=AsyncMock) as mock_learning:
                        # Run one iteration then stop
                        import asyncio
                        async def run_once():
                            await asyncio.sleep(0.1)
                            event_listener.stop_listening()

                        await asyncio.gather(
                            event_listener.start_listening("test_org", poll_interval=0.05),
                            run_once()
                        )

                        # Verify all check methods were called
                        assert mock_insights.called
                        assert mock_plans.called
                        assert mock_system.called
                        assert mock_learning.called
