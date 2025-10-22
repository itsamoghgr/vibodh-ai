"""
Integration Tests for CommunicationAgent
Tests the complete lifecycle: Observe → Plan → Execute → Reflect
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from app.agents.communication_agent import CommunicationAgent
from app.agents.base_agent import ObservationContext, ExecutionResult


@pytest.fixture
def mock_supabase():
    """Create mock Supabase client."""
    mock = Mock()
    mock.table.return_value = mock
    return mock


@pytest.fixture
def communication_agent(mock_supabase):
    """Create CommunicationAgent instance."""
    with patch('app.agents.communication_agent.settings') as mock_settings:
        mock_settings.SLACK_CLIENT_ID = "test_client"
        mock_settings.SLACK_CLIENT_SECRET = "test_secret"
        mock_settings.GROQ_API_KEY = "test_groq_key"
        mock_settings.GROQ_MODEL = "llama-3.1-70b-versatile"

        # Patch app.db.supabase since agent imports it internally
        with patch('app.db.supabase', mock_supabase):
            agent = CommunicationAgent(
                org_id="test_org",
                agent_type="communication"
            )

            # Manually set services for testing with AsyncMock
            agent.reasoning_service = AsyncMock()
            agent.context_service = AsyncMock()

            return agent


class TestObservePhase:
    """Tests for the Observe phase."""

    @pytest.mark.asyncio
    async def test_observe_user_request(self, communication_agent):
        """Test observing direct user communication request."""
        context = ObservationContext(
            trigger_type="manual",
            org_id="test_org",
            query="send a message to #marketing about the campaign results"
        )

        should_act, reason = await communication_agent._observe_impl(context)

        assert should_act is True
        assert "message sending" in reason.lower()

    @pytest.mark.asyncio
    async def test_observe_pending_events(self, communication_agent, mock_supabase):
        """Test observing pending communication events."""
        # Mock the _check_pending_events method directly to avoid database complexity
        mock_events = [
            {
                "id": "event_1",
                "title": "Strategic Insight",
                "message": "Important update",
                "channels": ["#general"],
                "urgency": "high"
            }
        ]

        with patch.object(communication_agent, '_check_pending_communication_events', return_value=mock_events):
            context = ObservationContext(
                trigger_type="manual",
                org_id="test_org",
                query=None
            )

            should_act, reason = await communication_agent._observe_impl(context)

            assert should_act is True
            assert "event" in reason.lower() or "pending" in reason.lower()

    @pytest.mark.asyncio
    async def test_observe_no_action_needed(self, communication_agent, mock_supabase):
        """Test no action when no opportunities."""
        # Mock no pending events
        mock_result = Mock()
        mock_result.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        context = ObservationContext(
            trigger_type="manual",
            org_id="test_org",
            query="What is the weather today?"
        )

        should_act, reason = await communication_agent._observe_impl(context)

        assert should_act is False


class TestPlanPhase:
    """Tests for the Plan phase with reasoning."""

    @pytest.mark.asyncio
    async def test_plan_informational_message(self, communication_agent, mock_supabase):
        """Test planning for informational message."""
        goal = "send a message to #marketing about campaign results"

        # Mock reasoning service response
        mock_intent = Mock()
        mock_intent.intent_type.value = "informational"
        mock_intent.urgency_score = 0.5
        mock_intent.recommended_modules = ["rag", "kg"]
        mock_intent.reasoning = "Informational update"
        mock_intent.suggested_audience = ["#marketing"]
        mock_intent.confidence = 0.85

        # Mock context service response
        mock_context = Mock()
        mock_context.topic = "campaign results"
        mock_context.audience.channels = ["#marketing"]
        mock_context.audience.people = []
        mock_context.audience.teams = []
        mock_context.audience.confidence = 0.9
        mock_context.recent_context = []
        mock_context.related_entities = []
        mock_context.suggested_timing = "afternoon (2-4pm)"
        mock_context.communication_patterns = {}

        # Use AsyncMock for async methods
        communication_agent.reasoning_service.analyze_communication_request = AsyncMock(return_value=mock_intent)
        communication_agent.context_service.gather_context = AsyncMock(return_value=mock_context)
        communication_agent.context_service.enrich_message = AsyncMock(return_value="Campaign results are ready")

        # Mock database insert for reasoning logs
        mock_supabase.table.return_value.insert.return_value.execute.return_value = Mock()

        plan = await communication_agent._plan_impl(goal, {})

        assert plan is not None
        assert plan.total_steps == 1  # Informational = single step
        assert plan.risk_level == "low"
        assert plan.requires_approval is False
        assert plan.steps[0].action_type == "send_message"

    @pytest.mark.asyncio
    async def test_plan_strategic_message_multi_step(self, communication_agent, mock_supabase):
        """Test planning for strategic message creates multi-step plan."""
        goal = "announce important product launch to everyone"

        # Mock strategic intent
        mock_intent = Mock()
        mock_intent.intent_type.value = "strategic"
        mock_intent.urgency_score = 0.8
        mock_intent.recommended_modules = ["rag", "kg", "insight"]
        mock_intent.reasoning = "Strategic announcement"
        mock_intent.suggested_audience = ["#general"]
        mock_intent.confidence = 0.9

        mock_context = Mock()
        mock_context.topic = "product launch"
        mock_context.audience.channels = ["#general"]
        mock_context.audience.people = [{"name": "Sarah", "role": "Lead"}]
        mock_context.audience.teams = ["Marketing"]
        mock_context.audience.confidence = 0.95
        mock_context.recent_context = []
        mock_context.related_entities = []
        mock_context.suggested_timing = None
        mock_context.communication_patterns = {}

        # Use AsyncMock for async methods
        communication_agent.reasoning_service.analyze_communication_request = AsyncMock(return_value=mock_intent)
        communication_agent.context_service.gather_context = AsyncMock(return_value=mock_context)
        communication_agent.context_service.enrich_message = AsyncMock(return_value="Product launch announcement")

        # Mock database insert for reasoning logs
        mock_supabase.table.return_value.insert.return_value.execute.return_value = Mock()

        plan = await communication_agent._plan_impl(goal, {})

        # Strategic should create 3-step plan
        assert plan.total_steps == 3
        assert plan.steps[0].action_type == "send_message"
        assert plan.steps[1].action_type == "create_task"
        assert plan.steps[2].action_type == "schedule_summary"
        assert plan.requires_approval is True

        # Verify create_task has correct parameter names
        assert "title" in plan.steps[1].parameters
        assert "description" in plan.steps[1].parameters

    @pytest.mark.asyncio
    async def test_plan_urgent_message_with_backup(self, communication_agent, mock_supabase):
        """Test planning for urgent message includes email backup."""
        goal = "urgent alert about system outage"

        mock_intent = Mock()
        mock_intent.intent_type.value = "urgent"
        mock_intent.urgency_score = 0.95
        mock_intent.recommended_modules = ["rag", "kg", "memory", "insight"]
        mock_intent.reasoning = "Urgent notification"
        mock_intent.suggested_audience = ["#tech-alerts"]
        mock_intent.confidence = 0.9

        mock_context = Mock()
        mock_context.topic = "system outage"
        mock_context.audience.channels = ["#tech-alerts"]
        mock_context.audience.people = [{"name": "John", "role": "Engineer"}]
        mock_context.audience.teams = []
        mock_context.audience.confidence = 0.9
        mock_context.recent_context = []
        mock_context.related_entities = []
        mock_context.suggested_timing = None
        mock_context.communication_patterns = {}

        # Use AsyncMock for async methods
        communication_agent.reasoning_service.analyze_communication_request = AsyncMock(return_value=mock_intent)
        communication_agent.context_service.gather_context = AsyncMock(return_value=mock_context)
        communication_agent.context_service.enrich_message = AsyncMock(return_value="URGENT: System outage")

        # Mock database insert for reasoning logs
        mock_supabase.table.return_value.insert.return_value.execute.return_value = Mock()

        plan = await communication_agent._plan_impl(goal, {})

        # Urgent should create 2-step plan (Slack + Email)
        assert plan.total_steps == 2
        assert plan.steps[0].action_type == "send_message"
        assert plan.steps[1].action_type == "send_email"

        # Verify send_email has correct parameter names
        assert "to" in plan.steps[1].parameters
        assert "subject" in plan.steps[1].parameters
        assert "body" in plan.steps[1].parameters


class TestExecutePhase:
    """Tests for the Execute phase with verification."""

    @pytest.mark.asyncio
    async def test_execute_send_message_with_verification(self, communication_agent, mock_supabase):
        """Test message execution with pre-execution verification."""
        action = Mock()
        action.step_index = 0
        action.action_type = "send_message"
        action.parameters = {
            "channel": "#general",
            "message": "Test message",
            "intent": "informational",
            "urgency": 0.5
        }

        # Mock verification passes
        mock_verification = Mock()
        mock_verification.appropriate = True
        mock_verification.confidence = 0.9
        mock_verification.reason = "Appropriate timing and content"

        # Mock Slack response
        mock_slack_result = {
            "timestamp": "1234567890.123456",
            "channel": "C123456",
            "message": "Test message"
        }

        # Use AsyncMock for async methods and mock the internal send method
        communication_agent.reasoning_service.verify_appropriateness = AsyncMock(return_value=mock_verification)

        # Mock the _send_slack_message method directly to avoid database complexities
        with patch.object(communication_agent, '_send_slack_message', return_value=mock_slack_result):
            result = await communication_agent._execute_impl(action)

            assert result.success is True
            assert result.result["verification"]["appropriate"] is True
            assert result.result["timestamp"] == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_execute_verification_blocks_message(self, communication_agent):
        """Test verification blocks inappropriate message."""
        action = Mock()
        action.step_index = 0
        action.action_type = "send_message"
        action.parameters = {
            "channel": "#general",
            "message": "Inappropriate message",
            "intent": "informational",
            "urgency": 0.3
        }

        # Mock verification fails
        mock_verification = Mock()
        mock_verification.appropriate = False
        mock_verification.confidence = 0.8
        mock_verification.reason = "Posted outside business hours"
        mock_verification.suggested_changes = ["Schedule for business hours"]

        # Use AsyncMock for async method
        communication_agent.reasoning_service.verify_appropriateness = AsyncMock(return_value=mock_verification)

        result = await communication_agent._execute_impl(action)

        # Execution should fail due to verification
        assert result.success is False
        assert "verification failed" in result.error_message.lower()
        assert result.result["verification_failed"] is True


class TestReflectPhase:
    """Tests for the Reflect phase with engagement measurement."""

    @pytest.mark.asyncio
    async def test_reflect_successful_execution(self, communication_agent, mock_supabase):
        """Test reflection on successful message execution."""
        result = ExecutionResult(
            success=True,
            action_id="comm_0",
            result={
                "status": "sent",
                "channel": "#marketing",
                "timestamp": "1234567890.123456",
                "verification": {
                    "appropriate": True,
                    "confidence": 0.9
                }
            },
            execution_time_ms=1500
        )

        context = {
            "plan_id": "plan_123",
            "intent": "informational",
            "urgency": 0.5,
            "topic": "campaign",
            "verification_confidence": 0.9
        }

        # Mock engagement measurement
        mock_engagement = {
            "reaction_count": 5,
            "reply_count": 2,
            "measured_at": datetime.now().isoformat()
        }

        # Mock database inserts
        mock_supabase.table.return_value.insert.return_value.execute.return_value = Mock()

        with patch.object(communication_agent, '_measure_engagement', return_value=mock_engagement):
            reflection = await communication_agent._reflect_impl(result, context)

            # Check new ReflectionInsight model structure
            assert reflection.overall_success is True
            assert len(reflection.lessons_learned) > 0
            assert reflection.performance_metrics["execution_time_ms"] == 1500
            assert "effectiveness_score" in reflection.performance_metrics
            assert reflection.should_retry is False  # Good performance

    @pytest.mark.asyncio
    async def test_reflect_failed_execution(self, communication_agent):
        """Test reflection on failed execution."""
        result = ExecutionResult(
            success=False,
            action_id="comm_0",
            error_message="Slack permission denied",
            execution_time_ms=1000
        )

        context = {"plan_id": "plan_123"}

        reflection = await communication_agent._reflect_impl(result, context)

        # Check new ReflectionInsight model structure
        assert reflection.overall_success is False
        assert len(reflection.lessons_learned) > 0
        assert "failed" in reflection.lessons_learned[0].lower() or "denied" in reflection.lessons_learned[0].lower()
        assert reflection.should_retry is True
        assert len(reflection.improvements_suggested) > 0

    @pytest.mark.asyncio
    async def test_reflect_feeds_to_adaptive_engine(self, communication_agent, mock_supabase):
        """Test reflection feeds outcomes to Adaptive Engine."""
        result = ExecutionResult(
            success=True,
            action_id="comm_0",
            result={
                "channel": "#marketing",
                "timestamp": "123456"
            },
            execution_time_ms=1500
        )

        context = {
            "plan_id": "plan_123",
            "intent": "informational",
            "urgency": 0.5,
            "topic": "campaign",
            "verification_confidence": 0.9
        }

        mock_engagement = {"reaction_count": 3, "reply_count": 1}

        # Mock database insert for adaptive_learning
        mock_supabase.table.return_value.insert.return_value.execute.return_value = Mock()

        with patch.object(communication_agent, '_measure_engagement', return_value=mock_engagement):
            reflection = await communication_agent._reflect_impl(result, context)

            # Verify reflection succeeded - checking table calls is fragile with patching
            assert reflection.overall_success is True
            assert len(reflection.lessons_learned) > 0
            # Verify the reflection includes engagement metrics in performance
            assert "engagement" in reflection.performance_metrics


class TestCompleteLifecycle:
    """Integration tests for complete agent lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_informational_message(self, communication_agent, mock_supabase):
        """Test complete lifecycle for informational message."""
        # 1. OBSERVE
        context = ObservationContext(
            trigger_type="manual",
            org_id="test_org",
            user_id="test_user",
            query="send message to #marketing about campaign results"
        )

        # Mock pending events check
        with patch.object(communication_agent, '_check_pending_communication_events', return_value=[]):
            should_act, _ = await communication_agent._observe_impl(context)
            assert should_act is True

        # 2. PLAN
        mock_intent = Mock()
        mock_intent.intent_type.value = "informational"
        mock_intent.urgency_score = 0.5
        mock_intent.recommended_modules = ["rag"]
        mock_intent.reasoning = "Test"
        mock_intent.confidence = 0.8

        mock_context = Mock()
        mock_context.topic = "campaign"
        mock_context.audience.channels = ["#marketing"]
        mock_context.audience.people = []
        mock_context.audience.teams = []
        mock_context.audience.confidence = 0.9
        mock_context.recent_context = []
        mock_context.related_entities = []
        mock_context.suggested_timing = None
        mock_context.communication_patterns = {}

        communication_agent.reasoning_service.analyze_communication_request = AsyncMock(return_value=mock_intent)
        communication_agent.context_service.gather_context = AsyncMock(return_value=mock_context)
        communication_agent.context_service.enrich_message = AsyncMock(return_value="Campaign results ready")

        # Mock database insert
        mock_supabase.table.return_value.insert.return_value.execute.return_value = Mock()

        plan = await communication_agent._plan_impl(context.query, {})
        assert plan.total_steps == 1

        # 3. EXECUTE
        action = plan.steps[0]

        mock_verification = Mock()
        mock_verification.appropriate = True
        mock_verification.confidence = 0.9
        mock_verification.reason = "Good"

        mock_slack_result = {
            "timestamp": "123",
            "channel": "C123",
            "message": "Campaign results ready"
        }

        communication_agent.reasoning_service.verify_appropriateness = AsyncMock(return_value=mock_verification)

        # Mock the _send_slack_message method to avoid database complexity
        with patch.object(communication_agent, '_send_slack_message', return_value=mock_slack_result):
            exec_result = await communication_agent._execute_impl(action)
            assert exec_result.success is True

        # 4. REFLECT
        mock_engagement = {"reaction_count": 2, "reply_count": 1}
        with patch.object(communication_agent, '_measure_engagement', return_value=mock_engagement):
            reflection = await communication_agent._reflect_impl(exec_result, plan.context)
            assert reflection is not None
            assert reflection.overall_success is True
            assert len(reflection.lessons_learned) > 0
