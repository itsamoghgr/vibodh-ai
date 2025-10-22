"""
Action Planning Service - Phase 4, Step 1
Manages action plan lifecycle and execution queue
"""

from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from supabase import Client

from app.agents.base_agent import ActionPlan, ActionStep
from app.models.agent import (
    ActionPlanRequest, ActionPlanResponse,
    PendingActionResponse, ExecutedActionResponse,
    PlanStatus, ActionStatus, RiskLevel, TriggerType
)
from app.core.logging import logger


class ActionPlanningService:
    """
    Service for managing action plans and their lifecycle.

    Responsibilities:
    - Create and store action plans
    - Queue actions for approval
    - Track action execution
    - Manage plan status transitions
    """

    def __init__(self, supabase: Client):
        """
        Initialize action planning service.

        Args:
            supabase: Supabase client
        """
        self.supabase = supabase
        logger.info("[ACTION_PLANNING] Service initialized")

    async def create_action_plan(
        self,
        org_id: str,
        plan: ActionPlan,
        agent_type: str,
        trigger_type: TriggerType = TriggerType.MANUAL,
        trigger_source: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> ActionPlanResponse:
        """
        Create and store an action plan.

        Args:
            org_id: Organization ID
            plan: Action plan from agent
            agent_type: Type of agent that created the plan
            trigger_type: How the plan was triggered
            trigger_source: Source of the trigger
            user_id: User who initiated the plan

        Returns:
            Created action plan response
        """
        try:
            logger.info(f"[ACTION_PLANNING] Creating plan for goal: {plan.goal}")

            # Convert steps to JSONB format
            steps_json = [
                {
                    "step_index": step.step_index,
                    "action_type": step.action_type,
                    "action_name": step.action_name,
                    "description": step.description,
                    "target_integration": step.target_integration,
                    "target_resource": step.target_resource,
                    "parameters": step.parameters,
                    "risk_level": step.risk_level,
                    "requires_approval": step.requires_approval,
                    "depends_on": step.depends_on,
                    "estimated_duration_ms": step.estimated_duration_ms
                }
                for step in plan.steps
            ]

            # Determine initial status
            initial_status = PlanStatus.PENDING if plan.requires_approval else PlanStatus.APPROVED

            # Create plan in database
            plan_data = {
                "org_id": org_id,
                "agent_type": agent_type,
                "goal": plan.goal,
                "description": plan.description,
                "status": initial_status.value,
                "trigger_type": trigger_type.value,
                "trigger_source": trigger_source or {},
                "steps": steps_json,
                "total_steps": plan.total_steps,
                "completed_steps": 0,
                "risk_level": plan.risk_level,
                "requires_approval": plan.requires_approval,
                "approval_status": "pending" if plan.requires_approval else None,
                "context": plan.context,
                "initiated_by": user_id,
                # Add missing fields that were calculated by the agent
                "confidence_score": plan.confidence_score,
                "estimated_total_duration_ms": plan.estimated_total_duration_ms
            }

            result = self.supabase.table("ai_action_plans").insert(plan_data).execute()

            if not result.data:
                raise Exception("Failed to create action plan")

            created_plan = result.data[0]
            plan_id = created_plan["id"]

            logger.info(f"[ACTION_PLANNING] Created plan {plan_id} with {plan.total_steps} steps")

            # Queue actions if plan doesn't require approval
            if not plan.requires_approval:
                await self.queue_actions(plan_id, plan.steps, org_id)

            return ActionPlanResponse(**created_plan)

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to create plan: {e}")
            raise

    async def queue_actions(
        self,
        plan_id: str,
        steps: List[ActionStep],
        org_id: str
    ) -> List[PendingActionResponse]:
        """
        Queue actions for execution or approval.

        Args:
            plan_id: Plan ID
            steps: List of action steps
            org_id: Organization ID

        Returns:
            List of queued pending actions
        """
        try:
            logger.info(f"[ACTION_PLANNING] Queueing {len(steps)} actions for plan {plan_id}")

            pending_actions = []
            action_id_map = {}  # Map step index to action ID for dependencies

            for step in steps:
                # Calculate auto-approve timeout based on risk
                auto_approve_timeout = None
                if not step.requires_approval and step.risk_level == "low":
                    auto_approve_timeout = (datetime.utcnow() + timedelta(minutes=5)).isoformat()

                # Map dependencies
                depends_on = []
                for dep_index in step.depends_on:
                    if dep_index in action_id_map:
                        depends_on.append(action_id_map[dep_index])

                # Create pending action
                action_data = {
                    "org_id": org_id,
                    "plan_id": plan_id,
                    "step_index": step.step_index,
                    "action_type": step.action_type,
                    "action_name": step.action_name,
                    "description": step.description,
                    "target_integration": step.target_integration,
                    "target_resource": step.target_resource,
                    "parameters": step.parameters,
                    "risk_level": step.risk_level,
                    "requires_approval": step.requires_approval,
                    "auto_approve_timeout": auto_approve_timeout,
                    "depends_on": depends_on,
                    "status": "pending"
                }

                result = self.supabase.table("ai_actions_pending").insert(action_data).execute()

                if result.data:
                    action = result.data[0]
                    action_id_map[step.step_index] = action["id"]
                    pending_actions.append(PendingActionResponse(**action))

            logger.info(f"[ACTION_PLANNING] Queued {len(pending_actions)} actions")

            return pending_actions

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to queue actions: {e}")
            raise

    async def get_pending_actions(
        self,
        org_id: str,
        plan_id: Optional[str] = None,
        requires_approval: Optional[bool] = None
    ) -> List[PendingActionResponse]:
        """
        Get pending actions for an organization.

        Args:
            org_id: Organization ID
            plan_id: Optional plan ID filter
            requires_approval: Optional approval requirement filter

        Returns:
            List of pending actions
        """
        try:
            query = self.supabase.table("ai_actions_pending")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("status", "pending")

            if plan_id:
                query = query.eq("plan_id", plan_id)

            if requires_approval is not None:
                query = query.eq("requires_approval", requires_approval)

            result = query.order("created_at", desc=False).execute()

            return [PendingActionResponse(**action) for action in result.data]

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to get pending actions: {e}")
            raise

    async def approve_action(
        self,
        action_id: str,
        approved_by: str,
        org_id: str
    ) -> PendingActionResponse:
        """
        Approve a pending action.

        Args:
            action_id: Action ID
            approved_by: User ID who approved
            org_id: Organization ID

        Returns:
            Updated action
        """
        try:
            logger.info(f"[ACTION_PLANNING] Approving action {action_id}")

            # Update action status
            result = self.supabase.table("ai_actions_pending")\
                .update({
                    "status": "approved",
                    "approved_by": approved_by,
                    "approved_at": datetime.utcnow().isoformat()
                })\
                .eq("id", action_id)\
                .eq("org_id", org_id)\
                .execute()

            if not result.data:
                raise Exception("Action not found or already processed")

            action = result.data[0]

            # Check if all actions in the plan are approved
            await self._check_plan_approval_status(action["plan_id"], org_id)

            return PendingActionResponse(**action)

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to approve action: {e}")
            raise

    async def reject_action(
        self,
        action_id: str,
        rejected_by: str,
        reason: str,
        org_id: str
    ) -> PendingActionResponse:
        """
        Reject a pending action.

        Args:
            action_id: Action ID
            rejected_by: User ID who rejected
            reason: Rejection reason
            org_id: Organization ID

        Returns:
            Updated action
        """
        try:
            logger.info(f"[ACTION_PLANNING] Rejecting action {action_id}: {reason}")

            # Update action status
            result = self.supabase.table("ai_actions_pending")\
                .update({
                    "status": "rejected",
                    "approved_by": rejected_by,
                    "approved_at": datetime.utcnow().isoformat(),
                    "rejection_reason": reason
                })\
                .eq("id", action_id)\
                .eq("org_id", org_id)\
                .execute()

            if not result.data:
                raise Exception("Action not found or already processed")

            action = result.data[0]

            # Update plan status to failed if any action is rejected
            await self._update_plan_status(action["plan_id"], PlanStatus.FAILED, org_id)

            return PendingActionResponse(**action)

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to reject action: {e}")
            raise

    async def execute_action(
        self,
        action_id: str,
        result: Dict[str, Any],
        status: ActionStatus,
        org_id: str,
        execution_time_ms: int,
        error_message: Optional[str] = None
    ) -> ExecutedActionResponse:
        """
        Record action execution result.

        Args:
            action_id: Pending action ID
            result: Execution result
            status: Execution status
            org_id: Organization ID
            execution_time_ms: Execution time in milliseconds
            error_message: Optional error message

        Returns:
            Executed action record
        """
        try:
            logger.info(f"[ACTION_PLANNING] Recording execution for action {action_id}: {status.value}")

            # Get pending action details
            pending = self.supabase.table("ai_actions_pending")\
                .select("*")\
                .eq("id", action_id)\
                .eq("org_id", org_id)\
                .single()\
                .execute()

            if not pending.data:
                raise Exception("Pending action not found")

            pending_action = pending.data

            # Create executed action record
            executed_data = {
                "org_id": org_id,
                "plan_id": pending_action["plan_id"],
                "pending_action_id": action_id,
                "agent_type": pending_action.get("agent_type", "unknown"),
                "action_type": pending_action["action_type"],
                "action_name": pending_action["action_name"],
                "description": pending_action["description"],
                "target_integration": pending_action["target_integration"],
                "target_resource": pending_action["target_resource"],
                "parameters": pending_action["parameters"],
                "status": status.value,
                "result": result,
                "error_message": error_message,
                "started_at": (datetime.utcnow() - timedelta(milliseconds=execution_time_ms)).isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "execution_time_ms": execution_time_ms
            }

            result = self.supabase.table("ai_actions_executed").insert(executed_data).execute()

            if not result.data:
                raise Exception("Failed to record executed action")

            # Update pending action status
            self.supabase.table("ai_actions_pending")\
                .delete()\
                .eq("id", action_id)\
                .execute()

            # Update plan progress
            await self._update_plan_progress(pending_action["plan_id"], org_id)

            return ExecutedActionResponse(**result.data[0])

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to record execution: {e}")
            raise

    async def get_plan(self, plan_id: str, org_id: str) -> ActionPlanResponse:
        """
        Get an action plan by ID.

        Args:
            plan_id: Plan ID
            org_id: Organization ID

        Returns:
            Action plan
        """
        try:
            result = self.supabase.table("ai_action_plans")\
                .select("*")\
                .eq("id", plan_id)\
                .eq("org_id", org_id)\
                .single()\
                .execute()

            if not result.data:
                raise Exception("Plan not found")

            return ActionPlanResponse(**result.data)

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to get plan: {e}")
            raise

    async def list_plans(
        self,
        org_id: str,
        agent_type: Optional[str] = None,
        status: Optional[PlanStatus] = None,
        limit: int = 50
    ) -> List[ActionPlanResponse]:
        """
        List action plans for an organization.

        Args:
            org_id: Organization ID
            agent_type: Optional agent type filter
            status: Optional status filter
            limit: Maximum number of plans to return

        Returns:
            List of action plans
        """
        try:
            query = self.supabase.table("ai_action_plans")\
                .select("*")\
                .eq("org_id", org_id)

            if agent_type:
                query = query.eq("agent_type", agent_type)

            if status:
                query = query.eq("status", status.value)

            result = query.order("created_at", desc=True).limit(limit).execute()

            return [ActionPlanResponse(**plan) for plan in result.data]

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to list plans: {e}")
            raise

    async def cancel_plan(self, plan_id: str, org_id: str) -> ActionPlanResponse:
        """
        Cancel an action plan and its pending actions.

        Args:
            plan_id: Plan ID
            org_id: Organization ID

        Returns:
            Updated plan
        """
        try:
            logger.info(f"[ACTION_PLANNING] Cancelling plan {plan_id}")

            # Update plan status
            result = self.supabase.table("ai_action_plans")\
                .update({"status": PlanStatus.CANCELLED.value})\
                .eq("id", plan_id)\
                .eq("org_id", org_id)\
                .execute()

            if not result.data:
                raise Exception("Plan not found")

            # Delete pending actions
            self.supabase.table("ai_actions_pending")\
                .delete()\
                .eq("plan_id", plan_id)\
                .eq("org_id", org_id)\
                .execute()

            return ActionPlanResponse(**result.data[0])

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to cancel plan: {e}")
            raise

    async def _check_plan_approval_status(self, plan_id: str, org_id: str) -> None:
        """
        Check if all actions in a plan are approved and update plan status.

        Args:
            plan_id: Plan ID
            org_id: Organization ID
        """
        try:
            # Check for any pending actions requiring approval
            pending = self.supabase.table("ai_actions_pending")\
                .select("id", count="exact")\
                .eq("plan_id", plan_id)\
                .eq("org_id", org_id)\
                .eq("requires_approval", True)\
                .eq("status", "pending")\
                .execute()

            if pending.count == 0:
                # All actions approved, update plan
                self.supabase.table("ai_action_plans")\
                    .update({
                        "status": PlanStatus.APPROVED.value,
                        "approval_status": "approved",
                        "approved_at": datetime.utcnow().isoformat()
                    })\
                    .eq("id", plan_id)\
                    .eq("org_id", org_id)\
                    .execute()

                logger.info(f"[ACTION_PLANNING] Plan {plan_id} fully approved")

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to check approval status: {e}")

    async def _update_plan_progress(self, plan_id: str, org_id: str) -> None:
        """
        Update plan progress based on executed actions.

        Args:
            plan_id: Plan ID
            org_id: Organization ID
        """
        try:
            # Count executed actions
            executed = self.supabase.table("ai_actions_executed")\
                .select("id", count="exact")\
                .eq("plan_id", plan_id)\
                .eq("org_id", org_id)\
                .execute()

            completed_steps = executed.count if executed.count else 0

            # Get plan details
            plan = self.supabase.table("ai_action_plans")\
                .select("total_steps, status")\
                .eq("id", plan_id)\
                .eq("org_id", org_id)\
                .single()\
                .execute()

            if plan.data:
                total_steps = plan.data["total_steps"]

                # Update completed steps
                update_data = {"completed_steps": completed_steps}

                # If all steps completed, mark plan as completed
                if completed_steps >= total_steps:
                    update_data["status"] = PlanStatus.COMPLETED.value
                    update_data["completed_at"] = datetime.utcnow().isoformat()
                elif plan.data["status"] == PlanStatus.APPROVED.value:
                    update_data["status"] = PlanStatus.EXECUTING.value
                    update_data["started_at"] = datetime.utcnow().isoformat()

                self.supabase.table("ai_action_plans")\
                    .update(update_data)\
                    .eq("id", plan_id)\
                    .eq("org_id", org_id)\
                    .execute()

                logger.info(f"[ACTION_PLANNING] Plan {plan_id} progress: {completed_steps}/{total_steps}")

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to update plan progress: {e}")

    async def _update_plan_status(self, plan_id: str, status: PlanStatus, org_id: str) -> None:
        """
        Update plan status.

        Args:
            plan_id: Plan ID
            status: New status
            org_id: Organization ID
        """
        try:
            self.supabase.table("ai_action_plans")\
                .update({"status": status.value})\
                .eq("id", plan_id)\
                .eq("org_id", org_id)\
                .execute()

            logger.info(f"[ACTION_PLANNING] Updated plan {plan_id} status to {status.value}")

        except Exception as e:
            logger.error(f"[ACTION_PLANNING] Failed to update plan status: {e}")


# Singleton instance
_action_planning_service = None


def get_action_planning_service(supabase: Client) -> ActionPlanningService:
    """Get or create ActionPlanningService instance."""
    global _action_planning_service
    if _action_planning_service is None:
        _action_planning_service = ActionPlanningService(supabase)
    return _action_planning_service