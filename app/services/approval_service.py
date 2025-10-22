"""
Approval Service - Phase 4, Human-in-the-Loop

Manages AI action approvals with automatic timeout mechanism.

Features:
- Approval workflow management
- Auto-timeout for pending approvals
- Approval analytics and tracking
- Notification integration
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from app.core.logging import logger, log_error


class ApprovalStatus(str, Enum):
    """Approval status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


class ApprovalService:
    """
    Service for managing AI action approvals.

    Handles:
    - Approval requests and decisions
    - Automatic timeout and expiration
    - Approval analytics
    - Notification triggers
    """

    def __init__(self, supabase):
        """
        Initialize approval service.

        Args:
            supabase: Supabase client
        """
        self.supabase = supabase
        logger.info("[APPROVAL_SERVICE] Initialized")

    async def create_approval_request(
        self,
        org_id: str,
        plan_id: str,
        action: Dict[str, Any],
        risk_level: str,
        timeout_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Create an approval request for an action.

        Args:
            org_id: Organization ID
            plan_id: Plan ID
            action: Action details
            risk_level: Risk level (low/medium/high/critical)
            timeout_hours: Hours until auto-timeout

        Returns:
            Created approval request
        """
        try:
            import uuid

            approval_id = str(uuid.uuid4())
            auto_approve_timeout = datetime.utcnow() + timedelta(hours=timeout_hours)

            approval_data = {
                "id": approval_id,
                "org_id": org_id,
                "plan_id": plan_id,
                "step_index": action.get("step_index", 0),
                "action_type": action.get("action_type"),
                "action_name": action.get("action_name"),
                "description": action.get("description"),
                "target_integration": action.get("target_integration"),
                "target_resource": action.get("target_resource", {}),
                "parameters": action.get("parameters", {}),
                "risk_level": risk_level,
                "requires_approval": True,
                "auto_approve_timeout": auto_approve_timeout.isoformat(),
                "status": ApprovalStatus.PENDING,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(hours=timeout_hours)).isoformat()
            }

            result = self.supabase.table("ai_actions_pending")\
                .insert(approval_data)\
                .execute()

            logger.info(
                f"[APPROVAL_SERVICE] Created approval request",
                extra={
                    "approval_id": approval_id,
                    "plan_id": plan_id,
                    "risk_level": risk_level,
                    "timeout_hours": timeout_hours
                }
            )

            return result.data[0] if result.data else approval_data

        except Exception as e:
            log_error(e, context="ApprovalService.create_approval_request")
            raise

    async def approve_action(
        self,
        approval_id: str,
        user_id: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Approve a pending action.

        Args:
            approval_id: Approval ID
            user_id: User ID who approved
            notes: Optional approval notes

        Returns:
            Updated approval record
        """
        try:
            update_data = {
                "status": ApprovalStatus.APPROVED,
                "approved_by": user_id,
                "approved_at": datetime.utcnow().isoformat(),
                "notes": notes
            }

            result = self.supabase.table("ai_actions_pending")\
                .update(update_data)\
                .eq("id", approval_id)\
                .eq("status", ApprovalStatus.PENDING)\
                .execute()

            if not result.data:
                raise ValueError(f"Approval {approval_id} not found or already processed")

            logger.info(
                f"[APPROVAL_SERVICE] Action approved",
                extra={
                    "approval_id": approval_id,
                    "user_id": user_id
                }
            )

            return result.data[0]

        except Exception as e:
            log_error(e, context="ApprovalService.approve_action")
            raise

    async def reject_action(
        self,
        approval_id: str,
        user_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Reject a pending action.

        Args:
            approval_id: Approval ID
            user_id: User ID who rejected
            reason: Rejection reason

        Returns:
            Updated approval record
        """
        try:
            update_data = {
                "status": ApprovalStatus.REJECTED,
                "approved_by": user_id,
                "approved_at": datetime.utcnow().isoformat(),
                "rejection_reason": reason
            }

            result = self.supabase.table("ai_actions_pending")\
                .update(update_data)\
                .eq("id", approval_id)\
                .eq("status", ApprovalStatus.PENDING)\
                .execute()

            if not result.data:
                raise ValueError(f"Approval {approval_id} not found or already processed")

            logger.info(
                f"[APPROVAL_SERVICE] Action rejected",
                extra={
                    "approval_id": approval_id,
                    "user_id": user_id,
                    "reason": reason
                }
            )

            return result.data[0]

        except Exception as e:
            log_error(e, context="ApprovalService.reject_action")
            raise

    async def process_expired_approvals(
        self,
        org_id: str,
        auto_approve: bool = False
    ) -> Dict[str, Any]:
        """
        Process approvals that have passed their timeout.

        Args:
            org_id: Organization ID
            auto_approve: If True, auto-approve expired actions; otherwise mark as expired

        Returns:
            Processing results
        """
        try:
            current_time = datetime.utcnow().isoformat()

            # Find expired pending approvals
            result = self.supabase.table("ai_actions_pending")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("status", ApprovalStatus.PENDING)\
                .lte("auto_approve_timeout", current_time)\
                .execute()

            expired_approvals = result.data if result.data else []

            if not expired_approvals:
                logger.info(
                    f"[APPROVAL_SERVICE] No expired approvals found",
                    extra={"org_id": org_id}
                )
                return {
                    "processed_count": 0,
                    "auto_approved_count": 0,
                    "expired_count": 0
                }

            logger.info(
                f"[APPROVAL_SERVICE] Processing {len(expired_approvals)} expired approvals",
                extra={
                    "org_id": org_id,
                    "auto_approve": auto_approve
                }
            )

            auto_approved_count = 0
            expired_count = 0

            for approval in expired_approvals:
                approval_id = approval["id"]
                risk_level = approval.get("risk_level", "medium")

                # Auto-approve low-risk actions only
                if auto_approve and risk_level == "low":
                    self.supabase.table("ai_actions_pending")\
                        .update({
                            "status": ApprovalStatus.AUTO_APPROVED,
                            "approved_at": datetime.utcnow().isoformat(),
                            "notes": "Auto-approved after timeout (low risk)"
                        })\
                        .eq("id", approval_id)\
                        .execute()

                    auto_approved_count += 1

                    logger.info(
                        f"[APPROVAL_SERVICE] Auto-approved low-risk action",
                        extra={"approval_id": approval_id}
                    )

                else:
                    # Mark as expired
                    self.supabase.table("ai_actions_pending")\
                        .update({
                            "status": ApprovalStatus.EXPIRED,
                            "notes": f"Expired after timeout ({risk_level} risk)"
                        })\
                        .eq("id", approval_id)\
                        .execute()

                    expired_count += 1

                    logger.info(
                        f"[APPROVAL_SERVICE] Marked action as expired",
                        extra={"approval_id": approval_id, "risk_level": risk_level}
                    )

            return {
                "processed_count": len(expired_approvals),
                "auto_approved_count": auto_approved_count,
                "expired_count": expired_count
            }

        except Exception as e:
            log_error(e, context="ApprovalService.process_expired_approvals")
            return {
                "processed_count": 0,
                "auto_approved_count": 0,
                "expired_count": 0,
                "error": str(e)
            }

    async def get_pending_approvals(
        self,
        org_id: str,
        risk_level: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get pending approvals for an organization.

        Args:
            org_id: Organization ID
            risk_level: Optional filter by risk level
            limit: Maximum number of results

        Returns:
            List of pending approvals
        """
        try:
            query = self.supabase.table("ai_actions_pending")\
                .select("*")\
                .eq("org_id", org_id)\
                .eq("status", ApprovalStatus.PENDING)\
                .order("created_at", desc=False)\
                .limit(limit)

            if risk_level:
                query = query.eq("risk_level", risk_level)

            result = query.execute()

            return result.data if result.data else []

        except Exception as e:
            log_error(e, context="ApprovalService.get_pending_approvals")
            return []

    async def get_approval_stats(
        self,
        org_id: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get approval statistics for an organization.

        Args:
            org_id: Organization ID
            days_back: Number of days to look back

        Returns:
            Approval statistics
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

            # Get all approvals within timeframe
            result = self.supabase.table("ai_actions_pending")\
                .select("status, risk_level, created_at, approved_at")\
                .eq("org_id", org_id)\
                .gte("created_at", cutoff_date)\
                .execute()

            approvals = result.data if result.data else []

            # Calculate statistics
            stats = {
                "total": len(approvals),
                "pending": 0,
                "approved": 0,
                "rejected": 0,
                "expired": 0,
                "auto_approved": 0,
                "by_risk_level": {
                    "low": 0,
                    "medium": 0,
                    "high": 0,
                    "critical": 0
                },
                "avg_approval_time_minutes": None
            }

            approval_times = []

            for approval in approvals:
                status = approval.get("status")
                risk = approval.get("risk_level", "medium")

                # Count by status
                if status == ApprovalStatus.PENDING:
                    stats["pending"] += 1
                elif status == ApprovalStatus.APPROVED:
                    stats["approved"] += 1
                elif status == ApprovalStatus.REJECTED:
                    stats["rejected"] += 1
                elif status == ApprovalStatus.EXPIRED:
                    stats["expired"] += 1
                elif status == ApprovalStatus.AUTO_APPROVED:
                    stats["auto_approved"] += 1

                # Count by risk level
                if risk in stats["by_risk_level"]:
                    stats["by_risk_level"][risk] += 1

                # Calculate approval time
                if approval.get("created_at") and approval.get("approved_at"):
                    created = datetime.fromisoformat(approval["created_at"].replace("Z", "+00:00"))
                    approved = datetime.fromisoformat(approval["approved_at"].replace("Z", "+00:00"))
                    approval_times.append((approved - created).total_seconds() / 60)

            if approval_times:
                stats["avg_approval_time_minutes"] = sum(approval_times) / len(approval_times)

            return stats

        except Exception as e:
            log_error(e, context="ApprovalService.get_approval_stats")
            return {
                "total": 0,
                "pending": 0,
                "approved": 0,
                "rejected": 0,
                "expired": 0,
                "auto_approved": 0
            }


# Global instance (lazy initialization)
_approval_service_instance = None


def get_approval_service(supabase) -> ApprovalService:
    """
    Get or create the global ApprovalService instance.

    Args:
        supabase: Supabase client

    Returns:
        ApprovalService instance
    """
    global _approval_service_instance

    if _approval_service_instance is None:
        _approval_service_instance = ApprovalService(supabase)

    return _approval_service_instance
