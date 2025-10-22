"""
CIL Policy Service
Manages versioned policy configurations and their lifecycle
"""

from typing import Dict, Optional, Any, List
from datetime import datetime
from uuid import UUID
import json

from app.core.logging import logger
from app.db import get_supabase_admin_client


class CILPolicyService:
    """
    Manages CIL policies including:
    - Creating and versioning policies
    - Activating/deactivating policies
    - Comparing policy versions
    - Default policy generation
    """

    def __init__(self):
        self.supabase = get_supabase_admin_client()

    def get_default_policy(self) -> Dict[str, Any]:
        """
        Generate default policy configuration for new organizations
        """
        return {
            "confidence_thresholds": {
                "question": 0.7,
                "execute": 0.75,
                "task": 0.7,
                "summary": 0.65,
                "insight": 0.7,
                "risk": 0.85
            },
            "module_weights": {
                "question": {
                    "rag": 0.8,
                    "kg": 0.5,
                    "memory": 0.6,
                    "insight": 0.4
                },
                "execute": {
                    "rag": 0.4,
                    "kg": 0.3,
                    "memory": 0.5,
                    "insight": 0.6
                },
                "risk": {
                    "rag": 0.5,
                    "kg": 0.9,
                    "memory": 0.6,
                    "insight": 0.9
                },
                "summary": {
                    "rag": 0.7,
                    "kg": 0.6,
                    "memory": 0.8,
                    "insight": 0.7
                }
            },
            "risk_sensitivity": {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.85,
                "critical": 0.95
            },
            "agent_preferences": {
                "communication": {
                    "priority": 1,
                    "max_retries": 3,
                    "timeout_seconds": 300
                },
                "marketing": {
                    "priority": 2,
                    "max_retries": 2,
                    "timeout_seconds": 300
                },
                "research": {
                    "priority": 3,
                    "max_retries": 2,
                    "timeout_seconds": 600
                }
            },
            "approval_timeouts": {
                "low": 3600,      # 1 hour
                "medium": 1800,   # 30 minutes
                "high": 900,      # 15 minutes
                "critical": 300   # 5 minutes
            },
            "safety_guardrails": {
                "min_confidence_threshold": 0.4,
                "max_risk_sensitivity": 0.98,
                "max_approval_timeout": 7200
            }
        }

    def create_policy(
        self,
        org_id: str,
        policy_config: Dict[str, Any],
        learning_cycle_id: Optional[str] = None,
        change_reason: Optional[str] = None,
        change_summary: Optional[Dict] = None,
        approval_required: bool = False,
        created_by: str = "cil_system"
    ) -> Optional[Dict]:
        """
        Create a new policy version

        Args:
            org_id: Organization ID
            policy_config: Policy configuration dictionary
            learning_cycle_id: ID of learning cycle that generated this
            change_reason: Why this policy was created
            change_summary: What changed from previous version
            approval_required: Whether this needs human approval
            created_by: Who/what created this policy

        Returns:
            Created policy record or None on error
        """
        try:
            # Get current policy version
            current_version = self._get_latest_version(org_id)
            new_version = current_version + 1

            # If approval not required, deactivate old policy
            is_active = not approval_required

            if is_active:
                self._deactivate_current_policy(org_id)

            # Create new policy
            policy_data = {
                'org_id': org_id,
                'version': new_version,
                'is_active': is_active,
                'policy_config': json.dumps(policy_config),
                'learning_cycle_id': learning_cycle_id,
                'change_reason': change_reason,
                'change_summary': json.dumps(change_summary) if change_summary else None,
                'approval_required': approval_required,
                'created_by': created_by,
                'activated_at': datetime.utcnow().isoformat() if is_active else None
            }

            result = self.supabase.table('cil_policies').insert(policy_data).execute()

            if result.data:
                logger.info(
                    f"Created CIL policy v{new_version} for org {org_id}",
                    extra={
                        'org_id': org_id,
                        'version': new_version,
                        'approval_required': approval_required
                    }
                )
                return result.data[0]

            return None

        except Exception as e:
            logger.error(f"Error creating CIL policy: {e}", exc_info=True)
            return None

    def get_active_policy(self, org_id: str) -> Optional[Dict]:
        """
        Get the currently active policy for an organization
        """
        try:
            result = self.supabase.table('cil_policies')\
                .select('*')\
                .eq('org_id', org_id)\
                .eq('is_active', True)\
                .single()\
                .execute()

            if result.data:
                # Parse JSONB fields
                policy = result.data
                policy['policy_config'] = json.loads(policy['policy_config']) if isinstance(policy['policy_config'], str) else policy['policy_config']
                if policy.get('change_summary'):
                    policy['change_summary'] = json.loads(policy['change_summary']) if isinstance(policy['change_summary'], str) else policy['change_summary']
                return policy

            # No active policy, create default
            logger.info(f"No active policy for org {org_id}, creating default")
            return self._create_default_policy(org_id)

        except Exception as e:
            logger.error(f"Error getting active policy: {e}", exc_info=True)
            return None

    def get_policy_by_version(self, org_id: str, version: int) -> Optional[Dict]:
        """Get specific policy version"""
        try:
            result = self.supabase.table('cil_policies')\
                .select('*')\
                .eq('org_id', org_id)\
                .eq('version', version)\
                .single()\
                .execute()

            if result.data:
                policy = result.data
                policy['policy_config'] = json.loads(policy['policy_config']) if isinstance(policy['policy_config'], str) else policy['policy_config']
                if policy.get('change_summary'):
                    policy['change_summary'] = json.loads(policy['change_summary']) if isinstance(policy['change_summary'], str) else policy['change_summary']
                return policy

            return None

        except Exception as e:
            logger.error(f"Error getting policy version {version}: {e}", exc_info=True)
            return None

    def get_policy_history(self, org_id: str, limit: int = 10) -> List[Dict]:
        """Get policy version history"""
        try:
            result = self.supabase.table('cil_policies')\
                .select('*')\
                .eq('org_id', org_id)\
                .order('version', desc=True)\
                .limit(limit)\
                .execute()

            policies = result.data or []
            for policy in policies:
                policy['policy_config'] = json.loads(policy['policy_config']) if isinstance(policy['policy_config'], str) else policy['policy_config']
                if policy.get('change_summary'):
                    policy['change_summary'] = json.loads(policy['change_summary']) if isinstance(policy['change_summary'], str) else policy['change_summary']

            return policies

        except Exception as e:
            logger.error(f"Error getting policy history: {e}", exc_info=True)
            return []

    def activate_policy(self, policy_id: str, activated_by: Optional[str] = None) -> bool:
        """
        Activate a policy (typically after approval)

        Args:
            policy_id: Policy ID to activate
            activated_by: User ID who approved

        Returns:
            True if successful
        """
        try:
            # Get the policy to activate
            policy_query = self.supabase.table('cil_policies')\
                .select('*')\
                .eq('id', policy_id)\
                .single()\
                .execute()

            if not policy_query.data:
                logger.error(f"Policy {policy_id} not found")
                return False

            policy = policy_query.data
            org_id = policy['org_id']

            # Deactivate current policy
            self._deactivate_current_policy(org_id)

            # Activate new policy
            update_data = {
                'is_active': True,
                'activated_at': datetime.utcnow().isoformat(),
                'approved_by': activated_by,
                'approved_at': datetime.utcnow().isoformat()
            }

            self.supabase.table('cil_policies')\
                .update(update_data)\
                .eq('id', policy_id)\
                .execute()

            logger.info(f"Activated CIL policy {policy_id} for org {org_id}")
            return True

        except Exception as e:
            logger.error(f"Error activating policy: {e}", exc_info=True)
            return False

    def compare_policies(self, org_id: str, version_a: int, version_b: int) -> Dict[str, Any]:
        """
        Compare two policy versions and show differences

        Returns:
            Dictionary with differences
        """
        try:
            policy_a = self.get_policy_by_version(org_id, version_a)
            policy_b = self.get_policy_by_version(org_id, version_b)

            if not policy_a or not policy_b:
                return {'error': 'One or both policy versions not found'}

            config_a = policy_a['policy_config']
            config_b = policy_b['policy_config']

            differences = self._find_differences(config_a, config_b)

            return {
                'version_a': version_a,
                'version_b': version_b,
                'differences': differences,
                'total_changes': len(differences)
            }

        except Exception as e:
            logger.error(f"Error comparing policies: {e}", exc_info=True)
            return {'error': str(e)}

    def _get_latest_version(self, org_id: str) -> int:
        """Get the latest policy version number for an org"""
        try:
            result = self.supabase.table('cil_policies')\
                .select('version')\
                .eq('org_id', org_id)\
                .order('version', desc=True)\
                .limit(1)\
                .execute()

            if result.data and len(result.data) > 0:
                return result.data[0]['version']

            return 0

        except Exception:
            return 0

    def _deactivate_current_policy(self, org_id: str):
        """Deactivate the currently active policy"""
        try:
            self.supabase.table('cil_policies')\
                .update({
                    'is_active': False,
                    'deactivated_at': datetime.utcnow().isoformat()
                })\
                .eq('org_id', org_id)\
                .eq('is_active', True)\
                .execute()

        except Exception as e:
            logger.error(f"Error deactivating policy: {e}", exc_info=True)

    def _create_default_policy(self, org_id: str) -> Optional[Dict]:
        """Create and activate default policy for new org"""
        default_config = self.get_default_policy()

        return self.create_policy(
            org_id=org_id,
            policy_config=default_config,
            change_reason="Initial default policy",
            approval_required=False,
            created_by="system"
        )

    def _find_differences(self, config_a: Dict, config_b: Dict, path: str = "") -> List[Dict]:
        """Recursively find differences between two configs"""
        differences = []

        all_keys = set(config_a.keys()) | set(config_b.keys())

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            if key not in config_a:
                differences.append({
                    'path': current_path,
                    'change_type': 'added',
                    'new_value': config_b[key]
                })
            elif key not in config_b:
                differences.append({
                    'path': current_path,
                    'change_type': 'removed',
                    'old_value': config_a[key]
                })
            elif isinstance(config_a[key], dict) and isinstance(config_b[key], dict):
                # Recurse into nested dictionaries
                differences.extend(self._find_differences(config_a[key], config_b[key], current_path))
            elif config_a[key] != config_b[key]:
                differences.append({
                    'path': current_path,
                    'change_type': 'modified',
                    'old_value': config_a[key],
                    'new_value': config_b[key]
                })

        return differences


# Singleton instance
_cil_policy_service: Optional[CILPolicyService] = None


def get_cil_policy_service() -> CILPolicyService:
    """Get singleton CIL policy service instance"""
    global _cil_policy_service
    if _cil_policy_service is None:
        _cil_policy_service = CILPolicyService()
    return _cil_policy_service
