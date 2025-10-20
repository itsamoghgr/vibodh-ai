"""
Agent Framework - Phase 4
Autonomous decision agents for business operations
"""

from .base_agent import BaseAgent, AgentCapability, AgentLifecycleState
from .marketing_agent import MarketingAgent
from .communication_agent import CommunicationAgent

__all__ = [
    "BaseAgent",
    "AgentCapability",
    "AgentLifecycleState",
    "MarketingAgent",
    "CommunicationAgent"
]