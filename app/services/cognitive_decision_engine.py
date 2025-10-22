"""
Cognitive Decision Engine - Phase 4 Evolution
Pure LLM-driven reasoning and decision-making core

Replaces keyword-based heuristics with intelligent LLM reasoning.
Uses chain-of-thought prompting for transparent, explainable decisions.
"""

import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from supabase import Client
from app.core.config import settings
from app.core.logging import logger
import requests


class ReasoningChain(BaseModel):
    """Chain-of-thought reasoning breakdown"""
    chain_of_thought: List[str] = Field(..., description="Step-by-step reasoning process")
    key_indicators: List[str] = Field(..., description="Key signals that influenced the decision")
    decision_factors: List[str] = Field(..., description="Factors considered in final decision")


class RiskAssessment(BaseModel):
    """Risk evaluation for the query"""
    level: str = Field(..., description="Risk level: low, medium, high, critical")
    factors: List[str] = Field(default_factory=list, description="Risk factors identified")


class DecisionMetadata(BaseModel):
    """Processing metadata"""
    processing_time_ms: int = Field(..., description="Time taken to make decision")
    model: str = Field(..., description="LLM model used")
    temperature: float = Field(..., description="LLM temperature used")
    timestamp: str = Field(..., description="Decision timestamp")


class CognitiveDecision(BaseModel):
    """Complete cognitive decision output"""
    intent: str = Field(..., description="Classified intent: execute, question, task, summary, insight, risk")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    reasoning: ReasoningChain = Field(..., description="Detailed reasoning breakdown")
    recommended_modules: List[str] = Field(..., description="AI modules to activate: rag, kg, memory, insight")
    recommended_agent: Optional[str] = Field(None, description="Agent to delegate to: communication, none")
    risk_assessment: RiskAssessment = Field(..., description="Risk evaluation")
    requires_human_review: bool = Field(..., description="Flag for low-confidence decisions")
    metadata: Optional[DecisionMetadata] = Field(None, description="Processing metadata (added after LLM response)")


class CognitiveDecisionEngine:
    """
    Pure LLM-driven cognitive engine for intent classification and decision-making.

    Uses Groq Llama 3.3 70B as the primary reasoning core with chain-of-thought
    prompting for transparent, explainable decision-making.
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.groq_api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL
        self.confidence_threshold = 0.7  # Below this triggers human review

    async def make_decision(
        self,
        query: str,
        org_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CognitiveDecision:
        """
        Make a cognitive decision about how to handle a query.

        Args:
            query: User query to analyze
            org_id: Organization ID
            user_id: Optional user ID
            context: Additional context (memories, recent activity, etc.)

        Returns:
            CognitiveDecision with full reasoning chain and recommendations
        """
        start_time = time.time()

        logger.info(f"[COGNITIVE_ENGINE] Processing query: {query[:80]}...")

        try:
            # Build comprehensive system prompt
            system_prompt = self._build_system_prompt()

            # Build user prompt with context
            user_prompt = self._build_user_prompt(query, context or {})

            # Call LLM for cognitive reasoning
            decision_json = await self._call_llm(system_prompt, user_prompt)

            # Parse and validate decision
            decision = self._parse_decision(decision_json)

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Add metadata
            decision.metadata = DecisionMetadata(
                processing_time_ms=processing_time_ms,
                model=self.model,
                temperature=0.3,
                timestamp=datetime.utcnow().isoformat()
            )

            # Flag low-confidence decisions for human review
            if decision.confidence < self.confidence_threshold:
                decision.requires_human_review = True
                logger.warning(
                    f"[COGNITIVE_ENGINE] Low confidence ({decision.confidence:.2f}) - "
                    f"flagging for human review"
                )

            # Log decision to database
            await self._log_decision(query, decision, org_id, user_id)

            logger.info(
                f"[COGNITIVE_ENGINE] Decision: intent={decision.intent}, "
                f"confidence={decision.confidence:.2f}, "
                f"modules={decision.recommended_modules}, "
                f"agent={decision.recommended_agent}"
            )

            return decision

        except Exception as e:
            logger.error(f"[COGNITIVE_ENGINE] Decision failed: {e}")
            # Return safe fallback decision
            return self._fallback_decision(query, str(e))

    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt with Vibodh AI context."""

        return """You are the Cognitive Decision Engine for Vibodh AI, an intelligent enterprise knowledge platform.

Your role is to analyze user queries and make intelligent decisions about how to handle them.

## Vibodh AI Capabilities

**Available AI Modules:**
- RAG (Retrieval-Augmented Generation): Semantic search across documents, Slack messages, ClickUp tasks
- KG (Knowledge Graph): Entity relationships, org structure, people connections, channel mappings
- Memory: Conversational history, past decisions, user preferences
- Insight: Analytics, trends, patterns, activity summaries

**Available Agents:**
- Communication Agent: Sends messages to Slack, emails, creates tasks, schedules insights
- (More agents coming soon)

## Intent Categories

1. **execute**: User wants to take action or automate something
   - Examples: "send hello to #channel", "create a task", "schedule a reminder"
   - Requires: Agent delegation (usually Communication Agent)

2. **question**: User asking for information
   - Examples: "what's the update from the team?", "who joined recently?"
   - Requires: RAG + potentially KG/Memory/Insight for context

3. **task**: User asking how to do something (not doing it now)
   - Examples: "how do I create a task?", "what's the process for..."
   - Requires: RAG for documentation

4. **summary**: User wants overview or recap
   - Examples: "summarize recent activity", "give me an overview"
   - Requires: Insight + RAG

5. **insight**: User wants analysis or trends
   - Examples: "what patterns do you see?", "show me trends"
   - Requires: Insight + KG

6. **risk**: User asking about risks or concerns
   - Examples: "any security concerns?", "what are the risks?"
   - Requires: Insight + RAG

## Your Task

For each query, you must:
1. **Think step-by-step** (chain-of-thought reasoning)
2. **Identify key indicators** (words, phrases, patterns)
3. **Consider decision factors** (context, user intent, urgency)
4. **Classify intent** (execute, question, task, summary, insight, risk)
5. **Assess confidence** (0.0-1.0, based on clarity of intent)
6. **Recommend modules** (which AI modules should activate)
7. **Recommend agent** (if execution needed, which agent)
8. **Assess risk** (low, medium, high, critical)

## Output Format

Return ONLY a JSON object (no markdown, no explanation):

```json
{
  "intent": "execute|question|task|summary|insight|risk",
  "confidence": 0.0-1.0,
  "reasoning": {
    "chain_of_thought": [
      "Step 1: Observe that query contains action verb 'send'",
      "Step 2: Identify target channel '#channel'",
      "Step 3: Classify as execution request",
      "Step 4: Recommend Communication Agent"
    ],
    "key_indicators": ["action verb: send", "channel mention: #channel"],
    "decision_factors": ["clear action intent", "specific target", "no ambiguity"]
  },
  "recommended_modules": ["rag", "kg"],
  "recommended_agent": "communication",
  "risk_assessment": {
    "level": "low|medium|high|critical",
    "factors": ["public channel posting", "no sensitive data"]
  },
  "requires_human_review": false
}
```

## Guidelines

- **Be transparent**: Show your reasoning clearly in chain_of_thought
- **Be decisive**: Don't hedge - make a clear classification
- **Be context-aware**: Consider the full query, not just keywords
- **Be confident**: High confidence (>0.8) for clear queries, lower for ambiguous ones
- **Flag uncertainty**: Set requires_human_review=true for confidence <0.7
- **Think holistically**: Consider intent, urgency, risk, and context together

Now process each query with full cognitive reasoning."""

    def _build_user_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build user prompt with query and context."""

        prompt_parts = [f"Query: \"{query}\""]

        # Add context if available
        if context.get("memories"):
            prompt_parts.append(f"\nRecent context: {len(context['memories'])} relevant memories")

        if context.get("recent_activity"):
            prompt_parts.append(f"Recent activity: {context['recent_activity']}")

        prompt_parts.append("\nAnalyze this query and provide your cognitive decision as JSON.")

        return "\n".join(prompt_parts)

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Groq LLM for cognitive reasoning."""

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,  # Lower for more consistent reasoning
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"}  # Force JSON output
                },
                timeout=10
            )

            if response.status_code != 200:
                raise Exception(f"LLM API error: {response.status_code} - {response.text}")

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON from response
            decision_json = json.loads(content)

            return decision_json

        except requests.Timeout:
            raise Exception("LLM request timeout after 10s")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse LLM JSON output: {e}")
        except Exception as e:
            raise Exception(f"LLM call failed: {e}")

    def _parse_decision(self, decision_json: Dict[str, Any]) -> CognitiveDecision:
        """Parse and validate LLM decision output."""

        try:
            # Validate using Pydantic model
            decision = CognitiveDecision(**decision_json)

            # Additional validation
            valid_intents = ["execute", "question", "task", "summary", "insight", "risk"]
            if decision.intent not in valid_intents:
                raise ValueError(f"Invalid intent: {decision.intent}")

            valid_modules = ["rag", "kg", "memory", "insight"]
            for module in decision.recommended_modules:
                if module not in valid_modules:
                    raise ValueError(f"Invalid module: {module}")

            valid_agents = ["communication", None, "none"]
            if decision.recommended_agent and decision.recommended_agent not in valid_agents:
                raise ValueError(f"Invalid agent: {decision.recommended_agent}")

            valid_risk_levels = ["low", "medium", "high", "critical"]
            if decision.risk_assessment.level not in valid_risk_levels:
                raise ValueError(f"Invalid risk level: {decision.risk_assessment.level}")

            # Normalize agent "none" to None
            if decision.recommended_agent == "none":
                decision.recommended_agent = None

            return decision

        except ValidationError as e:
            raise Exception(f"Decision validation failed: {e}")

    def _fallback_decision(self, query: str, error: str) -> CognitiveDecision:
        """Create safe fallback decision when LLM fails."""

        logger.warning(f"[COGNITIVE_ENGINE] Using fallback decision due to: {error}")

        # Simple keyword detection for fallback
        query_lower = query.lower()

        # Detect execution intent
        if any(word in query_lower for word in ["send", "create", "post", "schedule", "execute"]):
            intent = "execute"
            agent = "communication"
        elif "?" in query:
            intent = "question"
            agent = None
        else:
            intent = "question"
            agent = None

        return CognitiveDecision(
            intent=intent,
            confidence=0.3,  # Low confidence for fallback
            reasoning=ReasoningChain(
                chain_of_thought=[
                    "LLM reasoning failed - using fallback heuristics",
                    f"Error: {error[:100]}",
                    f"Detected intent: {intent} based on keywords"
                ],
                key_indicators=["fallback mode"],
                decision_factors=["error recovery"]
            ),
            recommended_modules=["rag"],  # Safe default
            recommended_agent=agent,
            risk_assessment=RiskAssessment(
                level="medium",
                factors=["fallback decision", "lower confidence"]
            ),
            requires_human_review=True,  # Always flag fallback for review
            metadata=DecisionMetadata(
                processing_time_ms=0,
                model=self.model,
                temperature=0.0,
                timestamp=datetime.utcnow().isoformat()
            )
        )

    async def _log_decision(
        self,
        query: str,
        decision: CognitiveDecision,
        org_id: str,
        user_id: Optional[str] = None
    ) -> None:
        """Log cognitive decision to database for analysis."""

        try:
            log_data = {
                "org_id": org_id,
                "user_id": user_id,
                "query": query,
                "intent": decision.intent,
                "confidence": decision.confidence,
                "reasoning_chain": decision.reasoning.chain_of_thought,
                "key_indicators": decision.reasoning.key_indicators,
                "decision_factors": decision.reasoning.decision_factors,
                "recommended_modules": decision.recommended_modules,
                "recommended_agent": decision.recommended_agent,
                "risk_level": decision.risk_assessment.level,
                "risk_factors": decision.risk_assessment.factors,
                "requires_human_review": decision.requires_human_review,
                "model": decision.metadata.model,
                "processing_time_ms": decision.metadata.processing_time_ms,
                "created_at": datetime.utcnow().isoformat()
            }

            # Store in cognitive_decisions table (or reasoning_logs if table doesn't exist)
            # Try cognitive_decisions first, fallback to reasoning_logs
            try:
                self.supabase.table("cognitive_decisions").insert(log_data).execute()
            except Exception:
                # Fallback to reasoning_logs with compatible schema
                fallback_log = {
                    "org_id": org_id,
                    "user_id": user_id,
                    "query": query,
                    "intent": decision.intent,
                    "modules_used": decision.recommended_modules,
                    "reasoning_steps": [{
                        "step": "cognitive_decision",
                        "reasoning": decision.reasoning.dict(),
                        "confidence": decision.confidence
                    }],
                    "execution_time_ms": decision.metadata.processing_time_ms,
                    "response_summary": f"Intent: {decision.intent}, Confidence: {decision.confidence:.2f}",
                    "final_answer": "",  # Not generated yet
                    "context_sources": [],
                    "tokens_used": 0
                }
                self.supabase.table("reasoning_logs").insert(fallback_log).execute()

            logger.info(f"[COGNITIVE_ENGINE] Decision logged to database")

        except Exception as e:
            logger.error(f"[COGNITIVE_ENGINE] Failed to log decision: {e}")
            # Don't fail the request if logging fails


def get_cognitive_decision_engine(supabase: Client) -> CognitiveDecisionEngine:
    """Factory function to get CognitiveDecisionEngine instance."""
    return CognitiveDecisionEngine(supabase)
