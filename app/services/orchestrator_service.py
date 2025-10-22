"""
Orchestrator Service - Cognitive Core
Phase 3, Step 1

Routes queries to appropriate modules (RAG, KG, Insights, Memory)
and combines results using reasoning chains.
"""

import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from supabase import Client
from app.services.rag_service import get_rag_service
from app.services.kg_service import get_kg_service
from app.services.insight_service import get_insight_service
from app.services.memory_service import get_memory_service
from app.services.feedback_service import get_feedback_service
from app.services.adaptive_engine import get_adaptive_engine
from app.services.meta_learning_service import get_meta_learning_service
from app.services.agent_registry import get_agent_registry
from app.services.action_planning_service import get_action_planning_service
from app.services.safety_service import get_safety_service
from app.services.cognitive_decision_engine import get_cognitive_decision_engine
from app.agents.base_agent import ObservationContext
from app.models.agent import TriggerType
from app.core.config import settings
from app.core.logging import logger as app_logger
import requests


class OrchestratorService:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.rag_service = get_rag_service(supabase)
        self.kg_service = get_kg_service(supabase)
        self.insight_service = get_insight_service(supabase)
        self.memory_service = get_memory_service(supabase)
        self.feedback_service = get_feedback_service(supabase)
        self.adaptive_engine = get_adaptive_engine(supabase)
        self.meta_learning_service = get_meta_learning_service(supabase)
        # Agent services (Phase 4)
        self.agent_registry = get_agent_registry(supabase)
        self.action_planning = get_action_planning_service(supabase)
        self.safety_service = get_safety_service(supabase)
        # Cognitive Decision Engine (Phase 4 Evolution)
        self.cognitive_engine = get_cognitive_decision_engine(supabase)
        self.groq_api_key = settings.GROQ_API_KEY

    async def classify_intent(
        self,
        query: str,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Classify user query intent using Cognitive Decision Engine.

        Now powered by pure LLM reasoning with chain-of-thought.
        Stores full decision in self._last_cognitive_decision for access by other methods.

        Returns: question | task | summary | insight | risk | execute
        """
        try:
            # Use Cognitive Decision Engine for LLM-driven reasoning
            decision = await self.cognitive_engine.make_decision(
                query=query,
                org_id=org_id or "default",
                user_id=user_id,
                context=context
            )

            # Store decision for access by route_to_modules() and other methods
            self._last_cognitive_decision = decision

            app_logger.info(
                f"[ORCHESTRATOR] Cognitive decision: intent={decision.intent}, "
                f"confidence={decision.confidence:.2f}, "
                f"modules={decision.recommended_modules}, "
                f"agent={decision.recommended_agent}, "
                f"requires_review={decision.requires_human_review}"
            )

            return decision.intent

        except Exception as e:
            app_logger.error(f"[ORCHESTRATOR] Cognitive engine failed: {e}")
            # Fallback to simple keyword-based classification
            return self._fallback_classify_intent(query)

    def _fallback_classify_intent(self, query: str) -> str:
        """
        Fallback intent classification using simple keyword heuristics.
        Used only when Cognitive Decision Engine fails.
        """
        query_lower = query.lower()

        # Detect action requests
        action_keywords = ["send", "create", "post", "schedule", "execute", "run"]
        if any(word in query_lower for word in action_keywords):
            return "execute"

        # Detect questions
        if "?" in query or query_lower.startswith(("what", "who", "when", "where", "why", "how")):
            return "question"

        # Default to question
        return "question"

    async def _check_conversation_continuation(
        self,
        query: str,
        org_id: str,
        user_id: Optional[str],
        memories: List[Dict[str, Any]]
    ) -> str:
        """
        Check if current query is a continuation of a previous incomplete request.

        Detects patterns like:
        - AI: "What message would you like to send?"
        - User: "product launch date..."

        Returns enriched query combining original context + current response.
        """
        try:
            # Get recent conversation history from database
            recent_messages = self.supabase.table("reasoning_logs")\
                .select("query, final_answer, created_at")\
                .eq("org_id", org_id)\
                .order("created_at", desc=True)\
                .limit(3)\
                .execute()

            if not recent_messages.data or len(recent_messages.data) < 2:
                return query  # Not enough history

            # Get the last AI response
            last_ai_response = recent_messages.data[0].get("final_answer", "")

            # Check if last AI response was a clarifying question
            clarifying_indicators = [
                "what message",
                "which channel",
                "what would you like",
                "could you provide",
                "could you tell me",
                "please specify",
                "what should i",
                "where should i"
            ]

            is_clarifying_question = any(
                indicator in last_ai_response.lower()
                for indicator in clarifying_indicators
            )

            if not is_clarifying_question:
                # Check for retry request
                retry_keywords = ["try again", "retry", "again", "do it", "go ahead"]
                if any(keyword in query.lower() for keyword in retry_keywords):
                    # Find the last user query (2 messages back)
                    if len(recent_messages.data) >= 2:
                        original_query = recent_messages.data[1].get("query", "")
                        app_logger.info(f"[ORCHESTRATOR] Retry detected, using original query: {original_query[:50]}...")
                        return original_query

                return query  # No continuation detected

            # This is a continuation! Combine with original request
            # The original incomplete request is 2 messages back
            if len(recent_messages.data) >= 2:
                original_query = recent_messages.data[1].get("query", "")

                app_logger.info(
                    f"[ORCHESTRATOR] Conversation continuation detected:\n"
                    f"  Original: {original_query[:50]}...\n"
                    f"  Follow-up: {query[:50]}..."
                )

                # Use LLM to intelligently combine the requests
                enriched = await self._combine_conversation_turns(
                    original_query,
                    last_ai_response,
                    query
                )

                return enriched

            return query

        except Exception as e:
            app_logger.error(f"[ORCHESTRATOR] Conversation continuation check failed: {e}")
            return query

    async def _combine_conversation_turns(
        self,
        original_query: str,
        ai_question: str,
        user_response: str
    ) -> str:
        """
        Use LLM to intelligently combine multi-turn conversation into complete request.

        Args:
            original_query: Original incomplete request (e.g., "send a message in slack")
            ai_question: AI's clarifying question (e.g., "What message would you like to send?")
            user_response: User's answer (e.g., "product launch date, send to #private-ch")

        Returns:
            Complete enriched query combining all information
        """
        try:
            prompt = f"""Combine this multi-turn conversation into a single, complete request.

**Original Request:** "{original_query}"

**AI Asked:** "{ai_question}"

**User Answered:** "{user_response}"

Combine these into ONE complete request that includes ALL the information.

Return ONLY the combined request, no explanation.

Examples:

Original: "send a message in slack"
AI Asked: "What message would you like to send, and which channel?"
User Answered: "product launch date, 17th nov 2025, send to private-ch channel"
Combined: "send a message in slack about product launch date 17th nov 2025 to private-ch channel"

Original: "create a task"
AI Asked: "What task would you like to create?"
User Answered: "review Q4 budget by Friday"
Combined: "create a task to review Q4 budget by Friday"

Now combine the conversation above:"""

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 200
                },
                timeout=10
            )

            if response.status_code == 200:
                combined = response.json()["choices"][0]["message"]["content"].strip()
                app_logger.info(f"[ORCHESTRATOR] Combined query: {combined}")
                return combined

            return f"{original_query} {user_response}"  # Fallback: simple concatenation

        except Exception as e:
            app_logger.error(f"[ORCHESTRATOR] Failed to combine conversation: {e}")
            return f"{original_query} {user_response}"  # Fallback

    def apply_meta_rules(self, query: str, intent: str, org_id: str, base_modules: List[str]) -> List[str]:
        """
        Apply discovered meta-rules to refine module selection.
        Phase 3, Step 4: Meta-Learning integration

        Args:
            query: User query
            intent: Classified intent
            org_id: Organization ID
            base_modules: Modules selected by heuristics

        Returns:
            Refined list of modules based on meta-rules
        """
        try:
            # Get applicable meta-rules for this intent
            meta_rules = self.meta_learning_service.get_applicable_meta_rules(
                org_id=org_id,
                intent=intent,
                min_confidence=0.7
            )

            if not meta_rules:
                return base_modules

            # Apply the highest confidence rule
            best_rule = max(meta_rules, key=lambda r: (r['success_rate'], r['confidence']))

            # Extract recommended modules from metadata
            recommended_modules_str = best_rule.get('metadata', {}).get('modules', '')
            if recommended_modules_str:
                recommended_modules = recommended_modules_str.split('+')

                # Log meta-rule application
                app_logger.info(
                    f"[META-LEARNING] Applying rule for '{intent}': {best_rule['rule_text'][:80]}"
                )

                # Update rule application stats
                self.supabase.table('ai_meta_knowledge')\
                    .update({
                        'application_count': best_rule['application_count'] + 1,
                        'last_applied_at': datetime.now().isoformat()
                    })\
                    .eq('id', best_rule['id'])\
                    .execute()

                return recommended_modules

            return base_modules

        except Exception as e:
            app_logger.error(f"[META-LEARNING] Error applying meta-rules: {e}")
            return base_modules

    def route_to_modules(self, query: str, intent: str, org_id: str) -> List[str]:
        """
        Determine which modules to use based on intent and query content.
        Now enhanced with Cognitive Decision Engine recommendations.

        Returns: List of module names to query
        """
        # PRIORITY 1: Use Cognitive Decision Engine recommendations if available
        if hasattr(self, '_last_cognitive_decision') and self._last_cognitive_decision:
            decision = self._last_cognitive_decision
            app_logger.info(
                f"[ORCHESTRATOR] Using cognitive engine module recommendations: "
                f"{decision.recommended_modules}"
            )
            return decision.recommended_modules

        # FALLBACK: Use heuristic-based module selection
        modules = []
        query_lower = query.lower()

        # Always use RAG for semantic retrieval
        modules.append("rag")

        # Use KG for entity/relationship queries
        if any(word in query_lower for word in ["who", "relationship", "connected", "related", "entity", "person", "company"]):
            modules.append("kg")

        # Use insights for trend/pattern queries
        if intent in ["insight", "summary"] or any(word in query_lower for word in ["trend", "pattern", "frequent", "common", "analytics"]):
            modules.append("insight")

        # Risk queries always use insights
        if intent == "risk":
            modules.append("insight")

        base_modules = list(set(modules))  # Remove duplicates

        # Apply meta-rules to refine module selection
        refined_modules = self.apply_meta_rules(query, intent, org_id, base_modules)

        return refined_modules

    async def delegate_to_agent(
        self,
        query: str,
        org_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delegate execution task to appropriate agent.
        Phase 4 addition for autonomous execution.

        Args:
            query: User query requesting action
            org_id: Organization ID
            user_id: User ID
            context: Additional context

        Returns:
            Agent execution result
        """
        try:
            app_logger.info(f"[ORCHESTRATOR] Delegating to agent: {query[:50]}...")

            # FIRST: Check if this is a plan approval request
            query_lower = query.lower().strip()
            approval_keywords = [
                "yes", "approve", "confirm", "proceed", "go ahead",
                "accept", "agreed", "ok", "okay", "sure", "do it"
            ]

            is_approval = any(keyword == query_lower or keyword in query_lower.split() for keyword in approval_keywords)

            if is_approval:
                app_logger.info(f"[ORCHESTRATOR] Detected approval keyword in query: '{query}'")

                # Check for pending plans requiring approval
                # Try both "pending" and "pending_approval" statuses
                pending_plans = self.supabase.table("ai_action_plans")\
                    .select("*")\
                    .eq("org_id", org_id)\
                    .in_("status", ["pending", "pending_approval"])\
                    .order("created_at", desc=True)\
                    .limit(1)\
                    .execute()

                if pending_plans.data:
                    plan = pending_plans.data[0]
                    app_logger.info(f"[ORCHESTRATOR] Found pending plan {plan['id']} with status='{plan['status']}' - approving and executing")

                    # Approve and execute the plan
                    return await self._approve_and_execute_plan(plan, org_id, user_id)
                else:
                    app_logger.warning(f"[ORCHESTRATOR] No pending plans found for org {org_id} to approve")

            # No pending plan to approve, continue with normal flow
            # Route to appropriate agent
            agent_type = await self.agent_registry.route_to_agent(org_id, query, context or {})

            if not agent_type:
                return {
                    "success": False,
                    "error": "No suitable agent found for this request",
                    "suggestion": "Try rephrasing your request or check available agents"
                }

            # Get agent instance
            try:
                agent = self.agent_registry.get_agent(org_id, agent_type)
            except ValueError as e:
                app_logger.warning(f"[ORCHESTRATOR] Agent not available: {e}")
                return {
                    "success": False,
                    "error": f"Agent '{agent_type}' is not available",
                    "suggestion": "The requested agent type is not currently registered"
                }

            # Create observation context
            obs_context = ObservationContext(
                query=query,
                trigger_type="orchestrator",
                org_id=org_id,
                user_id=user_id,
                metadata=context or {}
            )

            # Check if agent thinks action is needed
            should_act, reason = await agent.observe(obs_context)

            if not should_act:
                # Check if agent needs more information (interactive gathering)
                if reason and reason.startswith("NEED_INFO:"):
                    # Extract the conversational question
                    clarifying_question = reason.replace("NEED_INFO:", "").strip()
                    app_logger.info(
                        f"[ORCHESTRATOR] Agent needs more info, asking: {clarifying_question}"
                    )
                    return {
                        "success": True,
                        "action_needed": False,
                        "needs_more_info": True,
                        "clarifying_question": clarifying_question,
                        "reason": clarifying_question,  # For backward compatibility
                        "agent_type": agent_type
                    }

                # No action needed and no info requested
                return {
                    "success": True,
                    "action_needed": False,
                    "reason": reason or "No action required for this request",
                    "agent_type": agent_type
                }

            # Generate action plan
            plan = await agent.plan(query, context or {})

            # Validate plan with safety service
            is_valid, risk_level, issues = await self.safety_service.validate_plan(plan, org_id)

            if not is_valid:
                return {
                    "success": False,
                    "error": "Action plan validation failed",
                    "validation_issues": issues,
                    "agent_type": agent_type
                }

            # Create action plan in database
            plan_response = await self.action_planning.create_action_plan(
                org_id=org_id,
                plan=plan,
                agent_type=agent_type,
                trigger_type=TriggerType.ORCHESTRATOR,
                trigger_source={"query": query},
                user_id=user_id
            )

            # Prepare response based on approval requirements
            if plan.requires_approval:
                return {
                    "success": True,
                    "plan_created": True,
                    "plan_id": str(plan_response.id),
                    "requires_approval": True,
                    "risk_level": risk_level.value,
                    "goal": plan.goal,
                    "total_steps": plan.total_steps,
                    "agent_type": agent_type,
                    "message": f"Action plan created but requires approval due to {risk_level.value} risk level",
                    "next_action": "Review and approve the plan in the dashboard"
                }
            else:
                # Auto-execute low-risk actions
                app_logger.info(f"[ORCHESTRATOR] Auto-executing plan {plan_response.id} (low risk)")

                # Update plan status to approved
                self.supabase.table("ai_action_plans")\
                    .update({
                        "status": "approved",
                        "approval_status": "approved",
                        "approved_by": user_id,
                        "approved_at": datetime.utcnow().isoformat()
                    })\
                    .eq("id", str(plan_response.id))\
                    .execute()

                # Execute each step
                executed_steps = []
                for step in plan.steps:
                    app_logger.info(f"[ORCHESTRATOR] Executing step {step.step_index + 1}/{plan.total_steps}: {step.action_name}")

                    # Execute the step
                    result = await agent.execute(step)

                    executed_steps.append({
                        "step_index": step.step_index,
                        "action_name": step.action_name,
                        "success": result.success,
                        "result": result.result if result.success else None,
                        "error": result.error_message if not result.success else None
                    })

                    # If step failed, stop execution
                    if not result.success:
                        app_logger.error(f"[ORCHESTRATOR] Step {step.step_index} failed: {result.error_message}")
                        # Update plan status to failed
                        self.supabase.table("ai_action_plans")\
                            .update({"status": "failed"})\
                            .eq("id", str(plan_response.id))\
                            .execute()

                        return {
                            "success": False,
                            "plan_created": True,
                            "plan_id": str(plan_response.id),
                            "executed_steps": executed_steps,
                            "error": f"Step {step.step_index + 1} failed: {result.error_message}",
                            "agent_type": agent_type
                        }

                # All steps succeeded - mark plan as completed
                self.supabase.table("ai_action_plans")\
                    .update({
                        "status": "completed",
                        "completed_steps": plan.total_steps,
                        "completed_at": datetime.utcnow().isoformat()
                    })\
                    .eq("id", str(plan_response.id))\
                    .execute()

                app_logger.info(f"[ORCHESTRATOR] Plan {plan_response.id} completed successfully")

                return {
                    "success": True,
                    "plan_created": True,
                    "plan_id": str(plan_response.id),
                    "requires_approval": False,
                    "risk_level": risk_level.value,
                    "goal": plan.goal,
                    "total_steps": plan.total_steps,
                    "executed_steps": executed_steps,
                    "agent_type": agent_type,
                    "message": f"Action plan executed successfully. Completed {len(executed_steps)} steps.",
                    "next_action": "View execution results"
                }

        except Exception as e:
            app_logger.error(f"[ORCHESTRATOR] Agent delegation failed: {e}")
            return {
                "success": False,
                "error": f"Failed to delegate to agent: {str(e)}"
            }

    async def _approve_and_execute_plan(
        self,
        plan_data: Dict[str, Any],
        org_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Approve a pending plan and execute its steps.

        Args:
            plan_data: Plan data from database
            org_id: Organization ID
            user_id: User ID who approved

        Returns:
            Execution result
        """
        try:
            plan_id = plan_data["id"]
            agent_type = plan_data["agent_type"]
            steps = plan_data["steps"]

            app_logger.info(f"[ORCHESTRATOR] Approving plan {plan_id} with {len(steps)} steps")

            # Update plan status to approved
            self.supabase.table("ai_action_plans")\
                .update({
                    "status": "approved",
                    "approval_status": "approved",
                    "approved_by": user_id,
                    "approved_at": datetime.utcnow().isoformat()
                })\
                .eq("id", plan_id)\
                .execute()

            # Get the agent to execute the steps
            agent = self.agent_registry.get_agent(org_id, agent_type)

            # Execute each step
            executed_steps = []
            for step_data in steps:
                # Reconstruct ActionStep from JSON
                from app.agents.base_agent import ActionStep
                step = ActionStep(
                    step_index=step_data["step_index"],
                    action_type=step_data["action_type"],
                    action_name=step_data["action_name"],
                    description=step_data["description"],
                    target_integration=step_data.get("target_integration"),
                    target_resource=step_data.get("target_resource", {}),
                    parameters=step_data.get("parameters", {}),
                    risk_level=step_data.get("risk_level", "low"),
                    requires_approval=step_data.get("requires_approval", False),
                    depends_on=step_data.get("depends_on", []),
                    estimated_duration_ms=step_data.get("estimated_duration_ms", 1000)
                )

                app_logger.info(f"[ORCHESTRATOR] Executing step {step.step_index + 1}/{len(steps)}: {step.action_name}")

                # Execute the step
                result = await agent.execute(step)

                executed_steps.append({
                    "step_index": step.step_index,
                    "action_name": step.action_name,
                    "success": result.success,
                    "result": result.result if result.success else None,
                    "error": result.error_message if not result.success else None
                })

                # If step failed and it's critical, stop execution
                if not result.success:
                    app_logger.error(f"[ORCHESTRATOR] Step {step.step_index} failed: {result.error_message}")
                    # Update plan status to failed
                    self.supabase.table("ai_action_plans")\
                        .update({"status": "failed"})\
                        .eq("id", plan_id)\
                        .execute()

                    return {
                        "success": False,
                        "plan_id": plan_id,
                        "executed_steps": executed_steps,
                        "error": f"Step {step.step_index + 1} failed: {result.error_message}"
                    }

            # All steps succeeded - mark plan as completed
            self.supabase.table("ai_action_plans")\
                .update({
                    "status": "completed",
                    "completed_steps": len(steps),
                    "completed_at": datetime.utcnow().isoformat()
                })\
                .eq("id", plan_id)\
                .execute()

            app_logger.info(f"[ORCHESTRATOR] Plan {plan_id} completed successfully")

            return {
                "success": True,
                "plan_id": plan_id,
                "plan_approved": True,
                "executed_steps": executed_steps,
                "total_steps": len(steps),
                "message": f"Plan approved and executed successfully. Completed {len(steps)} steps.",
                "goal": plan_data.get("goal", "Unknown goal")
            }

        except Exception as e:
            app_logger.error(f"[ORCHESTRATOR] Failed to approve/execute plan: {e}")
            return {
                "success": False,
                "error": f"Failed to approve and execute plan: {str(e)}"
            }

    def execute_rag(self, query: str, org_id: str) -> Dict[str, Any]:
        """Execute RAG search and return results."""
        try:
            # Use retrieve_context which returns combined context (embeddings + memory + graph + insights)
            context_items = self.rag_service.retrieve_context(
                query=query,
                org_id=org_id,
                limit=5
            )

            return {
                "success": True,
                "results": context_items,
                "count": len(context_items)
            }
        except Exception as e:
            print(f"[ORCHESTRATOR] RAG execution failed: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }

    def execute_kg(self, query: str, org_id: str) -> Dict[str, Any]:
        """Execute Knowledge Graph query and return results."""
        try:
            # Get full knowledge graph
            kg_data = self.kg_service.get_knowledge_graph(org_id)

            # Extract relevant entities based on query
            query_lower = query.lower()
            relevant_entities = []

            for entity in kg_data.get("entities", []):
                entity_name = entity.get("name", "").lower()
                if entity_name in query_lower or query_lower in entity_name:
                    relevant_entities.append(entity)

            return {
                "success": True,
                "entities": relevant_entities[:5],  # Limit to top 5
                "total_entities": len(kg_data.get("entities", [])),
                "total_relationships": len(kg_data.get("relationships", []))
            }
        except Exception as e:
            print(f"[ORCHESTRATOR] KG execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "entities": []
            }

    def execute_insight(self, query: str, org_id: str) -> Dict[str, Any]:
        """Execute Insight analysis and return results."""
        try:
            insights = self.insight_service.get_insights(org_id)
            return {
                "success": True,
                "insights": insights
            }
        except Exception as e:
            print(f"[ORCHESTRATOR] Insight execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "insights": {}
            }

    def build_reasoning_chain(
        self,
        query: str,
        intent: str,
        module_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Build a reasoning chain from module results.

        Returns: List of reasoning steps
        """
        steps = []

        # Step 1: Information Retrieval
        if "rag" in module_results:
            rag_data = module_results["rag"]
            steps.append({
                "step": "retrieve",
                "module": "rag",
                "action": "Retrieved relevant documents",
                "result_count": rag_data.get("count", 0),
                "success": rag_data.get("success", False)
            })

        # Step 2: Entity Analysis
        if "kg" in module_results:
            kg_data = module_results["kg"]
            steps.append({
                "step": "analyze_entities",
                "module": "kg",
                "action": "Identified relevant entities and relationships",
                "entity_count": len(kg_data.get("entities", [])),
                "success": kg_data.get("success", False)
            })

        # Step 3: Insight Generation
        if "insight" in module_results:
            insight_data = module_results["insight"]
            steps.append({
                "step": "generate_insights",
                "module": "insight",
                "action": "Generated organizational insights",
                "insights_found": len(insight_data.get("insights", {})),
                "success": insight_data.get("success", False)
            })

        # Step 4: Synthesis
        steps.append({
            "step": "synthesize",
            "action": "Combining all module outputs for final response",
            "modules_combined": list(module_results.keys())
        })

        return steps

    def generate_final_response(
        self,
        query: str,
        intent: str,
        module_results: Dict[str, Any],
        reasoning_steps: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> str:
        """
        Generate final LLM response using combined module results.

        Args:
            query: User query
            intent: Classified intent
            module_results: Results from all modules
            reasoning_steps: Chain of reasoning
            temperature: LLM temperature (adaptive)
        """
        # Build context from module results
        context_parts = []

        # Add Memory context
        memories = module_results.get("memories", [])
        if memories:
            context_parts.append("=== Relevant Memories ===")
            for idx, memory in enumerate(memories[:3], 1):
                context_parts.append(f"{idx}. [{memory.get('memory_type')}] {memory.get('title')}: {memory.get('content', '')[:150]}...")

        # Add RAG context
        if "rag" in module_results and module_results["rag"].get("success"):
            rag_results = module_results["rag"].get("results", [])
            if rag_results:
                context_parts.append("\n=== Retrieved Documents ===")
                for idx, result in enumerate(rag_results[:3], 1):
                    context_parts.append(f"{idx}. {result.get('content', '')[:200]}...")

        # Add KG context
        if "kg" in module_results and module_results["kg"].get("success"):
            entities = module_results["kg"].get("entities", [])
            if entities:
                context_parts.append("\n=== Relevant Entities ===")
                for entity in entities[:3]:
                    context_parts.append(f"- {entity.get('name')} ({entity.get('type')})")

        # Add Insight context
        if "insight" in module_results and module_results["insight"].get("success"):
            insights = module_results["insight"].get("insights", {})
            if insights:
                context_parts.append("\n=== Organizational Insights ===")
                context_parts.append(f"- Active Contributors: {insights.get('top_contributors', [])}")
                context_parts.append(f"- Recent Activity: {insights.get('recent_activity_count', 0)} events")

        context = "\n".join(context_parts)

        # Create system prompt based on intent
        intent_instructions = {
            "question": "Answer naturally and conversationally. For greetings, just greet back warmly. For questions, answer directly using the context without over-explaining.",
            "task": "Provide clear, actionable steps to complete the requested task.",
            "summary": "Provide a comprehensive summary of the relevant information.",
            "insight": "Analyze the data and provide meaningful insights and patterns.",
            "risk": "Identify potential risks, concerns, and recommendations."
        }

        system_prompt = f"""You are Vibodh AI, a helpful and conversational assistant for company knowledge.

{intent_instructions.get(intent, 'Respond helpfully and naturally.')}

Context from knowledge base:
{context}

Be direct, natural, and conversational. Don't explain what you're doing - just do it."""

        # Call Groq API
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    "temperature": temperature,
                    "max_tokens": 1000
                }
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error generating response: {response.text}"

        except Exception as e:
            print(f"[ORCHESTRATOR] LLM generation failed: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"

    async def orchestrate_query(
        self,
        query: str,
        org_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration method with adaptive reasoning.

        Workflow:
        1. Load adaptive configuration
        2. Retrieve relevant memories
        3. Classify intent
        4. Route to modules (using adaptive weights)
        5. Execute module queries (with memory context)
        6. Build reasoning chain
        7. Generate final response (using adaptive LLM params)
        8. Store conversation memory
        9. Record performance metrics
        10. Log reasoning

        Returns: Complete orchestration result
        """
        start_time = time.time()

        print(f"[ORCHESTRATOR] Processing query: {query}")

        # Step 0a: Get adaptive configuration
        adaptive_config = self.adaptive_engine.get_adaptive_config(org_id)
        print(f"[ORCHESTRATOR] Using adaptive config: temp={adaptive_config['llm_temperature']}, max_items={adaptive_config['max_context_items']}")

        # Step 0b: Retrieve relevant memories (using adaptive threshold)
        memories = []
        try:
            memories = await self.memory_service.retrieve_relevant_memories(
                org_id=org_id,
                query=query,
                limit=3,
                min_importance=adaptive_config['memory_importance_threshold'],
                user_id=user_id
            )
            print(f"[ORCHESTRATOR] Retrieved {len(memories)} relevant memories")
        except Exception as e:
            print(f"[ORCHESTRATOR] Failed to retrieve memories: {e}")

        # Step 0c: Check for conversation continuation (interactive multi-turn)
        # If the last AI response was a clarifying question, combine contexts
        enriched_query = await self._check_conversation_continuation(
            query=query,
            org_id=org_id,
            user_id=user_id,
            memories=memories
        )

        if enriched_query != query:
            print(f"[ORCHESTRATOR] Detected conversation continuation, enriched query: {enriched_query[:100]}...")
            query = enriched_query  # Use the enriched query for the rest of processing

        # Step 1: Classify intent using Cognitive Decision Engine
        intent = await self.classify_intent(
            query=query,
            org_id=org_id,
            user_id=user_id,
            context={"memories": memories} if memories else {}
        )
        print(f"[ORCHESTRATOR] Classified intent: {intent}")

        # Step 1b: Check for execution intent (Phase 4)
        if intent == "execute":
            print(f"[ORCHESTRATOR] Execution intent detected, delegating to agent")

            agent_result = await self.delegate_to_agent(
                query=query,
                org_id=org_id,
                user_id=user_id,
                context={"memories": memories} if memories else {}
            )

            # Format agent result as final answer
            if agent_result.get("success"):
                if agent_result.get("plan_created"):
                    final_answer = f"I've created an action plan to: {agent_result.get('goal', 'complete your request')}\n\n"
                    final_answer += f"Plan ID: {agent_result.get('plan_id')}\n"
                    final_answer += f"Risk Level: {agent_result.get('risk_level', 'unknown')}\n"
                    final_answer += f"Total Steps: {agent_result.get('total_steps', 0)}\n\n"

                    if agent_result.get("requires_approval"):
                        final_answer += "⚠️ This plan requires your approval before execution.\n"
                        final_answer += "Please review the plan in the Agents dashboard."
                    elif agent_result.get("executed_steps"):
                        # Plan was auto-executed
                        final_answer += "✅ Action completed successfully!\n\n"
                        final_answer += "**Execution Summary:**\n"
                        for step in agent_result.get("executed_steps", []):
                            status_icon = "✅" if step["success"] else "❌"
                            final_answer += f"{status_icon} {step['action_name']}\n"
                            if not step["success"] and step.get("error"):
                                final_answer += f"   Error: {step['error']}\n"
                    else:
                        final_answer += "✅ This plan has been queued for automatic execution.\n"
                        final_answer += "You can monitor progress in the Agents dashboard."
                else:
                    # Check if this is an interactive info gathering response
                    if agent_result.get("needs_more_info"):
                        # Return the clarifying question as a natural conversational response
                        final_answer = agent_result.get("clarifying_question", "Could you provide more details?")
                    else:
                        final_answer = agent_result.get("reason", "No action was needed for your request.")
            else:
                final_answer = f"I couldn't create an action plan: {agent_result.get('error', 'Unknown error')}"
                if agent_result.get("suggestion"):
                    final_answer += f"\n\nSuggestion: {agent_result.get('suggestion')}"
                if agent_result.get("validation_issues"):
                    final_answer += f"\n\nValidation issues:\n" + "\n".join(f"- {issue}" for issue in agent_result["validation_issues"])

            # Return early for agent execution
            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "intent": intent,
                "modules_used": ["agent"],
                "reasoning_steps": [
                    {
                        "step": "classify_intent",
                        "result": "execute"
                    },
                    {
                        "step": "delegate_to_agent",
                        "result": agent_result
                    }
                ],
                "final_answer": final_answer,
                "context_sources": [],
                "execution_time_ms": execution_time_ms,
                "module_results": {"agent": agent_result}
            }

        # Step 2: Route to modules
        modules_to_use = self.route_to_modules(query, intent, org_id)
        print(f"[ORCHESTRATOR] Routing to modules: {modules_to_use}")

        # Step 3: Execute module queries
        module_results = {}

        # Include memories in module results
        if memories:
            module_results["memories"] = memories

        if "rag" in modules_to_use:
            module_results["rag"] = self.execute_rag(query, org_id)

        if "kg" in modules_to_use:
            module_results["kg"] = self.execute_kg(query, org_id)

        if "insight" in modules_to_use:
            module_results["insight"] = self.execute_insight(query, org_id)

        # Step 4: Build reasoning chain
        reasoning_steps = self.build_reasoning_chain(query, intent, module_results)

        # Step 5: Generate final response (using adaptive LLM params)
        final_answer = self.generate_final_response(
            query, intent, module_results, reasoning_steps,
            temperature=adaptive_config['llm_temperature']
        )

        # Step 6: Store conversation memory
        try:
            # Determine importance based on intent and response length
            importance_map = {
                "task": 0.8,
                "decision": 0.9,
                "insight": 0.7,
                "risk": 0.9,
                "summary": 0.6,
                "question": 0.4
            }
            importance = importance_map.get(intent, 0.5)

            # Create concise memory title
            memory_title = query[:80] + "..." if len(query) > 80 else query

            # Store memory with response summary
            memory_content = f"Query: {query}\n\nResponse: {final_answer[:300]}..."

            await self.memory_service.store_memory(
                org_id=org_id,
                title=memory_title,
                content=memory_content,
                memory_type="conversation",
                importance=importance,
                user_id=user_id,
                source_refs=[{"type": "orchestrator", "intent": intent}],
                metadata={
                    "intent": intent,
                    "modules_used": modules_to_use,
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }
            )
            print(f"[ORCHESTRATOR] Stored conversation memory")
        except Exception as e:
            print(f"[ORCHESTRATOR] Failed to store memory: {e}")

        # Calculate execution time
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Extract context sources
        context_sources = []

        # Add memory sources
        if memories:
            for idx, memory in enumerate(memories[:3]):
                context_sources.append({
                    "id": memory.get("id", f"memory-{idx}"),  # Use actual memory ID or generate one
                    "type": "memory",
                    "title": memory.get("title", "Untitled Memory"),
                    "source": memory.get("memory_type", "conversation"),
                    "snippet": memory.get("content", "")[:100] + "..." if memory.get("content") else ""
                })

        # Add document sources
        if "rag" in module_results and module_results["rag"].get("success"):
            for idx, result in enumerate(module_results["rag"].get("results", [])[:5]):
                context_sources.append({
                    "id": result.get("id", f"doc-{idx}"),  # Use actual document ID or generate one
                    "type": "document",
                    "title": result.get("title", "Untitled"),
                    "source": result.get("source_type", "unknown"),
                    "snippet": result.get("content", "")[:100] + "..." if result.get("content") else "",
                    "score": result.get("similarity_score")  # Add relevance score if available
                })

        # Step 6: Log reasoning
        reasoning_log_id = None
        try:
            log_data = {
                "org_id": org_id,
                "user_id": user_id,
                "query": query,
                "intent": intent,
                "modules_used": modules_to_use,
                "reasoning_steps": reasoning_steps,
                "response_summary": final_answer[:200] if len(final_answer) > 200 else final_answer,
                "final_answer": final_answer,
                "context_sources": context_sources,
                "execution_time_ms": execution_time_ms,
                "tokens_used": len(final_answer.split())  # Rough estimate
            }

            log_result = self.supabase.table("reasoning_logs").insert(log_data).execute()
            if log_result.data:
                reasoning_log_id = log_result.data[0]["id"]
            print(f"[ORCHESTRATOR] Logged reasoning to database")
        except Exception as e:
            print(f"[ORCHESTRATOR] Failed to log reasoning: {e}")

        # Step 7: Record performance metrics for adaptive learning
        try:
            # Calculate context relevance (avg of module success indicators)
            context_relevance = 0.5  # Default
            success_count = 0
            total_modules = len(modules_to_use)

            for module in modules_to_use:
                if module in module_results and module_results[module].get("success"):
                    success_count += 1

            if total_modules > 0:
                context_relevance = success_count / total_modules

            # Calculate confidence based on response quality indicators
            confidence_score = 0.7  # Default
            if len(context_sources) > 3:  # Good context
                confidence_score = 0.8
            if len(memories) > 0:  # Has memory context
                confidence_score += 0.1
            confidence_score = min(1.0, confidence_score)

            # Record feedback metrics (without user feedback yet)
            await self.feedback_service.record_feedback(
                org_id=org_id,
                query=query,
                intent=intent,
                modules_used=modules_to_use,
                response_time_ms=execution_time_ms,
                token_usage=len(final_answer.split()),
                confidence_score=confidence_score,
                context_relevance_score=context_relevance,
                context_items_used=len(context_sources),
                reasoning_log_id=reasoning_log_id,
                user_id=user_id,
                metadata={
                    "adaptive_config_used": {
                        "temperature": adaptive_config['llm_temperature'],
                        "max_context": adaptive_config['max_context_items']
                    }
                }
            )
            print(f"[ORCHESTRATOR] Recorded performance metrics")
        except Exception as e:
            print(f"[ORCHESTRATOR] Failed to record metrics: {e}")

        # Return complete result
        return {
            "intent": intent,
            "modules_used": modules_to_use,
            "reasoning_steps": reasoning_steps,
            "final_answer": final_answer,
            "context_sources": context_sources,
            "execution_time_ms": execution_time_ms,
            "module_results": module_results
        }


# Singleton instance
_orchestrator_service = None

def get_orchestrator_service(supabase: Client) -> OrchestratorService:
    """Get or create OrchestratorService instance."""
    global _orchestrator_service
    if _orchestrator_service is None:
        _orchestrator_service = OrchestratorService(supabase)
    return _orchestrator_service
