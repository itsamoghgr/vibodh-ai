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
        self.groq_api_key = settings.GROQ_API_KEY

    def classify_intent(self, query: str) -> str:
        """
        Classify user query intent using keyword heuristics and LLM.

        Returns: question | task | summary | insight | risk
        """
        query_lower = query.lower()

        # Keyword-based heuristics
        if any(word in query_lower for word in ["summarize", "summary", "overview", "recap"]):
            return "summary"

        if any(word in query_lower for word in ["risk", "danger", "warning", "concern", "threat"]):
            return "risk"

        if any(word in query_lower for word in ["insight", "trend", "pattern", "analysis", "what's happening"]):
            return "insight"

        if any(word in query_lower for word in ["create", "make", "build", "generate", "do", "task"]):
            return "task"

        # Default to question
        if "?" in query or any(word in query_lower for word in ["what", "who", "when", "where", "why", "how"]):
            return "question"

        # Use LLM for complex cases
        try:
            prompt = f"""Classify the following query into one category: question, task, summary, insight, or risk.

Query: "{query}"

Respond with ONLY the category name, nothing else."""

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 10
                }
            )

            if response.status_code == 200:
                intent = response.json()["choices"][0]["message"]["content"].strip().lower()
                if intent in ["question", "task", "summary", "insight", "risk"]:
                    return intent
        except Exception as e:
            print(f"[ORCHESTRATOR] LLM intent classification failed: {e}")

        return "question"  # Default fallback

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
        Now enhanced with meta-learning (Phase 3, Step 4)

        Returns: List of module names to query
        """
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

        # Apply meta-rules to refine module selection (Phase 3, Step 4)
        refined_modules = self.apply_meta_rules(query, intent, org_id, base_modules)

        return refined_modules

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
            "question": "Answer the user's question directly and accurately using the provided context.",
            "task": "Provide clear, actionable steps to complete the requested task.",
            "summary": "Provide a comprehensive summary of the relevant information.",
            "insight": "Analyze the data and provide meaningful insights and patterns.",
            "risk": "Identify potential risks, concerns, and recommendations."
        }

        system_prompt = f"""You are Vibodh AI, an intelligent assistant analyzing company knowledge.

Intent: {intent}
Instructions: {intent_instructions.get(intent, 'Respond helpfully and accurately.')}

Context from multiple sources:
{context}

Provide a clear, well-structured response."""

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

        # Step 1: Classify intent
        intent = self.classify_intent(query)
        print(f"[ORCHESTRATOR] Classified intent: {intent}")

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
            for memory in memories[:3]:
                context_sources.append({
                    "type": "memory",
                    "title": memory.get("title", "Untitled Memory"),
                    "source": memory.get("memory_type", "conversation")
                })

        # Add document sources
        if "rag" in module_results and module_results["rag"].get("success"):
            for result in module_results["rag"].get("results", [])[:5]:
                context_sources.append({
                    "type": "document",
                    "title": result.get("title", "Untitled"),
                    "source": result.get("source_type", "unknown")
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
