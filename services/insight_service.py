"""
AI Insight Service - Phase 2, Step 3
Generates organizational insights by analyzing Knowledge Graph and memory data
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from supabase import Client


class InsightService:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Initialize LLM client
        if self.groq_api_key:
            from groq import Groq
            self.llm_client = Groq(api_key=self.groq_api_key)
            self.model = "llama-3.3-70b-versatile"
            self.provider = "groq"
        elif self.openai_api_key:
            from openai import OpenAI
            self.llm_client = OpenAI(api_key=self.openai_api_key)
            self.model = "gpt-4"
            self.provider = "openai"
        else:
            raise ValueError("No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY")

    def _fetch_recent_kg_activity(self, org_id: str, days: int = 7) -> Dict[str, Any]:
        """Fetch recent Knowledge Graph activity (entities and edges)"""
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Get recent entities
        entities_result = self.supabase.table("kg_entities")\
            .select("id, name, type, created_at")\
            .eq("org_id", org_id)\
            .gte("created_at", cutoff_date)\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute()

        # Get recent edges directly from table
        edges_result = self.supabase.table("kg_edges")\
            .select("id, source_id, target_id, relation, confidence, created_at")\
            .eq("org_id", org_id)\
            .gte("created_at", cutoff_date)\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute()

        # Enrich edges with entity names
        recent_edges = []
        if edges_result.data:
            for edge in edges_result.data:
                # Get source entity name
                source = self.supabase.table("kg_entities")\
                    .select("name, type")\
                    .eq("id", edge["source_id"])\
                    .single()\
                    .execute()

                # Get target entity name
                target = self.supabase.table("kg_entities")\
                    .select("name, type")\
                    .eq("id", edge["target_id"])\
                    .single()\
                    .execute()

                if source.data and target.data:
                    recent_edges.append({
                        "relation": edge["relation"],
                        "confidence": edge["confidence"],
                        "source_name": source.data["name"],
                        "source_type": source.data["type"],
                        "target_name": target.data["name"],
                        "target_type": target.data["type"]
                    })

        return {
            "entities": entities_result.data or [],
            "edges": recent_edges,
            "period_days": days
        }

    def _fetch_recent_memory(self, org_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch recent AI memory summaries"""
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        result = self.supabase.table("ai_memory")\
            .select("title, content, created_at")\
            .eq("org_id", org_id)\
            .gte("created_at", cutoff_date)\
            .order("created_at", desc=True)\
            .limit(10)\
            .execute()

        return result.data or []

    def _fetch_recent_documents(self, org_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch recent document metadata for context"""
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        result = self.supabase.table("documents")\
            .select("id, author, channel_name, source_type, created_at")\
            .eq("org_id", org_id)\
            .gte("created_at", cutoff_date)\
            .order("created_at", desc=True)\
            .limit(50)\
            .execute()

        return result.data or []

    def _generate_insights_with_llm(self, context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use LLM to generate insights from context data"""

        # Build context summary
        kg_data = context_data.get("kg_activity", {})
        memory_data = context_data.get("memory", [])
        docs_data = context_data.get("documents", [])

        entities_summary = []
        entity_counts = {}
        for entity in kg_data.get("entities", []):
            entity_type = entity.get("type", "unknown")
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            entities_summary.append(f"- {entity['name']} ({entity_type})")

        edges_summary = []
        relation_counts = {}
        for edge in kg_data.get("edges", [])[:20]:  # Limit to top 20
            relation = edge.get("relation", "unknown")
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
            edges_summary.append(
                f"- {edge.get('source_name', 'Unknown')} {relation} {edge.get('target_name', 'Unknown')}"
            )

        memory_summary = []
        for mem in memory_data:
            if mem.get("title"):
                memory_summary.append(f"- {mem['title']}: {mem.get('content', '')[:100]}")

        # Construct prompt
        prompt = f"""You are an AI analyst examining organizational activity over the last {kg_data.get('period_days', 7)} days.

**Recent Activity Summary:**

New Entities Discovered:
{chr(10).join(entities_summary[:30]) if entities_summary else "None"}

Entity Counts by Type:
{json.dumps(entity_counts, indent=2)}

Recent Relationships:
{chr(10).join(edges_summary[:20]) if edges_summary else "None"}

Relationship Counts:
{json.dumps(relation_counts, indent=2)}

Recent Memory Summaries:
{chr(10).join(memory_summary[:5]) if memory_summary else "None"}

Document Activity:
- Total documents: {len(docs_data)}
- Active channels: {len(set(d.get('channel_name') for d in docs_data if d.get('channel_name')))}
- Contributors: {len(set(d.get('author') for d in docs_data if d.get('author')))}

**Task:**
Generate 3-5 concise, actionable insights about this organization. Focus on:
1. Key projects or topics being discussed
2. Team collaboration patterns
3. Emerging trends or focus areas
4. Potential risks or blockers
5. General observations

For each insight, provide:
- category (one of: project, team, trend, risk, general)
- title (short, max 60 chars)
- summary (2-3 sentences)
- recommendations (specific, actionable steps)
- confidence (0.0 to 1.0)

Return ONLY valid JSON in this exact format:
{{
  "insights": [
    {{
      "category": "project",
      "title": "Authentication System Active Development",
      "summary": "Multiple team members are actively working on the authentication system...",
      "recommendations": "Consider scheduling a sync meeting to align on API design. Document authentication flows.",
      "confidence": 0.85
    }}
  ]
}}

IMPORTANT: Return only the JSON object. No markdown, no explanations."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert organizational analyst. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            return result.get("insights", [])

        except json.JSONDecodeError as e:
            print(f"[InsightService] Failed to parse LLM response: {e}")
            print(f"[InsightService] Raw response: {content}")
            return []
        except Exception as e:
            print(f"[InsightService] Error generating insights: {e}")
            return []

    def generate_insights(self, org_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Generate AI insights for an organization

        Args:
            org_id: Organization UUID
            days: Number of days to look back (default: 7)

        Returns:
            Dict with success status, count, and generated insights
        """
        print(f"[InsightService] Generating insights for org {org_id} (last {days} days)")

        # Fetch context data
        kg_activity = self._fetch_recent_kg_activity(org_id, days)
        memory = self._fetch_recent_memory(org_id, days)
        documents = self._fetch_recent_documents(org_id, days)

        context_data = {
            "kg_activity": kg_activity,
            "memory": memory,
            "documents": documents
        }

        # Check if there's enough data
        if not kg_activity.get("entities") and not kg_activity.get("edges") and not memory:
            print("[InsightService] Not enough data to generate insights")
            return {
                "success": False,
                "message": "Not enough recent activity to generate insights",
                "insights_created": 0
            }

        # Generate insights using LLM
        insights = self._generate_insights_with_llm(context_data)

        if not insights:
            print("[InsightService] LLM did not generate any insights")
            return {
                "success": False,
                "message": "Failed to generate insights",
                "insights_created": 0
            }

        # Store insights in database
        created_count = 0
        for insight in insights:
            try:
                # Build source references
                source_refs = {
                    "entities_count": len(kg_activity.get("entities", [])),
                    "edges_count": len(kg_activity.get("edges", [])),
                    "memory_count": len(memory),
                    "docs_count": len(documents),
                    "period_days": days
                }

                self.supabase.table("ai_insights").insert({
                    "org_id": org_id,
                    "category": insight.get("category", "general"),
                    "title": insight.get("title", "Untitled Insight"),
                    "summary": insight.get("summary", ""),
                    "recommendations": insight.get("recommendations", ""),
                    "confidence": insight.get("confidence", 0.8),
                    "source_refs": source_refs
                }).execute()

                created_count += 1
                print(f"[InsightService] Created insight: {insight.get('title')}")

            except Exception as e:
                print(f"[InsightService] Failed to store insight: {e}")
                continue

        return {
            "success": True,
            "message": f"Generated {created_count} insights",
            "insights_created": created_count,
            "period_days": days
        }

    def list_insights(
        self,
        org_id: str,
        limit: int = 10,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List recent insights for an organization

        Args:
            org_id: Organization UUID
            limit: Maximum number of insights to return
            category: Optional filter by category

        Returns:
            List of insights
        """
        query = self.supabase.table("ai_insights")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)

        if category:
            query = query.eq("category", category)

        result = query.execute()
        return result.data or []

    def get_insight_stats(self, org_id: str) -> Dict[str, Any]:
        """Get insight statistics for an organization"""
        result = self.supabase.rpc(
            "get_insight_stats",
            {"org_uuid": org_id}
        ).execute()

        if result.data and len(result.data) > 0:
            return result.data[0]

        return {
            "total_insights": 0,
            "project_insights": 0,
            "team_insights": 0,
            "trend_insights": 0,
            "risk_insights": 0,
            "general_insights": 0,
            "avg_confidence": 0.0,
            "last_generated": None
        }


# Singleton pattern for service
_insight_service = None

def get_insight_service(supabase: Client) -> InsightService:
    """Get or create InsightService instance"""
    global _insight_service
    if _insight_service is None:
        _insight_service = InsightService(supabase)
    return _insight_service
