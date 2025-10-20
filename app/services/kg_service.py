# -*- coding: utf-8 -*-
"""
Knowledge Graph Service
Extracts entities and relationships from text to build organizational knowledge graph
"""

from app.core.config import settings
from typing import List, Dict, Any, Optional, Tuple
from supabase import Client
import json

# Debug flag
DEBUG = str(settings.DEBUG).lower() == "true"


class KGService:
    def __init__(self, supabase_client: Client):
        """Initialize KG service"""
        self.supabase = supabase_client
        self.llm_provider = "groq"

        # Initialize LLM client for extraction
        if self.llm_provider == "groq":
            from groq import Groq
            self.llm_client = Groq(api_key=settings.GROQ_API_KEY)
            self.model = "llama-3.3-70b-versatile"
        else:
            from openai import OpenAI
            self.llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = "gpt-4o-mini"

    def extract_entities_and_relations(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships from text using LLM

        Args:
            text: Text to analyze
            metadata: Optional metadata (author, channel, etc.)

        Returns:
            Dict with 'entities' and 'relations' lists
        """
        # Build prompt with metadata context
        context = ""
        source_type = "message"

        if metadata:
            if metadata.get("author"):
                context += f"\nAuthor: {metadata['author']}"
            if metadata.get("channel_name"):
                context += f"\nChannel: {metadata['channel_name']}"
            if metadata.get("space_name"):
                context += f"\nSpace: {metadata['space_name']}"
                source_type = "task"
            if metadata.get("list_name"):
                context += f"\nList: {metadata['list_name']}"
            if metadata.get("status"):
                context += f"\nStatus: {metadata['status']}"
            if metadata.get("assignees"):
                assignees = metadata['assignees']
                if isinstance(assignees, list) and assignees:
                    context += f"\nAssignees: {', '.join(assignees)}"

        # Customize prompt based on source type
        if source_type == "task":
            prompt = f"""Extract entities and relationships from this ClickUp task.

{context}

Task:
{text}

Extract:
1. People (names of team members, assignees)
2. Projects (specific projects, features, or initiatives)
3. Topics (technical topics, tools, technologies)
4. Issues (bugs, blockers, problems)
5. Relationships between entities (who works on what, what blocks what, who is assigned to what)

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "John Doe", "type": "person"}},
    {{"name": "auth system", "type": "project"}},
    {{"name": "React", "type": "topic"}}
  ],
  "relations": [
    {{"source": "John Doe", "relation": "assigned_to", "target": "auth system", "confidence": 0.9}},
    {{"source": "John Doe", "relation": "discussed", "target": "React", "confidence": 0.8}}
  ]
}}

Types: person, project, topic, tool, issue
Relations: assigned_to, works_on, discussed, fixed, blocked_by, uses, created, mentioned, commented_on

Return empty arrays if nothing found. JSON only, no explanation."""
        else:
            prompt = f"""Extract entities and relationships from this message.

{context}

Message:
{text}

Extract:
1. People (names of team members)
2. Projects (specific projects, features, or initiatives)
3. Topics (technical topics, tools, technologies)
4. Issues (bugs, blockers, problems)
5. Relationships between entities (who works on what, what blocks what, who discussed what)

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "John Doe", "type": "person"}},
    {{"name": "auth system", "type": "project"}},
    {{"name": "React", "type": "topic"}}
  ],
  "relations": [
    {{"source": "John Doe", "relation": "works_on", "target": "auth system", "confidence": 0.9}},
    {{"source": "John Doe", "relation": "discussed", "target": "React", "confidence": 0.8}}
  ]
}}

Types: person, project, topic, tool, issue
Relations: works_on, discussed, fixed, blocked_by, uses, created, mentioned

Return empty arrays if nothing found. JSON only, no explanation."""

        try:
            # Call LLM for extraction
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an entity extraction expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()

            result = json.loads(result_text)

            if DEBUG:
                print(f"[KG EXTRACT] Found {len(result.get('entities', []))} entities, {len(result.get('relations', []))} relations")

            return result

        except Exception as e:
            if DEBUG:
                print(f"[KG EXTRACT ERROR] {str(e)}")
            return {"entities": [], "relations": []}

    def get_or_create_entity(
        self, org_id: str, name: str, entity_type: str, metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Get existing entity or create new one (deduplication)

        Args:
            org_id: Organization ID
            name: Entity name
            entity_type: Entity type
            metadata: Optional metadata

        Returns:
            Entity ID (UUID)
        """
        try:
            # Check if entity exists (case-insensitive)
            existing = self.supabase.rpc(
                "get_entity_by_name",
                {"filter_org_id": org_id, "entity_name": name, "entity_type": entity_type}
            ).execute()

            if existing.data and len(existing.data) > 0:
                return existing.data[0]["id"]

            # Create new entity
            new_entity = {
                "org_id": org_id,
                "name": name.strip(),
                "type": entity_type,
                "metadata": metadata or {}
            }

            result = self.supabase.table("kg_entities").insert(new_entity).execute()

            if result.data and len(result.data) > 0:
                if DEBUG:
                    print(f"[KG] Created entity: {name} ({entity_type})")
                return result.data[0]["id"]

            return None

        except Exception as e:
            if DEBUG:
                print(f"[KG ERROR] Failed to get/create entity {name}: {str(e)}")
            return None

    def create_edge(
        self,
        org_id: str,
        source_id: str,
        target_id: str,
        relation: str,
        confidence: float = 0.8,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Create relationship edge between entities

        Args:
            org_id: Organization ID
            source_id: Source entity ID
            target_id: Target entity ID
            relation: Relationship type
            confidence: Confidence score (0-1)
            metadata: Optional metadata

        Returns:
            True if created successfully
        """
        try:
            edge_data = {
                "org_id": org_id,
                "source_id": source_id,
                "target_id": target_id,
                "relation": relation,
                "confidence": max(0.0, min(1.0, confidence)),
                "metadata": metadata or {}
            }

            self.supabase.table("kg_edges").insert(edge_data).execute()

            if DEBUG:
                print(f"[KG] Created edge: {relation} (confidence: {confidence:.2f})")

            return True

        except Exception as e:
            # Silently ignore duplicate edges (unique constraint)
            if "duplicate" not in str(e).lower():
                if DEBUG:
                    print(f"[KG ERROR] Failed to create edge: {str(e)}")
            return False

    def build_kg_from_document(
        self,
        org_id: str,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Extract and store entities/relations from a document

        Args:
            org_id: Organization ID
            document_id: Document ID (for tracking)
            content: Document content
            metadata: Optional metadata (author, channel, etc.)

        Returns:
            Dict with counts: {'entities_created': N, 'relations_created': M}
        """
        # Extract entities and relations
        extraction = self.extract_entities_and_relations(content, metadata)

        entities_created = 0
        relations_created = 0

        # Store entities
        entity_ids = {}
        for entity in extraction.get("entities", []):
            entity_name = entity.get("name")
            entity_type = entity.get("type")

            if not entity_name or not entity_type:
                continue

            entity_id = self.get_or_create_entity(
                org_id=org_id,
                name=entity_name,
                entity_type=entity_type,
                metadata={"source_document": document_id}
            )

            if entity_id:
                entity_ids[entity_name.lower()] = entity_id
                entities_created += 1

        # Store relationships
        for relation in extraction.get("relations", []):
            source_name = relation.get("source")
            target_name = relation.get("target")
            relation_type = relation.get("relation")
            confidence = relation.get("confidence", 0.8)

            if not all([source_name, target_name, relation_type]):
                continue

            source_id = entity_ids.get(source_name.lower())
            target_id = entity_ids.get(target_name.lower())

            if source_id and target_id:
                if self.create_edge(
                    org_id=org_id,
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation_type,
                    confidence=confidence,
                    metadata={"source_document": document_id}
                ):
                    relations_created += 1

        if DEBUG:
            print(f"[KG BUILD] Document {document_id}: {entities_created} entities, {relations_created} relations")

        return {
            "entities_created": entities_created,
            "relations_created": relations_created
        }

    def query_related_entities(
        self, org_id: str, entity_name: str, relation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query entities related to a given entity

        Args:
            org_id: Organization ID
            entity_name: Entity name to search for
            relation: Optional relation type filter

        Returns:
            List of related entities with relationship info
        """
        try:
            # Find entity
            entity_result = self.supabase.rpc(
                "get_entity_by_name",
                {"filter_org_id": org_id, "entity_name": entity_name}
            ).execute()

            if not entity_result.data or len(entity_result.data) == 0:
                return []

            entity_id = entity_result.data[0]["id"]

            # Get relationships
            relations_result = self.supabase.rpc(
                "get_entity_relationships",
                {"entity_uuid": entity_id}
            ).execute()

            results = relations_result.data if relations_result.data else []

            # Filter by relation type if specified
            if relation:
                results = [r for r in results if r["relation"] == relation]

            return results

        except Exception as e:
            if DEBUG:
                print(f"[KG QUERY ERROR] {str(e)}")
            return []

    def get_kg_stats(self, org_id: str) -> Dict[str, int]:
        """Get KG statistics for an organization"""
        try:
            result = self.supabase.rpc("get_kg_stats", {"org_uuid": org_id}).execute()

            if result.data and len(result.data) > 0:
                return result.data[0]

            return {
                "total_entities": 0,
                "total_edges": 0,
                "people_count": 0,
                "projects_count": 0,
                "topics_count": 0,
                "tools_count": 0,
                "issues_count": 0
            }

        except Exception as e:
            if DEBUG:
                print(f"[KG STATS ERROR] {str(e)}")
            return {}

    # ===== SCHEMA EVOLUTION METHODS (Phase 3, Step 4) =====

    def propose_new_entity_types(self, org_id: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Analyze recent entities and propose new entity types.

        Args:
            org_id: Organization ID
            days_back: Number of days to analyze

        Returns:
            List of proposed new entity types with justification
        """
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Current allowed types
            current_types = ['person', 'project', 'topic', 'tool', 'issue', 'channel', 'document']

            # Get recent entities
            recent_entities = self.supabase.table('kg_entities')\
                .select('name, type, metadata')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff_date.isoformat())\
                .execute()

            if not recent_entities.data or len(recent_entities.data) < 20:
                return []

            # Analyze with LLM to identify potential new types
            entities_sample = recent_entities.data[:50]
            entity_names = [e['name'] for e in entities_sample]

            prompt = f"""Analyze these entity names from a knowledge graph and identify if new entity types should be added.

Current entity types: {', '.join(current_types)}

Recent entities:
{chr(10).join(f"- {name}" for name in entity_names[:30])}

Are there recurring patterns or categories of entities that don't fit well into existing types?
If yes, propose up to 3 new entity types with:
1. Type name (lowercase, singular)
2. Description
3. Example entities from the list that would fit this type

Respond in JSON format:
{{"proposals": [{{"type": "...", "description": "...", "examples": ["...", "..."]}}]}}

If no new types needed, return: {{"proposals": []}}"""

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in knowledge graph schema design."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            import json
            result_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            if result_text.startswith('```json'):
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif result_text.startswith('```'):
                result_text = result_text.split('```')[1].split('```')[0].strip()

            result = json.loads(result_text)
            proposals = result.get('proposals', [])

            # Store proposals in kg_schema_evolution table
            for proposal in proposals:
                self.supabase.table('kg_schema_evolution').insert({
                    'org_id': org_id,
                    'change_type': 'new_entity_type',
                    'new_value': proposal['type'],
                    'reason': proposal['description'],
                    'status': 'proposed',
                    'metadata': {
                        'examples': proposal.get('examples', []),
                        'analysis_date': datetime.now().isoformat()
                    }
                }).execute()

            if DEBUG and proposals:
                print(f"[KG EVOLUTION] Proposed {len(proposals)} new entity types")

            return proposals

        except Exception as e:
            if DEBUG:
                print(f"[KG EVOLUTION ERROR] Failed to propose entity types: {str(e)}")
            return []

    def propose_new_relation_types(self, org_id: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Analyze recent relationships and propose new relation types.

        Args:
            org_id: Organization ID
            days_back: Number of days to analyze

        Returns:
            List of proposed new relation types
        """
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Get recent edges
            recent_edges = self.supabase.table('kg_edges')\
                .select('relation, metadata')\
                .eq('org_id', org_id)\
                .gte('created_at', cutoff_date.isoformat())\
                .execute()

            if not recent_edges.data or len(recent_edges.data) < 10:
                return []

            # Get unique relation types
            relations = {}
            for edge in recent_edges.data:
                rel = edge['relation']
                relations[rel] = relations.get(rel, 0) + 1

            # Get distinct relation types and their counts
            relation_summary = sorted(
                [(rel, count) for rel, count in relations.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20]

            prompt = f"""Analyze these relationship types from a knowledge graph and identify if new, more specific relation types should be added.

Current relations (with frequency):
{chr(10).join(f"- {rel}: {count} occurrences" for rel, count in relation_summary)}

Are there patterns suggesting we need more specific or additional relationship types?
Propose up to 3 new relation types with:
1. Relation name (lowercase, verb-like)
2. Description
3. How it differs from existing relations

Respond in JSON format:
{{"proposals": [{{"relation": "...", "description": "...", "use_case": "..."}}]}}

If no new relations needed, return: {{"proposals": []}}"""

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in knowledge graph schema design."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )

            import json
            result_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            if result_text.startswith('```json'):
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif result_text.startswith('```'):
                result_text = result_text.split('```')[1].split('```')[0].strip()

            result = json.loads(result_text)
            proposals = result.get('proposals', [])

            # Store proposals
            for proposal in proposals:
                self.supabase.table('kg_schema_evolution').insert({
                    'org_id': org_id,
                    'change_type': 'new_relation_type',
                    'new_value': proposal['relation'],
                    'reason': proposal['description'],
                    'status': 'proposed',
                    'metadata': {
                        'use_case': proposal.get('use_case', ''),
                        'analysis_date': datetime.now().isoformat()
                    }
                }).execute()

            if DEBUG and proposals:
                print(f"[KG EVOLUTION] Proposed {len(proposals)} new relation types")

            return proposals

        except Exception as e:
            if DEBUG:
                print(f"[KG EVOLUTION ERROR] Failed to propose relation types: {str(e)}")
            return []

    def apply_schema_evolution(
        self,
        org_id: str,
        evolution_id: str,
        approved_by: Optional[str] = None
    ) -> bool:
        """
        Apply an approved schema evolution change.

        Args:
            org_id: Organization ID
            evolution_id: Schema evolution record ID
            approved_by: User ID who approved the change

        Returns:
            True if successful
        """
        try:
            from datetime import datetime

            # Get the evolution record
            evolution = self.supabase.table('kg_schema_evolution')\
                .select('*')\
                .eq('id', evolution_id)\
                .eq('org_id', org_id)\
                .single()\
                .execute()

            if not evolution.data:
                if DEBUG:
                    print(f"[KG EVOLUTION] Evolution {evolution_id} not found")
                return False

            change = evolution.data

            # Update status to applied
            self.supabase.table('kg_schema_evolution')\
                .update({
                    'status': 'applied',
                    'reviewed_by': approved_by,
                    'reviewed_at': datetime.now().isoformat(),
                    'applied_at': datetime.now().isoformat()
                })\
                .eq('id', evolution_id)\
                .execute()

            if DEBUG:
                print(f"[KG EVOLUTION] Applied schema evolution: {change['change_type']} - {change['new_value']}")

            return True

        except Exception as e:
            if DEBUG:
                print(f"[KG EVOLUTION ERROR] Failed to apply schema evolution: {str(e)}")
            return False

    def get_schema_version(self, org_id: str) -> Dict[str, Any]:
        """
        Get current KG schema version and composition.

        Args:
            org_id: Organization ID

        Returns:
            Schema version information
        """
        try:
            result = self.supabase.rpc(
                'get_kg_schema_version',
                {'org_uuid': org_id}
            ).execute()

            if result.data and len(result.data) > 0:
                return result.data[0]

            return {
                'version': 1,
                'entity_types': [],
                'relation_types': [],
                'total_entities': 0,
                'total_relations': 0,
                'last_evolution_at': None
            }

        except Exception as e:
            if DEBUG:
                print(f"[KG SCHEMA VERSION ERROR] {str(e)}")
            return {
                'version': 1,
                'entity_types': [],
                'relation_types': [],
                'total_entities': 0,
                'total_relations': 0,
                'last_evolution_at': None
            }


def get_kg_service(supabase_client: Client) -> KGService:
    """Get KG service instance"""
    return KGService(supabase_client)
