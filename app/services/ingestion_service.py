# -*- coding: utf-8 -*-
"""
Ingestion Service
Handles data ingestion from various sources and triggers embedding generation
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from supabase import Client
from app.services.embedding_service import get_embedding_service


class IngestionService:
    def __init__(self, supabase_client: Client):
        """Initialize ingestion service"""
        self.supabase = supabase_client
        # Slack connector will be injected when needed to avoid circular imports
        self._slack = None
        self.embedding_service = get_embedding_service()

    @property
    def slack(self):
        """Lazy load slack connector to avoid circular imports"""
        if self._slack is None:
            from app.connectors.slack_connector import SlackConnector
            self._slack = SlackConnector(self.supabase)
        return self._slack

    async def ingest_slack(
        self,
        org_id: str,
        connection_id: str,
        channel_ids: Optional[List[str]] = None,
        days_back: int = 3650  # ~10 years - effectively all history
    ) -> Dict[str, Any]:
        """
        Ingest messages from Slack

        Args:
            org_id: Organization ID
            connection_id: Connection ID
            channel_ids: Optional list of channel IDs to ingest (if None, ingest all)
            days_back: How many days back to fetch messages

        Returns:
            Ingestion job result
        """
        # Get connection details
        connection = self.supabase.table("connections")\
            .select("*")\
            .eq("id", connection_id)\
            .single()\
            .execute()

        if not connection.data:
            raise ValueError(f"Connection {connection_id} not found")

        conn_data = connection.data
        access_token = conn_data["access_token"]

        # Create ingestion job
        job = self.supabase.table("ingestion_jobs").insert({
            "org_id": org_id,
            "connection_id": connection_id,
            "source_type": "slack",
            "status": "running"
        }).execute()

        job_id = job.data[0]["id"]

        try:
            # Get channels to ingest (both public and private)
            if not channel_ids:
                # Get public channels
                public_channels = self.slack.list_channels(access_token, types="public_channel")

                # Try to get private channels (requires groups:read scope)
                private_channels = []
                try:
                    private_channels = self.slack.list_channels(access_token, types="private_channel", auto_join=False)
                    print(f"Found {len(public_channels)} public channels and {len(private_channels)} private channels")
                except Exception as e:
                    # If groups:read scope is missing, just skip private channels
                    error_message = str(e)
                    if "missing_scope" in error_message.lower() or "groups:read" in error_message.lower():
                        print(f"⊘ Skipping private channels: missing 'groups:read' and 'groups:history' scopes")
                        print(f"Found {len(public_channels)} public channels (private channels skipped)")
                        print(f"   To sync private channels, reinstall the Slack app with updated scopes")
                    else:
                        raise e

                # Combine both
                channels = public_channels + private_channels
                channel_ids = [ch["id"] for ch in channels]

            total_fetched = 0
            total_created = 0
            total_embedded = 0
            total_skipped = 0

            # Get existing document source_ids to avoid duplicates
            print(f"Checking for existing documents in org {org_id}...")
            existing_docs = self.supabase.table("documents")\
                .select("source_id")\
                .eq("org_id", org_id)\
                .eq("source_type", "slack")\
                .execute()

            existing_source_ids = set(doc["source_id"] for doc in existing_docs.data) if existing_docs.data else set()
            print(f"Found {len(existing_source_ids)} existing documents")

            # Ingest each channel
            for idx, channel_id in enumerate(channel_ids):
                print(f"[SYNC] Processing channel {idx+1}/{len(channel_ids)}: {channel_id}")

                # Fetch messages
                try:
                    messages = self.slack.fetch_messages(
                        access_token=access_token,
                        channel_id=channel_id,
                        days_back=days_back
                    )
                    print(f"[SYNC] Fetched {len(messages)} messages from channel {channel_id}")
                except Exception as e:
                    error_message = str(e)
                    # Skip channels we don't have access to
                    if "missing_scope" in error_message.lower() or "not_in_channel" in error_message.lower():
                        print(f"⊘ Skipping channel {channel_id}: {error_message}")
                        continue
                    else:
                        raise e

                total_fetched += len(messages)

                # Get channel info
                channels = self.slack.list_channels(access_token)
                channel_name = next(
                    (ch["name"] for ch in channels if ch["id"] == channel_id),
                    channel_id
                )

                # Process each message
                for msg_idx, msg in enumerate(messages):
                    if msg_idx % 10 == 0:  # Log every 10 messages
                        print(f"[SYNC] Processing message {msg_idx+1}/{len(messages)} in channel {channel_id}")
                    # Create unique source_id for this message
                    source_id = f"slack_{channel_id}_{msg['ts']}"

                    # Check if document already exists (using in-memory set for performance)
                    if source_id in existing_source_ids:
                        total_skipped += 1
                        continue

                    # Get user info
                    user_info = self.slack.get_user_info(access_token, msg["user"]) if msg.get("user") else {}

                    # Build full content including thread replies
                    full_content = msg["text"]

                    # Replace user mentions with actual names
                    import re
                    def replace_user_mention(match):
                        mentioned_user_id = match.group(1)
                        try:
                            mentioned_user = self.slack.get_user_info(access_token, mentioned_user_id)
                            return f"@{mentioned_user.get('real_name') or mentioned_user.get('name') or mentioned_user_id}"
                        except:
                            return f"<@{mentioned_user_id}>"

                    # Replace <@U123> with @Name
                    full_content = re.sub(r'<@([A-Z0-9]+)>', replace_user_mention, full_content)

                    # Extract and append file attachments, links, and other content
                    if msg.get("files"):
                        full_content += "\n\n--- Attached Files ---\n"
                        for file in msg["files"]:
                            file_name = file.get("name", "unknown")
                            file_type = file.get("mimetype", "unknown")
                            file_url = file.get("url_private", file.get("permalink", ""))
                            full_content += f"\n📎 {file_name} ({file_type})\nURL: {file_url}\n"

                            # For text files, try to download content
                            if file.get("mimetype", "").startswith("text/") or file_name.endswith((".txt", ".md", ".csv")):
                                try:
                                    import requests
                                    headers = {"Authorization": f"Bearer {access_token}"}
                                    file_response = requests.get(file.get("url_private", ""), headers=headers, timeout=10)
                                    if file_response.status_code == 200:
                                        file_content = file_response.text[:5000]  # Limit to 5000 chars
                                        full_content += f"\nFile Content:\n{file_content}\n"
                                except Exception as e:
                                    print(f"Could not download file {file_name}: {e}")

                    # Extract shared links
                    if msg.get("attachments"):
                        full_content += "\n\n--- Shared Links ---\n"
                        for attachment in msg["attachments"]:
                            if attachment.get("title") and attachment.get("title_link"):
                                full_content += f"\n🔗 {attachment['title']}: {attachment['title_link']}\n"
                            if attachment.get("text"):
                                full_content += f"{attachment['text'][:500]}\n"

                    # If message has replies, fetch and append them
                    if msg.get("reply_count", 0) > 0:
                        try:
                            thread_replies = self.slack.fetch_thread_replies(
                                access_token=access_token,
                                channel_id=channel_id,
                                thread_ts=msg["ts"]
                            )
                            if thread_replies:
                                # Also replace user mentions in thread replies
                                resolved_replies = [re.sub(r'<@([A-Z0-9]+)>', replace_user_mention, reply) for reply in thread_replies]
                                full_content += "\n\n--- Thread Replies ---\n" + "\n\n".join(resolved_replies)
                        except Exception as e:
                            print(f"Could not fetch thread replies for message {msg['ts']}: {e}")

                    # Create document
                    doc_data = {
                        "org_id": org_id,
                        "connection_id": connection_id,
                        "source_type": "slack",
                        "source_id": source_id,
                        "title": f"Message in #{channel_name}",
                        "content": full_content,
                        "author": user_info.get("real_name") or user_info.get("name"),
                        "author_id": msg.get("user"),
                        "channel_name": channel_name,
                        "channel_id": channel_id,
                        "url": f"slack://channel?id={channel_id}&message={msg['ts']}",
                        "metadata": {
                            "ts": msg["ts"],
                            "thread_ts": msg.get("thread_ts"),
                            "reply_count": msg.get("reply_count", 0)
                        },
                        "embedding_status": "pending"
                    }

                    # Insert new document
                    try:
                        doc = self.supabase.table("documents").insert(doc_data).execute()
                        if doc.data:
                            total_created += 1
                            existing_source_ids.add(source_id)  # Add to set to prevent duplicates in same run
                            print(f"✓ Created document: {doc_data['title']}")

                            # Generate embeddings and build KG
                            doc_metadata = {
                                "author": doc_data.get("author"),
                                "author_id": doc_data.get("author_id"),
                                "channel_name": doc_data.get("channel_name"),
                                "channel_id": doc_data.get("channel_id"),
                                "source_type": doc_data.get("source_type")
                            }
                            await self._generate_embeddings(doc.data[0]["id"], doc_data["content"], org_id, doc_metadata)
                            total_embedded += 1
                    except Exception as e:
                        # If it's a duplicate key error, skip silently
                        error_str = str(e).lower()
                        if "duplicate" in error_str or "unique" in error_str:
                            total_skipped += 1
                            existing_source_ids.add(source_id)
                        else:
                            print(f"Failed to insert document: {e}")
                        continue

            # Update job status
            self.supabase.table("ingestion_jobs").update({
                "status": "completed",
                "documents_fetched": total_fetched,
                "documents_created": total_created,
                "embeddings_generated": total_embedded,
                "completed_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()

            # Update connection last_sync_at
            self.supabase.table("connections").update({
                "last_sync_at": datetime.utcnow().isoformat()
            }).eq("id", connection_id).execute()

            # Invalidate embeddings cache since new documents were added
            from app.services.rag_service import invalidate_embeddings_cache
            invalidate_embeddings_cache(org_id)

            print(f"\n=== Sync Complete ===")
            print(f"Fetched: {total_fetched} messages")
            print(f"Created: {total_created} new documents")
            print(f"Skipped: {total_skipped} duplicates")
            print(f"Embeddings: {total_embedded} generated")

            return {
                "job_id": job_id,
                "status": "completed",
                "documents_fetched": total_fetched,
                "documents_created": total_created,
                "documents_skipped": total_skipped,
                "embeddings_generated": total_embedded
            }

        except Exception as e:
            # Update job status to failed
            self.supabase.table("ingestion_jobs").update({
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()

            raise Exception(f"Ingestion failed: {str(e)}")

    async def _generate_embeddings(self, document_id: str, content: str, org_id: str, metadata: dict = None):
        """
        Generate embeddings for a document and build knowledge graph

        Args:
            document_id: Document ID
            content: Document content
            org_id: Organization ID
            metadata: Document metadata (author, channel, etc.)
        """
        try:
            # Delete existing embeddings for this document (if any)
            self.supabase.table("embeddings")\
                .delete()\
                .eq("document_id", document_id)\
                .execute()

            # Chunk and embed the document
            embedded_chunks = self.embedding_service.embed_document(content)

            # Store embeddings
            for chunk in embedded_chunks:
                embedding_data = {
                    "org_id": org_id,
                    "document_id": document_id,
                    "content": chunk["content"],
                    "embedding": chunk["embedding"],
                    "metadata": chunk["metadata"]
                }

                self.supabase.table("embeddings").insert(embedding_data).execute()

            # Build knowledge graph from document
            try:
                from app.services.kg_service import get_kg_service
                kg_service = get_kg_service(self.supabase)
                kg_service.build_kg_from_document(
                    org_id=org_id,
                    document_id=document_id,
                    content=content,
                    metadata=metadata or {}
                )
            except Exception as kg_error:
                print(f"[KG] Failed to build graph for document {document_id}: {kg_error}")
                # Don't fail the whole process if KG extraction fails

            # Update document status
            self.supabase.table("documents").update({
                "embedding_status": "completed"
            }).eq("id", document_id).execute()

        except Exception as e:
            # Update document status to failed
            self.supabase.table("documents").update({
                "embedding_status": "failed"
            }).eq("id", document_id).execute()

            print(f"Failed to generate embeddings for document {document_id}: {e}")
            raise

    def search_embeddings(
        self,
        query: str,
        org_id: str,
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using embeddings

        Args:
            query: Search query
            org_id: Organization ID
            limit: Max results
            threshold: Similarity threshold

        Returns:
            List of matching documents with similarity scores
        """
        import numpy as np

        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embedding(query)
        query_vector = np.array(query_embedding)

        # Fetch all embeddings for this organization
        result = self.supabase.table("embeddings")\
            .select("id, document_id, content, embedding, metadata")\
            .eq("org_id", org_id)\
            .execute()

        print(f"[INGEST] DEBUG: Found {len(result.data) if result.data else 0} embeddings for org {org_id}")

        if not result.data:
            print("[INGEST] DEBUG: No embeddings found, returning empty list")
            return []

        # Calculate cosine similarity in Python
        similarities = []
        max_similarity = 0
        for item in result.data:
            # Convert embedding from database format to numpy array
            # Database may return it as string or list, handle both
            embedding = item["embedding"]
            if isinstance(embedding, str):
                import json
                embedding = json.loads(embedding)
            doc_vector = np.array(embedding, dtype=np.float32)

            # Calculate cosine similarity with zero-division protection
            # similarity = (A · B) / (||A|| * ||B||)
            dot_product = np.dot(query_vector, doc_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc_vector)
            similarity = dot_product / max(norm_query * norm_doc, 1e-8)

            max_similarity = max(max_similarity, similarity)

            # Only include if above threshold
            if similarity > threshold:
                similarities.append({
                    "document_id": item["document_id"],
                    "content": item["content"],
                    "similarity": float(similarity),
                    "metadata": item.get("metadata", {})
                })

        print(f"[INGEST] DEBUG: Max similarity found: {max_similarity:.4f}, threshold: {threshold}")
        print(f"[INGEST] DEBUG: Found {len(similarities)} documents above threshold")

        # Sort by similarity (highest first) and limit results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        print(f"[INGEST] DEBUG: Returning top {min(len(similarities), limit)} results")
        return similarities[:limit]


    async def ingest_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a single document (for ClickUp, files, etc.)

        Args:
            document: Document dict with org_id, content, metadata, etc.

        Returns:
            Ingested document with ID
        """
        try:
            org_id = document.get("org_id")
            content = document.get("content")

            if not org_id or not content:
                raise ValueError("Document must have org_id and content")

            # Insert or update document
            result = self.supabase.table("documents").upsert(document).execute()

            if not result.data:
                raise Exception("Failed to insert document")

            doc = result.data[0]
            document_id = doc["id"]

            # Generate embeddings
            await self._generate_embeddings(
                document_id=document_id,
                content=content,
                org_id=org_id,
                metadata=document.get("metadata", {})
            )

            return doc

        except Exception as e:
            logger.error(f"Error ingesting document: {e}", exc_info=True)
            raise

    async def handle_slack_event(
        self,
        event: Dict[str, Any],
        org_id: str,
        connection_id: str,
        access_token: str
    ) -> None:
        """
        Handle real-time Slack event (Phase 2 - Step 1)

        Args:
            event: Slack event payload
            org_id: Organization ID
            connection_id: Connection ID
            access_token: Slack access token
        """
        try:
            # Extract event data
            user_id = event.get("user")
            channel_id = event.get("channel")
            ts = event.get("ts")

            # Try to get formatted text from blocks first (has resolved mentions)
            text = ""
            blocks = event.get("blocks", [])
            if blocks:
                # Extract text from rich text blocks (preserves user mentions as names)
                for block in blocks:
                    if block.get("type") == "rich_text":
                        for element in block.get("elements", []):
                            if element.get("type") == "rich_text_section":
                                for item in element.get("elements", []):
                                    if item.get("type") == "text":
                                        text += item.get("text", "")
                                    elif item.get("type") == "user":
                                        # User mention - get the actual name
                                        user_id_mentioned = item.get("user_id")
                                        if user_id_mentioned:
                                            try:
                                                user_data = self.slack.get_user_info(access_token, user_id_mentioned)
                                                name = user_data.get("real_name") or user_data.get("name") or user_id_mentioned
                                                text += f"@{name}"
                                            except:
                                                text += f"<@{user_id_mentioned}>"
                                    elif item.get("type") == "link":
                                        text += item.get("url", "")

            # Fallback to plain text if blocks parsing failed
            if not text:
                text = event.get("text", "")

            if not all([text, channel_id, ts]):
                print(f"[REALTIME] Missing required fields in event")
                return

            # Create unique source_id for deduplication
            source_id = f"slack_{channel_id}_{ts}"

            # Check if document already exists
            existing = self.supabase.table("documents")\
                .select("id")\
                .eq("source_id", source_id)\
                .execute()

            if existing.data:
                print(f"[REALTIME] Document already exists: {source_id}")
                return

            # Get user info
            user_info = {}
            if user_id:
                try:
                    user_info = self.slack.get_user_info(access_token, user_id)
                except Exception as e:
                    print(f"[REALTIME] Could not fetch user info: {e}")

            # Get channel info
            channel_name = channel_id
            try:
                channels = self.slack.list_channels(access_token)
                channel_name = next(
                    (ch["name"] for ch in channels if ch["id"] == channel_id),
                    channel_id
                )
            except Exception as e:
                print(f"[REALTIME] Could not fetch channel info: {e}")

            author = user_info.get("real_name") or user_info.get("name") or user_id or "Unknown"

            # Replace user mentions with actual names
            import re
            def replace_user_mention(match):
                mentioned_user_id = match.group(1)
                try:
                    mentioned_user = self.slack.get_user_info(access_token, mentioned_user_id)
                    return f"@{mentioned_user.get('real_name') or mentioned_user.get('name') or mentioned_user_id}"
                except:
                    return f"<@{mentioned_user_id}>"

            # Replace <@U123> with @Name
            text = re.sub(r'<@([A-Z0-9]+)>', replace_user_mention, text)

            # Insert document
            doc_data = {
                "org_id": org_id,
                "connection_id": connection_id,
                "source_type": "slack",
                "source_id": source_id,
                "title": f"Message in #{channel_name}",
                "content": text,
                "author": author,
                "author_id": user_id,
                "channel_name": channel_name,
                "channel_id": channel_id,
                "embedding_status": "pending",
                "metadata": {
                    "ts": ts,
                    "event_time": event.get("event_ts"),
                    "channel_type": event.get("channel_type")
                }
            }

            doc = self.supabase.table("documents").insert(doc_data).execute()

            if not doc.data:
                print(f"[REALTIME] Failed to insert document")
                return

            document_id = doc.data[0]["id"]
            print(f"[REALTIME] Created document: {document_id}")

            # Generate and insert embeddings + build KG
            try:
                event_metadata = {
                    "author": author,
                    "author_id": user_id,
                    "channel_name": channel_name,
                    "channel_id": channel_id,
                    "source_type": "slack"
                }
                await self._generate_embeddings(document_id, text, org_id, event_metadata)
                print(f"[REALTIME] Generated embeddings and built KG for document: {document_id}")
            except Exception as e:
                print(f"[REALTIME] Failed to generate embeddings: {e}")

            # Store raw event in events table
            try:
                event_data = {
                    "org_id": org_id,
                    "source": "slack",
                    "actor_id": user_id,
                    "payload": event,
                    "happened_at": datetime.fromtimestamp(float(ts)).isoformat()
                }
                self.supabase.table("events").insert(event_data).execute()
                print(f"[REALTIME] Stored event in events table")
            except Exception as e:
                print(f"[REALTIME] Failed to store event: {e}")

            # Update connection with webhook status
            now_iso = datetime.now().isoformat()
            self.supabase.table("connections").update({
                "last_sync_at": now_iso,
                "metadata": {
                    "webhook_active": True,
                    "last_webhook_event": now_iso
                }
            }).eq("id", connection_id).execute()

        except Exception as e:
            print(f"[REALTIME] Error handling Slack event: {e}")
            raise


def get_ingestion_service(supabase_client: Client) -> IngestionService:
    """Create ingestion service instance"""
    return IngestionService(supabase_client)
