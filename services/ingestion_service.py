# -*- coding: utf-8 -*-
"""
Ingestion Service
Handles data ingestion from various sources and triggers embedding generation
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from supabase import Client
from connectors.slack_connector import get_slack_connector
from services.embedding_service import get_embedding_service


class IngestionService:
    def __init__(self, supabase_client: Client):
        """Initialize ingestion service"""
        self.supabase = supabase_client
        self.slack = get_slack_connector()
        self.embedding_service = get_embedding_service()

    async def ingest_slack(
        self,
        org_id: str,
        connection_id: str,
        channel_ids: Optional[List[str]] = None,
        days_back: int = 30
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
            for channel_id in channel_ids:
                # Fetch messages
                try:
                    messages = self.slack.fetch_messages(
                        access_token=access_token,
                        channel_id=channel_id,
                        days_back=days_back
                    )
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
                for msg in messages:
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

                    # If message has replies, fetch and append them
                    if msg.get("reply_count", 0) > 0:
                        try:
                            thread_replies = self.slack.fetch_thread_replies(
                                access_token=access_token,
                                channel_id=channel_id,
                                thread_ts=msg["ts"]
                            )
                            if thread_replies:
                                full_content += "\n\n--- Thread Replies ---\n" + "\n\n".join(thread_replies)
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

                            # Generate embeddings
                            await self._generate_embeddings(doc.data[0]["id"], doc_data["content"], org_id)
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
            from services.rag_service import invalidate_embeddings_cache
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

    async def _generate_embeddings(self, document_id: str, content: str, org_id: str):
        """
        Generate embeddings for a document

        Args:
            document_id: Document ID
            content: Document content
            org_id: Organization ID
        """
        try:
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
        threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using pgvector cosine similarity

        Args:
            query: Search query
            org_id: Organization ID
            limit: Max results (default 5)
            threshold: Similarity threshold 0-1 (default 0.4, lower = more permissive)

        Returns:
            List of matching documents with similarity scores
        """
        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embedding(query)

        try:
            # Use pgvector function for fast similarity search
            result = self.supabase.rpc(
                "match_embeddings",
                {
                    "query_embedding": query_embedding,
                    "filter_org_id": org_id,
                    "match_threshold": threshold,
                    "match_count": limit
                }
            ).execute()

            if not result.data:
                print(f"[INGEST] No embeddings found for org {org_id}")
                return []

            print(f"[INGEST] Found {len(result.data)} embeddings above threshold {threshold}")

            # Format results
            similarities = []
            for item in result.data:
                similarities.append({
                    "document_id": item["document_id"],
                    "content": item["content"],
                    "similarity": float(item["similarity"]),
                    "metadata": item.get("metadata", {})
                })

            return similarities

        except Exception as e:
            print(f"[INGEST] Error in pgvector search: {e}")
            # Fallback to empty results rather than crash
            return []


def get_ingestion_service(supabase_client: Client) -> IngestionService:
    """Create ingestion service instance"""
    return IngestionService(supabase_client)
