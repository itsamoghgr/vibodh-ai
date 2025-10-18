# -*- coding: utf-8 -*-
"""
Vibodh AI - FastAPI Backend
Phase 1 - Step 2: Data Ingestion + Embeddings + RAG
"""

from fastapi import FastAPI, HTTPException, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Optional
import hmac
import hashlib
import time

# Import models
from models.schemas import (
    HealthResponse,
    OrganizationInfo,
    SlackIngestRequest,
    EmbeddingSearchRequest,
    EmbeddingSearchResponse,
    ChatQueryRequest,
    ChatQueryResponse,
    ChatStreamRequest,
    ChatSessionResponse,
    ChatMessageResponse,
    FeedbackCreate
)

# Import services
from connectors.slack_connector import get_slack_connector
from services.ingestion_service import get_ingestion_service
from services.llm_service import get_llm_service
from services.rag_service import get_rag_service

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Vibodh AI API",
    description="AI Brain for Your Company - Backend API",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Initialize services
slack_connector = get_slack_connector()
llm_service = get_llm_service()

# ============================================
# BASIC ROUTES
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vibodh AI API",
        "version": "2.0.0",
        "phase": "Step 2 - Data Ingestion + Embeddings",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        response = supabase.table("organizations").select("id", count="exact").execute()
        db_status = "connected"
        if hasattr(response, 'count'):
            db_status = f"connected ({response.count} organizations)"

        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            database=db_status,
            version="2.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")

# ============================================
# SLACK OAUTH ROUTES
# ============================================

@app.get("/api/connect/slack")
async def start_slack_oauth(org_id: str = Query(...)):
    """
    Start Slack OAuth flow
    Redirects user to Slack authorization page
    """
    try:
        # Generate state for CSRF protection
        state = f"{org_id}:{datetime.utcnow().timestamp()}"

        # Get authorization URL
        auth_url = slack_connector.get_authorization_url(state=state)

        return RedirectResponse(url=auth_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start OAuth: {str(e)}")

@app.get("/api/connect/slack/callback")
async def slack_oauth_callback(code: str, state: str = None):
    """
    Handle Slack OAuth callback
    Exchange code for access token and save connection
    """
    try:
        # Extract org_id from state
        if not state:
            raise HTTPException(status_code=400, detail="Missing state parameter")

        org_id = state.split(":")[0] if state else None
        if not org_id or org_id.strip() == '':
            raise HTTPException(status_code=400, detail=f"Invalid or empty org_id in state: {state}")

        # Exchange code for token
        token_data = slack_connector.exchange_code_for_token(code)

        # Debug: Print scopes
        print(f"📋 Slack OAuth scopes received: {token_data.get('scope', 'N/A')}")

        # Get workspace info
        workspace_info = slack_connector.get_workspace_info(token_data["access_token"])

        # Save connection to database
        connection_data = {
            "org_id": org_id,
            "source_type": "slack",
            "status": "active",
            "access_token": token_data["access_token"],
            "workspace_name": workspace_info["name"],
            "workspace_id": workspace_info["id"],
            "metadata": {
                "domain": workspace_info.get("domain"),
                "team_id": token_data["team_id"]
            }
        }

        # Insert or update connection
        existing = supabase.table("connections")\
            .select("id")\
            .eq("org_id", org_id)\
            .eq("source_type", "slack")\
            .execute()

        if existing.data:
            # Update existing
            supabase.table("connections")\
                .update(connection_data)\
                .eq("id", existing.data[0]["id"])\
                .execute()
        else:
            # Insert new
            supabase.table("connections").insert(connection_data).execute()

        # Redirect back to frontend integrations page
        return RedirectResponse(url="http://localhost:3000/dashboard/integrations?slack=connected")

    except Exception as e:
        print(f"OAuth callback error: {e}")
        return RedirectResponse(url=f"http://localhost:3000/dashboard/integrations?error={str(e)}")

# ============================================
# INGESTION ROUTES
# ============================================

@app.post("/api/ingest/slack")
async def ingest_slack(request: SlackIngestRequest):
    """
    Ingest messages from Slack
    Fetches messages, creates documents, and generates embeddings
    """
    try:
        print(f"Starting Slack ingestion for org_id={request.org_id}, connection_id={request.connection_id}")
        ingestion_service = get_ingestion_service(supabase)

        result = await ingestion_service.ingest_slack(
            org_id=request.org_id,
            connection_id=request.connection_id,
            channel_ids=request.channel_ids,
            days_back=request.days_back
        )

        return result
    except Exception as e:
        import traceback
        print(f"Ingestion error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/api/ingestion/jobs")
async def get_ingestion_jobs(org_id: str = Query(...), limit: int = Query(default=10, ge=1, le=50)):
    """
    Get ingestion jobs for an organization
    """
    try:
        result = supabase.table("ingestion_jobs")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("started_at", desc=True)\
            .limit(limit)\
            .execute()

        return {"jobs": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch jobs: {str(e)}")

@app.get("/api/ingestion/stats")
async def get_ingestion_stats(org_id: str = Query(...)):
    """
    Get sync statistics by channel
    """
    try:
        # Get document stats grouped by channel
        result = supabase.rpc("get_channel_stats", {"p_org_id": org_id}).execute()

        return {"channels": result.data if result.data else []}
    except Exception as e:
        # Fallback if RPC doesn't exist
        result = supabase.table("documents")\
            .select("channel_name, channel_id, source_type")\
            .eq("org_id", org_id)\
            .eq("source_type", "slack")\
            .execute()

        # Group by channel
        from collections import defaultdict
        channel_stats = defaultdict(lambda: {"count": 0, "channel_name": "", "channel_id": ""})
        for doc in result.data:
            channel_id = doc.get("channel_id", "unknown")
            channel_stats[channel_id]["count"] += 1
            channel_stats[channel_id]["channel_name"] = doc.get("channel_name", channel_id)
            channel_stats[channel_id]["channel_id"] = channel_id

        return {"channels": list(channel_stats.values())}

# ============================================
# FIX USER MENTIONS ROUTE
# ============================================

@app.post("/api/fix-user-mentions/{org_id}")
async def fix_user_mentions(org_id: str):
    """
    Fix user mentions in all documents by replacing IDs with actual names.
    Uses the author information we already have stored.
    """
    try:
        import re

        # Get all unique users with their IDs and names
        user_result = supabase.table("documents")\
            .select("author_id, author")\
            .eq("org_id", org_id)\
            .not_.is_("author_id", "null")\
            .not_.is_("author", "null")\
            .execute()

        # Build user ID to name mapping
        user_mapping = {}
        for doc in user_result.data:
            if doc.get("author_id") and doc.get("author") and doc["author"] != "Unknown":
                user_mapping[doc["author_id"]] = doc["author"]

        print(f"[FIX MENTIONS] Found {len(user_mapping)} unique users to map")

        # Get all documents with user mentions
        docs_result = supabase.table("documents")\
            .select("id, content")\
            .eq("org_id", org_id)\
            .execute()

        docs_updated = 0
        embeddings_updated = 0

        # Update each document
        for doc in docs_result.data:
            original_content = doc["content"]
            updated_content = original_content

            # Replace each user ID with their name
            for user_id, user_name in user_mapping.items():
                # Replace <@U123> format
                updated_content = updated_content.replace(f"<@{user_id}>", f"@{user_name}")
                # Replace @U123 format
                updated_content = re.sub(rf'@{user_id}(?![A-Z0-9])', f"@{user_name}", updated_content)

            # Only update if content changed
            if updated_content != original_content:
                supabase.table("documents").update({
                    "content": updated_content
                }).eq("id", doc["id"]).execute()
                docs_updated += 1

                # Also update corresponding embeddings
                embeddings_result = supabase.table("embeddings")\
                    .select("id, content")\
                    .eq("document_id", doc["id"])\
                    .execute()

                for emb in embeddings_result.data:
                    emb_content = emb["content"]
                    for user_id, user_name in user_mapping.items():
                        emb_content = emb_content.replace(f"<@{user_id}>", f"@{user_name}")
                        emb_content = re.sub(rf'@{user_id}(?![A-Z0-9])', f"@{user_name}", emb_content)

                    if emb_content != emb["content"]:
                        supabase.table("embeddings").update({
                            "content": emb_content
                        }).eq("id", emb["id"]).execute()
                        embeddings_updated += 1

        print(f"[FIX MENTIONS] Updated {docs_updated} documents and {embeddings_updated} embeddings")

        return {
            "success": True,
            "users_mapped": len(user_mapping),
            "documents_updated": docs_updated,
            "embeddings_updated": embeddings_updated
        }

    except Exception as e:
        print(f"[FIX MENTIONS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# EMBEDDINGS & SEARCH ROUTES
# ============================================

@app.post("/api/retry-failed-embeddings/{org_id}")
async def retry_failed_embeddings(org_id: str):
    """
    Retry embedding generation for all documents with 'failed' status
    First check if embeddings exist, if so just mark as completed
    """
    try:
        ingestion_service = get_ingestion_service(supabase)

        # Get all failed documents
        failed_result = supabase.table("documents")\
            .select("id, content")\
            .eq("org_id", org_id)\
            .eq("embedding_status", "failed")\
            .execute()

        if not failed_result.data:
            return {
                "success": True,
                "message": "No failed embeddings to retry",
                "retried": 0
            }

        success_count = 0
        already_embedded_count = 0
        failed_count = 0

        for doc in failed_result.data:
            try:
                # Check if embeddings already exist for this document
                existing_emb = supabase.table("embeddings")\
                    .select("id")\
                    .eq("org_id", org_id)\
                    .eq("document_id", doc["id"])\
                    .limit(1)\
                    .execute()

                if existing_emb.data and len(existing_emb.data) > 0:
                    # Embeddings exist, just update status
                    supabase.table("documents")\
                        .update({"embedding_status": "completed"})\
                        .eq("id", doc["id"])\
                        .execute()
                    already_embedded_count += 1
                    print(f"[RETRY] Document {doc['id']} already has embeddings, marked as completed")
                else:
                    # No embeddings, generate them
                    await ingestion_service._generate_embeddings(
                        document_id=doc["id"],
                        content=doc["content"],
                        org_id=org_id
                    )
                    success_count += 1
                    print(f"[RETRY] Successfully embedded document {doc['id']}")

            except Exception as e:
                print(f"[RETRY] Failed to embed document {doc['id']}: {str(e)}")
                failed_count += 1

        return {
            "success": True,
            "message": f"Processed {len(failed_result.data)} failed documents",
            "newly_embedded": success_count,
            "already_had_embeddings": already_embedded_count,
            "failed": failed_count,
            "total_processed": len(failed_result.data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retry failed: {str(e)}")

@app.post("/api/embeddings/search", response_model=EmbeddingSearchResponse)
async def search_embeddings(request: EmbeddingSearchRequest):
    """
    Search for similar documents using semantic search
    """
    try:
        ingestion_service = get_ingestion_service(supabase)

        results = ingestion_service.search_embeddings(
            query=request.query,
            org_id=request.org_id,
            limit=request.limit,
            threshold=request.threshold
        )

        # Format results
        formatted_results = [
            {
                "document_id": r["document_id"],
                "content": r["content"],
                "similarity": r["similarity"],
                "metadata": r["metadata"]
            }
            for r in results
        ]

        return EmbeddingSearchResponse(
            results=formatted_results,
            query=request.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ============================================
# RAG / CHAT ROUTES
# ============================================

@app.post("/api/chat/query", response_model=ChatQueryResponse)
async def chat_query(request: ChatQueryRequest):
    """
    RAG endpoint: Retrieve relevant context and generate answer
    """
    try:
        ingestion_service = get_ingestion_service(supabase)

        # Search for relevant documents
        context_items = ingestion_service.search_embeddings(
            query=request.query,
            org_id=request.org_id,
            limit=request.max_context_items,
            threshold=0.3  # Lowered from 0.7 to get more results
        )

        if not context_items:
            return ChatQueryResponse(
                query=request.query,
                context=[],
                suggested_answer="I couldn't find any relevant information in your company's knowledge base to answer this question."
            )

        # Format context
        formatted_context = [
            {
                "document_id": item["document_id"],
                "content": item["content"],
                "similarity": item["similarity"],
                "metadata": item["metadata"]
            }
            for item in context_items
        ]

        # Generate answer using Groq
        answer = llm_service.generate_answer_from_context(
            query=request.query,
            context_items=formatted_context
        )

        return ChatQueryResponse(
            query=request.query,
            context=formatted_context,
            suggested_answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream(request: ChatStreamRequest):
    """
    Streaming RAG endpoint: Retrieve context and stream LLM response
    Creates/updates chat session and stores messages
    """
    try:
        rag_service = get_rag_service(supabase)

        # Create or get session
        session_id = request.session_id
        if not session_id and request.user_id:
            # Only create session if user_id is provided
            session_data = {
                "org_id": request.org_id,
                "user_id": request.user_id,
                "title": request.query[:100]  # Use first 100 chars of query as title
            }
            session = supabase.table("chat_sessions").insert(session_data).execute()
            session_id = session.data[0]["id"]

        # Get conversation history (last 10 messages)
        conversation_history = []
        if session_id:
            history_result = supabase.table("chat_messages")\
                .select("role, content")\
                .eq("session_id", session_id)\
                .order("created_at", desc=False)\
                .limit(10)\
                .execute()
            if history_result.data:
                conversation_history = history_result.data

        # Store user message (only if session exists)
        if session_id:
            user_message = {
                "session_id": session_id,
                "role": "user",
                "content": request.query
            }
            supabase.table("chat_messages").insert(user_message).execute()

        # Check if simple query
        simple_queries = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "ok", "okay"]
        is_simple_query = request.query.lower().strip() in simple_queries or len(request.query.strip()) < 10

        # Get context for metadata (skip for simple queries)
        context_items = []
        if not is_simple_query:
            rag_service_instance = get_rag_service(supabase)

            # If we have conversation history, use it to improve the search query
            search_query = request.query
            if conversation_history and len(conversation_history) > 0:
                # Get last few messages for context
                recent_context = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
                context_summary = " ".join([msg["content"] for msg in recent_context])
                # Combine with current query for better embedding
                search_query = f"{context_summary} {request.query}"

            context_items = rag_service_instance.retrieve_context(
                query=search_query,
                org_id=request.org_id,
                limit=request.max_context_items,
                threshold=0.3  # Lower threshold for better recall
            )

        # Generate streaming response
        async def generate():
            import json
            full_response = ""

            # Send session_id first
            yield f"data: {json.dumps({'session_id': session_id, 'type': 'session'})}\n\n"

            # Send context (only if not a simple query and has context)
            if context_items:
                yield f"data: {json.dumps({'type': 'context', 'context': context_items})}\n\n"

            # Stream answer (with user_id for memory personalization and conversation history)
            async for chunk in rag_service.generate_answer_stream(
                query=request.query,
                org_id=request.org_id,
                max_context_items=request.max_context_items,
                user_id=request.user_id,
                conversation_history=conversation_history
            ):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            # Store assistant message (only if session exists)
            if session_id:
                assistant_message = {
                    "session_id": session_id,
                    "role": "assistant",
                    "content": full_response,
                }
                # Only include context if there are items
                if context_items and len(context_items) > 0:
                    assistant_message["context"] = context_items
                supabase.table("chat_messages").insert(assistant_message).execute()

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        import traceback
        print(f"Chat stream error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Chat stream failed: {str(e)}")

@app.get("/api/chat/history")
async def get_chat_history(user_id: str = Query(...), limit: int = Query(default=10, ge=1, le=50)):
    """
    Get chat session history for a user
    """
    try:
        result = supabase.table("chat_sessions")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("updated_at", desc=True)\
            .limit(limit)\
            .execute()

        return {"sessions": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chat history: {str(e)}")

@app.get("/api/chat/{session_id}")
async def get_chat_session(session_id: str):
    """
    Get a specific chat session with all messages
    """
    try:
        # Get session
        session = supabase.table("chat_sessions")\
            .select("*")\
            .eq("id", session_id)\
            .single()\
            .execute()

        if not session.data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get messages
        messages = supabase.table("chat_messages")\
            .select("*")\
            .eq("session_id", session_id)\
            .order("created_at", desc=False)\
            .execute()

        return {
            "session": session.data,
            "messages": messages.data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch session: {str(e)}")

@app.post("/api/chat/feedback")
async def submit_feedback(feedback: FeedbackCreate, user_id: str = Query(...)):
    """
    Submit feedback (thumbs up/down) for an assistant message
    """
    try:
        feedback_data = {
            "message_id": feedback.message_id,
            "user_id": user_id,
            "rating": feedback.rating,
            "comment": feedback.comment
        }

        # Upsert (insert or update if exists)
        result = supabase.table("chat_feedback")\
            .upsert(feedback_data)\
            .execute()

        return {"message": "Feedback submitted successfully", "data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

# ============================================
# CONNECTION MANAGEMENT
# ============================================

@app.get("/api/connections")
async def list_connections(org_id: str = Query(...)):
    """List all connections for an organization"""
    try:
        result = supabase.table("connections")\
            .select("*")\
            .eq("org_id", org_id)\
            .execute()

        return {"connections": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch connections: {str(e)}")

@app.delete("/api/connections/{connection_id}")
async def delete_connection(connection_id: str):
    """Delete a connection"""
    try:
        supabase.table("connections")\
            .delete()\
            .eq("id", connection_id)\
            .execute()

        return {"message": "Connection deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete connection: {str(e)}")

# ============================================
# DOCUMENTS ROUTES
# ============================================

@app.get("/api/documents")
async def list_documents(org_id: str = Query(...), limit: int = 100):
    """List documents for an organization"""
    try:
        result = supabase.table("documents")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()

        return {"documents": result.data, "total": len(result.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

@app.delete("/api/documents/cleanup")
async def cleanup_slack_documents(org_id: str = Query(...)):
    """Delete all Slack documents and embeddings for an organization"""
    try:
        # Get all Slack document IDs
        docs = supabase.table("documents")\
            .select("id")\
            .eq("org_id", org_id)\
            .eq("source_type", "slack")\
            .execute()

        doc_ids = [doc["id"] for doc in docs.data] if docs.data else []

        if not doc_ids:
            return {"message": "No Slack documents to delete", "deleted": 0}

        # Delete embeddings first (to avoid foreign key issues)
        for doc_id in doc_ids:
            supabase.table("embeddings")\
                .delete()\
                .eq("document_id", doc_id)\
                .execute()

        # Delete documents
        supabase.table("documents")\
            .delete()\
            .eq("org_id", org_id)\
            .eq("source_type", "slack")\
            .execute()

        return {"message": f"Deleted {len(doc_ids)} Slack documents and their embeddings", "deleted": len(doc_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup documents: {str(e)}")

@app.get("/api/slack/message-count")
async def get_slack_message_count(connection_id: str = Query(...), days_back: int = 30):
    """
    Check directly with Slack API to count total messages across all channels
    This provides a diagnostic view of what should be synced
    """
    try:
        # Get connection
        connection = supabase.table("connections")\
            .select("*")\
            .eq("id", connection_id)\
            .single()\
            .execute()

        if not connection.data:
            raise HTTPException(status_code=404, detail="Connection not found")

        access_token = connection.data["access_token"]
        org_id = connection.data["org_id"]

        slack = get_slack_connector()

        from datetime import timedelta
        oldest = (datetime.now() - timedelta(days=days_back)).timestamp()

        # Get all channels (public and private)
        public_channels = slack.list_channels(access_token, types="public_channel", auto_join=False)
        private_channels = []
        try:
            private_channels = slack.list_channels(access_token, types="private_channel", auto_join=False)
        except:
            pass

        all_channels = public_channels + private_channels

        channel_stats = []
        total_messages = 0

        from slack_sdk import WebClient
        client = WebClient(token=access_token)

        # Count messages in each channel
        for channel in all_channels:
            try:
                # Get message count for this channel
                response = client.conversations_history(
                    channel=channel["id"],
                    oldest=str(oldest),
                    limit=1  # We just want to see if there are messages
                )

                # Count total by paginating
                count = 0
                cursor = None
                while True:
                    response = client.conversations_history(
                        channel=channel["id"],
                        oldest=str(oldest),
                        limit=200,
                        cursor=cursor
                    )

                    messages = response.get("messages", [])
                    # Filter out bot messages and system messages
                    real_messages = [m for m in messages if not m.get("subtype") and not m.get("bot_id")]
                    count += len(real_messages)

                    cursor = response.get("response_metadata", {}).get("next_cursor")
                    if not cursor:
                        break

                channel_stats.append({
                    "channel_id": channel["id"],
                    "channel_name": channel["name"],
                    "is_private": channel.get("is_private", False),
                    "message_count": count
                })
                total_messages += count

            except Exception as e:
                channel_stats.append({
                    "channel_id": channel["id"],
                    "channel_name": channel["name"],
                    "is_private": channel.get("is_private", False),
                    "message_count": 0,
                    "error": str(e)
                })

        # Get current document count from database
        db_docs = supabase.table("documents")\
            .select("id", count="exact")\
            .eq("org_id", org_id)\
            .eq("source_type", "slack")\
            .execute()

        db_count = len(db_docs.data) if db_docs.data else 0

        return {
            "slack_total_messages": total_messages,
            "slack_total_channels": len(all_channels),
            "slack_public_channels": len(public_channels),
            "slack_private_channels": len(private_channels),
            "database_document_count": db_count,
            "difference": total_messages - db_count,
            "channels": channel_stats,
            "days_back": days_back
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to count Slack messages: {str(e)}")

@app.post("/api/admin/run-migration")
async def run_migration(migration_name: str = Query(...)):
    """
    Run a SQL migration file
    """
    try:
        import os
        migration_file = f"sql/{migration_name}.sql"

        if not os.path.exists(migration_file):
            raise HTTPException(status_code=404, detail=f"Migration file {migration_name}.sql not found")

        with open(migration_file, 'r') as f:
            sql_content = f.read()

        # Split by semicolon and execute each statement
        statements = [s.strip() for s in sql_content.split(';') if s.strip()]

        results = []
        for stmt in statements:
            if stmt:
                try:
                    # Use raw SQL execution (note: supabase-py doesn't support this directly)
                    # We'll need to create the table manually or use a different approach
                    results.append({"statement": stmt[:100] + "...", "status": "executed"})
                except Exception as e:
                    results.append({"statement": stmt[:100] + "...", "status": "failed", "error": str(e)})

        return {
            "migration": migration_name,
            "message": "Migration file loaded. Please run via Supabase SQL Editor",
            "file_path": migration_file,
            "statements_found": len(statements)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")

@app.get("/api/embeddings/stats")
async def get_embedding_stats(org_id: str = Query(...)):
    """
    Get statistics about embeddings for an organization
    """
    try:
        # Count documents
        docs = supabase.table("documents")\
            .select("id, embedding_status", count="exact")\
            .eq("org_id", org_id)\
            .execute()

        total_docs = len(docs.data) if docs.data else 0
        completed_docs = len([d for d in docs.data if d.get("embedding_status") == "completed"]) if docs.data else 0

        # Count embeddings and get full structure
        embeddings = supabase.table("embeddings")\
            .select("*")\
            .eq("org_id", org_id)\
            .limit(3)\
            .execute()

        total_embeddings_count = supabase.table("embeddings")\
            .select("id", count="exact")\
            .eq("org_id", org_id)\
            .execute()

        total_embeddings = len(total_embeddings_count.data) if total_embeddings_count.data else 0

        # Sample a few embeddings to check structure
        sample_embeddings = []
        if embeddings.data and len(embeddings.data) > 0:
            for emb in embeddings.data:
                embedding_field = emb.get("embedding")
                emb_info = {
                    "id": emb["id"],
                    "document_id": emb["document_id"],
                    "content_length": len(emb.get("content", "")) if emb.get("content") else 0,
                    "has_embedding_field": embedding_field is not None,
                }

                if embedding_field:
                    if isinstance(embedding_field, list):
                        emb_info["embedding_type"] = "list"
                        emb_info["embedding_dimension"] = len(embedding_field)
                    elif isinstance(embedding_field, str):
                        emb_info["embedding_type"] = "string"
                        emb_info["embedding_length"] = len(embedding_field)
                    else:
                        emb_info["embedding_type"] = str(type(embedding_field))

                sample_embeddings.append(emb_info)

        return {
            "org_id": org_id,
            "total_documents": total_docs,
            "documents_with_completed_status": completed_docs,
            "total_embeddings": total_embeddings,
            "sample_embeddings": sample_embeddings
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embedding stats: {str(e)}")

@app.get("/api/admin/memory")
async def get_memories(
    org_id: str = Query(...),
    user_id: str = Query(None),
    limit: int = Query(10)
):
    """
    Get AI memories for an organization
    """
    try:
        query = (
            supabase.table("ai_memory")
            .select("*")
            .eq("org_id", org_id)
            .order("importance", desc=True)
            .order("created_at", desc=True)
            .limit(limit)
        )

        if user_id:
            query = query.eq("user_id", user_id)

        result = query.execute()

        return {
            "org_id": org_id,
            "user_id": user_id,
            "count": len(result.data) if result.data else 0,
            "memories": result.data if result.data else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {str(e)}")

# ============================================
# SLACK EVENTS WEBHOOK (PHASE 2)
# ============================================

def verify_slack_signature(
    signing_secret: str,
    timestamp: str,
    body: bytes,
    signature: str
) -> bool:
    """
    Verify Slack request signature for security
    https://api.slack.com/authentication/verifying-requests-from-slack
    """
    # Prevent replay attacks (request must be within 5 minutes)
    if abs(time.time() - int(timestamp)) > 60 * 5:
        print("[SLACK SECURITY] Request timestamp too old")
        return False

    # Compute the signature
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    computed_signature = 'v0=' + hmac.new(
        signing_secret.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()

    # Compare signatures using constant-time comparison
    return hmac.compare_digest(computed_signature, signature)


@app.post("/api/slack/events")
async def slack_events(
    request: Request,
    x_slack_signature: str = Header(None),
    x_slack_request_timestamp: str = Header(None)
):
    """
    Handle Slack Events API webhook for real-time message sync
    Phase 2 - Step 1: Real-time sync with signature verification
    """
    try:
        # Read raw body for signature verification
        body = await request.body()
        body_json = await request.json()

        # Verify Slack signature if signing secret is configured
        slack_signing_secret = os.getenv("SLACK_SIGNING_SECRET")
        if slack_signing_secret and x_slack_signature and x_slack_request_timestamp:
            if not verify_slack_signature(
                slack_signing_secret,
                x_slack_request_timestamp,
                body,
                x_slack_signature
            ):
                print("[SLACK SECURITY] Invalid signature")
                raise HTTPException(status_code=401, detail="Invalid signature")
        elif slack_signing_secret:
            print("[SLACK SECURITY WARNING] Signing secret configured but headers missing")

        # Handle Slack URL verification challenge
        if body_json.get("type") == "url_verification":
            print("[SLACK EVENTS] Handling URL verification challenge")
            return {"challenge": body_json.get("challenge")}

        # Handle event callbacks
        if body_json.get("type") == "event_callback":
            event = body_json.get("event", {})
            event_type = event.get("type")

            print(f"[SLACK EVENTS] Received event: {event_type}")

            # Handle new message events
            if event_type == "message" and not event.get("subtype"):
                # Get team_id to find the connection
                team_id = body_json.get("team_id")

                # Find the connection for this workspace
                connection_result = supabase.table("connections")\
                    .select("id, org_id, access_token")\
                    .eq("workspace_id", team_id)\
                    .eq("source_type", "slack")\
                    .eq("status", "active")\
                    .single()\
                    .execute()

                if not connection_result.data:
                    print(f"[SLACK EVENTS] No active connection found for team {team_id}")
                    return {"ok": True}

                connection = connection_result.data

                # Process the event asynchronously
                ingestion = get_ingestion_service(supabase)
                await ingestion.handle_slack_event(
                    event=event,
                    org_id=connection["org_id"],
                    connection_id=connection["id"],
                    access_token=connection["access_token"]
                )

                print(f"[SLACK EVENTS] Successfully processed message event")
                return {"ok": True}

            else:
                print(f"[SLACK EVENTS] Ignoring event type: {event_type}")
                return {"ok": True}

        # Return OK for other event types
        return {"ok": True}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[SLACK EVENTS] Error processing event: {e}")
        # Always return 200 to Slack to avoid retries
        return {"ok": True}

# ============================================
# STARTUP & SHUTDOWN
# ============================================

# ============================================
# KNOWLEDGE GRAPH ROUTES
# ============================================

@app.post("/api/kg/build/{org_id}")
async def build_knowledge_graph(org_id: str, limit: int = Query(default=100, ge=1, le=1000)):
    """
    Build or rebuild knowledge graph for an organization
    Processes existing documents to extract entities and relationships
    """
    try:
        from services.kg_service import get_kg_service
        kg_service = get_kg_service(supabase)

        # Get documents for this org
        docs_result = supabase.table("documents")\
            .select("id, content, author, author_id, channel_name, channel_id, source_type")\
            .eq("org_id", org_id)\
            .eq("embedding_status", "completed")\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()

        if not docs_result.data:
            return {
                "success": True,
                "message": "No documents found to process",
                "processed": 0
            }

        total_entities = 0
        total_relations = 0

        for doc in docs_result.data:
            metadata = {
                "author": doc.get("author"),
                "author_id": doc.get("author_id"),
                "channel_name": doc.get("channel_name"),
                "channel_id": doc.get("channel_id"),
                "source_type": doc.get("source_type")
            }

            result = kg_service.build_kg_from_document(
                org_id=org_id,
                document_id=doc["id"],
                content=doc["content"],
                metadata=metadata
            )

            total_entities += result["entities_created"]
            total_relations += result["relations_created"]

        return {
            "success": True,
            "message": f"Processed {len(docs_result.data)} documents",
            "documents_processed": len(docs_result.data),
            "entities_created": total_entities,
            "relations_created": total_relations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KG build failed: {str(e)}")

@app.get("/api/kg/query/{org_id}")
async def query_knowledge_graph(
    org_id: str,
    entity: str = Query(..., description="Entity name to query"),
    relation: str = Query(None, description="Optional relation type filter")
):
    """
    Query relationships for a specific entity
    """
    try:
        from services.kg_service import get_kg_service
        kg_service = get_kg_service(supabase)

        results = kg_service.query_related_entities(org_id, entity, relation)

        return {
            "entity": entity,
            "relation_filter": relation,
            "relationships": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KG query failed: {str(e)}")

@app.get("/api/kg/stats/{org_id}")
async def get_kg_stats(org_id: str):
    """
    Get knowledge graph statistics for an organization
    """
    try:
        from services.kg_service import get_kg_service
        kg_service = get_kg_service(supabase)

        stats = kg_service.get_kg_stats(org_id)

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KG stats: {str(e)}")

@app.get("/api/kg/entities/{org_id}")
async def get_entities(
    org_id: str,
    entity_type: str = Query(None, description="Filter by entity type"),
    limit: int = Query(default=50, ge=1, le=200)
):
    """
    Get entities from the knowledge graph
    """
    try:
        query = supabase.table("kg_entities")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)

        if entity_type:
            query = query.eq("type", entity_type)

        result = query.execute()

        return {
            "entities": result.data if result.data else [],
            "count": len(result.data) if result.data else 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entities: {str(e)}")

@app.get("/api/kg/edges/{org_id}")
async def get_edges(
    org_id: str,
    relation: str = Query(None, description="Filter by relation type"),
    limit: int = Query(default=50, ge=1, le=200)
):
    """
    Get relationship edges from the knowledge graph
    """
    try:
        query = supabase.table("kg_edges")\
            .select("*, source:kg_entities!kg_edges_source_id_fkey(name, type), target:kg_entities!kg_edges_target_id_fkey(name, type)")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)

        if relation:
            query = query.eq("relation", relation)

        result = query.execute()

        return {
            "edges": result.data if result.data else [],
            "count": len(result.data) if result.data else 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get edges: {str(e)}")

# ============================================
# AI INSIGHTS ENDPOINTS
# ============================================

@app.post("/api/insights/run/{org_id}")
async def generate_insights(
    org_id: str,
    days: int = Query(default=7, ge=1, le=30, description="Number of days to analyze")
):
    """
    Generate AI insights for an organization

    Analyzes recent KG activity, memory summaries, and documents to generate
    actionable insights about projects, teams, trends, and risks.
    """
    try:
        from services.insight_service import get_insight_service
        insight_service = get_insight_service(supabase)

        result = insight_service.generate_insights(org_id, days)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

@app.get("/api/insights/list/{org_id}")
async def list_insights(
    org_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of insights to return"),
    category: Optional[str] = Query(default=None, description="Filter by category: project, team, trend, risk, general")
):
    """
    List recent AI insights for an organization

    Returns insights sorted by creation date (most recent first).
    Optionally filter by category.
    """
    try:
        from services.insight_service import get_insight_service
        insight_service = get_insight_service(supabase)

        insights = insight_service.list_insights(org_id, limit, category)

        return {
            "insights": insights,
            "count": len(insights)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list insights: {str(e)}")

@app.get("/api/insights/stats/{org_id}")
async def get_insight_stats(org_id: str):
    """
    Get insight statistics for an organization

    Returns counts by category, average confidence, and last generation time.
    """
    try:
        from services.insight_service import get_insight_service
        insight_service = get_insight_service(supabase)

        stats = insight_service.get_insight_stats(org_id)

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insight stats: {str(e)}")

# ============================================
# CLICKUP INTEGRATION ROUTES
# ============================================

@app.get("/api/clickup/connect")
async def clickup_connect(org_id: str = Query(...)):
    """
    Initiate ClickUp OAuth 2.0 flow

    Redirects user to ClickUp authorization page.
    """
    try:
        from services.clickup_service import get_clickup_service
        clickup_service = get_clickup_service(supabase)

        # Generate state parameter (org_id for simplicity, should use JWT in production)
        state = org_id

        # Get authorization URL
        auth_url = clickup_service.get_authorization_url(state)

        return RedirectResponse(url=auth_url)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate ClickUp OAuth: {str(e)}")


@app.get("/api/clickup/callback")
async def clickup_callback(
    code: str = Query(...),
    state: str = Query(...)
):
    """
    ClickUp OAuth callback endpoint

    Exchanges authorization code for access token and stores in connections table.
    """
    try:
        from services.clickup_service import get_clickup_service
        clickup_service = get_clickup_service(supabase)

        org_id = state  # Extract org_id from state parameter

        # Exchange code for access token
        token_data = clickup_service.exchange_code_for_token(code)
        access_token = token_data.get("access_token")

        if not access_token:
            raise HTTPException(status_code=400, detail="Failed to obtain access token")

        # Get authorized user info
        user_info = clickup_service.get_authorized_user(access_token)

        # Get user's workspaces
        workspaces = clickup_service.get_workspaces(access_token)

        if not workspaces:
            raise HTTPException(status_code=400, detail="No ClickUp workspaces found")

        # Use first workspace (or let user select in production)
        workspace = workspaces[0]
        workspace_id = workspace.get("id")
        workspace_name = workspace.get("name")

        # Store connection in Supabase
        connection_data = {
            "org_id": org_id,
            "source_type": "clickup",
            "access_token": access_token,
            "workspace_id": workspace_id,
            "workspace_name": workspace_name,
            "metadata": {
                "user_id": user_info.get("user", {}).get("id"),
                "user_email": user_info.get("user", {}).get("email"),
                "workspace": workspace
            }
        }

        result = supabase.table("connections").insert(connection_data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to store connection")

        connection_id = result.data[0]["id"]

        # Redirect to frontend integrations page with success message
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        return RedirectResponse(url=f"{frontend_url}/dashboard/integrations?clickup=connected")

    except Exception as e:
        # Redirect to frontend with error
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        return RedirectResponse(url=f"{frontend_url}/dashboard/integrations?clickup=error&message={str(e)}")


@app.post("/api/clickup/sync/{connection_id}")
async def clickup_sync(connection_id: str, org_id: str = Query(...)):
    """
    Manually trigger ClickUp data sync

    Fetches all tasks and comments from connected ClickUp workspace and ingests them.
    """
    try:
        from services.clickup_service import get_clickup_service
        from services.ingestion_service import get_ingestion_service
        import traceback

        clickup_service = get_clickup_service(supabase)
        ingestion_service = get_ingestion_service(supabase)

        # Fetch all tasks
        print("[SYNC] Fetching all tasks from ClickUp")
        tasks_data = clickup_service.fetch_all_tasks(connection_id, org_id)
        print(f"[SYNC] Fetched {len(tasks_data)} tasks")

        # Normalize and ingest each task
        documents_ingested = 0
        for idx, task_data in enumerate(tasks_data):
            try:
                print(f"[SYNC] Processing task {idx+1}/{len(tasks_data)}")
                document = clickup_service.normalize_task_to_document(task_data, org_id, connection_id)
            except Exception as e:
                print(f"[SYNC] ERROR normalizing task {idx+1}: {type(e).__name__}: {e}")
                print(f"[SYNC] Traceback: {traceback.format_exc()}")
                raise

            # Check if document already exists
            existing = supabase.table("documents")\
                .select("id")\
                .eq("org_id", org_id)\
                .eq("source_type", "clickup")\
                .eq("source_id", document["source_id"])\
                .execute()

            if existing.data:
                # Update existing document
                supabase.table("documents")\
                    .update(document)\
                    .eq("id", existing.data[0]["id"])\
                    .execute()

                # Re-generate embeddings for updated document
                doc_id = existing.data[0]["id"]
                doc_metadata = {
                    "author": document.get("author"),
                    "author_id": document.get("author_id"),
                    "source_type": "clickup",
                    "task_id": document.get("metadata", {}).get("task_id"),
                    "space_name": document.get("metadata", {}).get("space_name"),
                    "list_name": document.get("metadata", {}).get("list_name")
                }
                await ingestion_service._generate_embeddings(doc_id, document["content"], org_id, doc_metadata)
            else:
                # Add embedding_status to document
                document["embedding_status"] = "pending"

                # Insert new document
                result = supabase.table("documents").insert(document).execute()

                if result.data:
                    doc_id = result.data[0]["id"]

                    # Generate embeddings
                    doc_metadata = {
                        "author": document.get("author"),
                        "author_id": document.get("author_id"),
                        "source_type": "clickup",
                        "task_id": document.get("metadata", {}).get("task_id"),
                        "space_name": document.get("metadata", {}).get("space_name"),
                        "list_name": document.get("metadata", {}).get("list_name")
                    }
                    await ingestion_service._generate_embeddings(doc_id, document["content"], org_id, doc_metadata)
                    documents_ingested += 1

        # Update connection last_sync_at
        from datetime import datetime
        supabase.table("connections").update({
            "last_sync_at": datetime.utcnow().isoformat()
        }).eq("id", connection_id).execute()

        # After successful initial sync, set up webhook for real-time updates
        try:
            # Get connection details
            conn_result = supabase.table("connections")\
                .select("*")\
                .eq("id", connection_id)\
                .execute()

            if conn_result.data:
                connection = conn_result.data[0]
                access_token = connection.get("access_token")
                workspace_id = connection.get("workspace_id")

                # Check if webhook already exists
                webhook_exists = connection.get("metadata", {}).get("webhook_id")

                if not webhook_exists and access_token and workspace_id:
                    # Create webhook for real-time updates
                    webhook_url = os.getenv("CLICKUP_WEBHOOK_URL", f"{os.getenv('BACKEND_URL', 'http://localhost:8000')}/api/clickup/webhook")

                    webhook_events = [
                        "taskCreated",
                        "taskUpdated",
                        "taskDeleted",
                        "taskCommentPosted"
                    ]

                    print(f"[SYNC] Setting up webhook at {webhook_url}")
                    webhook_response = clickup_service.create_webhook(
                        access_token=access_token,
                        team_id=workspace_id,
                        endpoint=webhook_url,
                        events=webhook_events
                    )

                    webhook_id = webhook_response.get("id")

                    # Update connection metadata with webhook info
                    supabase.table("connections").update({
                        "metadata": {
                            **connection.get("metadata", {}),
                            "webhook_id": webhook_id,
                            "webhook_active": True,
                            "webhook_url": webhook_url,
                            "webhook_events": webhook_events
                        }
                    }).eq("id", connection_id).execute()

                    print(f"[SYNC] ✓ Webhook created successfully (ID: {webhook_id})")
        except Exception as webhook_error:
            # Don't fail the sync if webhook setup fails
            print(f"[SYNC] Warning: Failed to set up webhook: {webhook_error}")

        return {
            "success": True,
            "message": f"Synced {len(tasks_data)} tasks from ClickUp",
            "documents_ingested": documents_ingested
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ClickUp sync failed: {str(e)}")


@app.post("/api/clickup/webhook")
async def clickup_webhook(request: Request):
    """
    ClickUp webhook endpoint for real-time updates

    Receives events from ClickUp when tasks are created/updated/deleted.
    """
    try:
        from services.clickup_service import get_clickup_service
        from services.ingestion_service import get_ingestion_service

        clickup_service = get_clickup_service(supabase)
        ingestion_service = get_ingestion_service(supabase)

        # Parse webhook payload
        payload = await request.json()

        event = payload.get("event")
        task_id = payload.get("task_id")
        workspace_id = payload.get("workspace_id")

        if not event or not task_id or not workspace_id:
            raise HTTPException(status_code=400, detail="Invalid webhook payload")

        # Find connection for this workspace
        conn_result = supabase.table("connections")\
            .select("*")\
            .eq("source_type", "clickup")\
            .eq("workspace_id", workspace_id)\
            .execute()

        if not conn_result.data:
            return {"success": True, "message": "No active connection for this workspace"}

        connection = conn_result.data[0]
        connection_id = connection["id"]
        org_id = connection["org_id"]
        access_token = connection["access_token"]

        # Handle different event types
        if event in ["taskCreated", "taskUpdated"]:
            # Fetch task details
            task = payload.get("task", {})

            # Fetch comments
            comments = clickup_service.get_task_comments(access_token, task_id)

            # Build task_data structure
            task_data = {
                "task": task,
                "space_name": task.get("space", {}).get("name", "Unknown"),
                "list_name": task.get("list", {}).get("name", "Unknown"),
                "comments": comments
            }

            # Normalize to document
            document = clickup_service.normalize_task_to_document(task_data, org_id, connection_id)

            # Check if document exists
            existing = supabase.table("documents")\
                .select("id")\
                .eq("org_id", org_id)\
                .eq("source_type", "clickup")\
                .eq("source_id", task_id)\
                .execute()

            if existing.data:
                # Update existing
                supabase.table("documents")\
                    .update(document)\
                    .eq("id", existing.data[0]["id"])\
                    .execute()

                # Regenerate embeddings
                doc_id = existing.data[0]["id"]
                doc_metadata = {
                    "author": document.get("author"),
                    "author_id": document.get("author_id"),
                    "source_type": "clickup",
                    "task_id": document.get("metadata", {}).get("task_id"),
                    "space_name": document.get("metadata", {}).get("space_name"),
                    "list_name": document.get("metadata", {}).get("list_name")
                }
                await ingestion_service._generate_embeddings(doc_id, document["content"], org_id, doc_metadata)
            else:
                # Add embedding_status to document
                document["embedding_status"] = "pending"

                # Insert new
                result = supabase.table("documents").insert(document).execute()
                if result.data:
                    doc_id = result.data[0]["id"]
                    doc_metadata = {
                        "author": document.get("author"),
                        "author_id": document.get("author_id"),
                        "source_type": "clickup",
                        "task_id": document.get("metadata", {}).get("task_id"),
                        "space_name": document.get("metadata", {}).get("space_name"),
                        "list_name": document.get("metadata", {}).get("list_name")
                    }
                    await ingestion_service._generate_embeddings(doc_id, document["content"], org_id, doc_metadata)

        elif event == "taskDeleted":
            # Delete document
            supabase.table("documents")\
                .delete()\
                .eq("org_id", org_id)\
                .eq("source_type", "clickup")\
                .eq("source_id", task_id)\
                .execute()

        return {"success": True, "event": event, "task_id": task_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

# ============================================
# ORCHESTRATOR / COGNITIVE CORE
# Phase 3, Step 1
# ============================================

@app.post("/api/orchestrate/query")
async def orchestrate_query(
    request: Request,
    query: str = Query(..., description="User query to process"),
    org_id: str = Query(..., description="Organization ID")
):
    """
    Orchestrate a query through the cognitive core.

    Workflow:
    1. Classify intent (question, task, summary, insight, risk)
    2. Route to appropriate modules (RAG, KG, Insights)
    3. Execute module queries in parallel
    4. Build reasoning chain
    5. Generate final response
    6. Log reasoning steps
    """
    try:
        from services.orchestrator_service import get_orchestrator_service

        # Get user ID from request (if authenticated)
        user_id = None
        try:
            auth_header = request.headers.get("Authorization")
            if auth_header:
                # Extract user from Supabase auth
                token = auth_header.replace("Bearer ", "")
                user_response = supabase.auth.get_user(token)
                if user_response and user_response.user:
                    user_id = str(user_response.user.id)
        except:
            pass  # Continue without user_id

        orchestrator = get_orchestrator_service(supabase)
        result = await orchestrator.orchestrate_query(query, org_id, user_id)

        return result

    except Exception as e:
        import traceback
        print(f"[ORCHESTRATOR] Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")


@app.get("/api/orchestrate/logs")
async def get_reasoning_logs(
    org_id: str = Query(..., description="Organization ID"),
    limit: int = Query(10, description="Number of logs to retrieve")
):
    """
    Get recent reasoning logs for an organization.
    """
    try:
        logs = supabase.table("reasoning_logs")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()

        return {
            "logs": logs.data if logs.data else [],
            "count": len(logs.data) if logs.data else 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")


# ============================================
# STARTUP / SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 50)
    print("Vibodh AI API - Phase 3, Step 1")
    print("=" * 50)
    print(f"Supabase: {SUPABASE_URL}")
    print(f"OpenAI: {'Configured' if os.getenv('OPENAI_API_KEY') else 'NOT configured'}")
    print(f"Groq: {'Configured' if os.getenv('GROQ_API_KEY') else 'NOT configured'}")
    print(f"Slack: {'Configured' if os.getenv('SLACK_CLIENT_ID') else 'NOT configured'}")
    print("=" * 50)
    print("Ready to receive requests!")
    print("API Docs: http://localhost:8000/docs")
    print("Knowledge Graph: ENABLED")
    print("AI Insights: ENABLED")
    print("ClickUp Integration: ENABLED")
    print("Cognitive Core (Orchestrator): ENABLED")
    print("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("Vibodh AI API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
