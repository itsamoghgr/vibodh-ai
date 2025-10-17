# -*- coding: utf-8 -*-
"""
Vibodh AI - FastAPI Backend
Phase 1 - Step 2: Data Ingestion + Embeddings + RAG
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from datetime import datetime

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
# EMBEDDINGS & SEARCH ROUTES
# ============================================

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
            threshold=0.7
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
        if not session_id:
            # Create new session
            session_data = {
                "org_id": request.org_id,
                "user_id": request.user_id,
                "title": request.query[:100]  # Use first 100 chars of query as title
            }
            session = supabase.table("chat_sessions").insert(session_data).execute()
            session_id = session.data[0]["id"]

        # Store user message
        user_message = {
            "session_id": session_id,
            "role": "user",
            "content": request.query
        }
        supabase.table("chat_messages").insert(user_message).execute()

        # Get context for metadata
        rag_service_instance = get_rag_service(supabase)
        context_items = rag_service_instance.retrieve_context(
            query=request.query,
            org_id=request.org_id,
            limit=request.max_context_items
        )

        # Generate streaming response
        async def generate():
            import json
            full_response = ""

            # Send session_id first
            yield f"data: {json.dumps({'session_id': session_id, 'type': 'session'})}\n\n"

            # Send context
            yield f"data: {json.dumps({'type': 'context', 'context': context_items})}\n\n"

            # Stream answer
            async for chunk in rag_service.generate_answer_stream(
                query=request.query,
                org_id=request.org_id,
                max_context_items=request.max_context_items
            ):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            # Store assistant message
            assistant_message = {
                "session_id": session_id,
                "role": "assistant",
                "content": full_response,
                "context": context_items
            }
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

# ============================================
# STARTUP & SHUTDOWN
# ============================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 50)
    print("Vibodh AI API - Phase 1, Step 2")
    print("=" * 50)
    print(f"Supabase: {SUPABASE_URL}")
    print(f"OpenAI: {'Configured' if os.getenv('OPENAI_API_KEY') else 'NOT configured'}")
    print(f"Groq: {'Configured' if os.getenv('GROQ_API_KEY') else 'NOT configured'}")
    print(f"Slack: {'Configured' if os.getenv('SLACK_CLIENT_ID') else 'NOT configured'}")
    print("=" * 50)
    print("Ready to receive requests!")
    print("API Docs: http://localhost:8000/docs")
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
