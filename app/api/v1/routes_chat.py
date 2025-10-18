"""
Chat API Routes
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from app.models.legacy_schemas import ChatStreamRequest, ChatSessionResponse, ChatMessageResponse, FeedbackCreate

from app.services import get_rag_service
from app.db import supabase
from app.core.logging import logger

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/stream")
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

            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Chat stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_chat_history(org_id: str, user_id: Optional[str] = None, limit: int = 10):
    """Get chat session history"""
    try:
        query = supabase.table("chat_sessions")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)

        if user_id:
            query = query.eq("user_id", user_id)

        result = query.execute()
        return {"sessions": result.data}

    except Exception as e:
        logger.error(f"Get chat history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: str):
    """Get a specific chat session with messages"""
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
            **session.data,
            "messages": messages.data or []
        }

    except Exception as e:
        logger.error(f"Get chat session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackCreate):
    """Submit feedback for a message"""
    try:
        feedback_data = {
            "message_id": feedback.message_id,
            "rating": feedback.rating
        }

        result = supabase.table("message_feedback").insert(feedback_data).execute()
        return {"success": True, "data": result.data[0]}

    except Exception as e:
        logger.error(f"Submit feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
