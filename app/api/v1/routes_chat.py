"""
Chat API Routes
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from app.models.legacy_schemas import ChatStreamRequest, ChatSessionResponse, ChatMessageResponse, FeedbackCreate

from app.services import get_rag_service
from app.services.orchestrator_service import OrchestratorService
from app.db import supabase, supabase_admin
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

        # Create or get session (use admin client to bypass RLS)
        session_id = request.session_id
        if not session_id and request.user_id:
            # Only create session if user_id is provided
            session_data = {
                "org_id": request.org_id,
                "user_id": request.user_id,
                "title": request.query[:100]  # Use first 100 chars of query as title
            }
            session = supabase_admin.table("chat_sessions").insert(session_data).execute()
            session_id = session.data[0]["id"]

        # Get conversation history (last 10 messages)
        conversation_history = []
        if session_id:
            history_result = supabase_admin.table("chat_messages")\
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
            supabase_admin.table("chat_messages").insert(user_message).execute()

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
            import asyncio

            # Send session_id first
            yield f"data: {json.dumps({'session_id': session_id, 'type': 'session'})}\n\n"

            # Use orchestrator for full reasoning and metrics tracking
            orchestrator = OrchestratorService(supabase)

            try:
                logger.info(f"Chat stream: org_id={request.org_id}, user_id={request.user_id}, query={request.query[:50]}")

                # Send thinking indicator
                yield f"data: {json.dumps({'type': 'thinking'})}\n\n"
                await asyncio.sleep(0.1)

                result = await orchestrator.orchestrate_query(
                    query=request.query,
                    org_id=request.org_id,
                    user_id=request.user_id
                )

                logger.info(f"Orchestrator result received, answer length: {len(result['final_answer'])}")

                full_response = result['final_answer']
                context_sources = result.get('context_sources', [])

                # Send context
                if context_sources:
                    yield f"data: {json.dumps({'type': 'context', 'context': context_sources})}\n\n"

                # Check if this is an agent result with an action plan
                module_results = result.get('module_results', {})
                agent_result = module_results.get('agent', {})

                if agent_result.get('plan_created'):
                    # Emit action_plan event with structured data
                    action_plan_event = {
                        'type': 'action_plan',
                        'actionPlan': {
                            'id': agent_result.get('plan_id'),
                            'agentType': agent_result.get('agent_type', 'communication'),
                            'goal': agent_result.get('goal', ''),
                            'steps': [],  # Will be fetched from database if needed
                            'riskLevel': agent_result.get('risk_level', 'low'),
                            'requiresApproval': agent_result.get('requires_approval', False),
                            'status': 'pending_approval' if agent_result.get('requires_approval') else 'approved',
                            'totalSteps': agent_result.get('total_steps', 0),
                            'completedSteps': len(agent_result.get('executed_steps', [])),
                            'executedSteps': agent_result.get('executed_steps', [])
                        }
                    }
                    yield f"data: {json.dumps(action_plan_event)}\n\n"
                    await asyncio.sleep(0.1)

                # Stream the response character by character for better UX
                chunk_size = 5  # Stream 5 characters at a time
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i:i+chunk_size]
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for streaming effect

                # Store assistant message (only if session exists)
                if session_id:
                    assistant_message = {
                        "session_id": session_id,
                        "role": "assistant",
                        "content": full_response,
                    }
                    # Only include context if there are items
                    if context_sources and len(context_sources) > 0:
                        assistant_message["context"] = context_sources

                    # Store message type and action plan data in metadata JSONB field
                    if agent_result.get('plan_created'):
                        assistant_message["metadata"] = {
                            "message_type": "action_plan",
                            "plan_id": agent_result.get('plan_id'),
                            "agent_type": agent_result.get('agent_type'),
                            "goal": agent_result.get('goal'),
                            "risk_level": agent_result.get('risk_level'),
                            "total_steps": agent_result.get('total_steps'),
                            "requires_approval": agent_result.get('requires_approval'),
                            "executed_steps": agent_result.get('executed_steps', [])
                        }
                    else:
                        assistant_message["metadata"] = {"message_type": "text"}

                    supabase_admin.table("chat_messages").insert(assistant_message).execute()

            except Exception as e:
                logger.error(f"Orchestrator error: {e}", exc_info=True)
                # Send error message
                error_msg = "I apologize, but I encountered an error processing your request."
                yield f"data: {json.dumps({'type': 'token', 'content': error_msg})}\n\n"

            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Chat stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_chat_history(org_id: str, user_id: Optional[str] = None, limit: int = 10):
    """Get chat session history with message counts"""
    try:
        # Use admin client to bypass RLS since this is a backend operation
        query = supabase_admin.table("chat_sessions")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)

        if user_id:
            query = query.eq("user_id", user_id)

        result = query.execute()

        # Add message count to each session
        sessions_with_counts = []
        for session in result.data:
            # Get message count for this session
            message_count_result = supabase_admin.table("chat_messages")\
                .select("id", count="exact")\
                .eq("session_id", session["id"])\
                .execute()

            session["message_count"] = message_count_result.count or 0
            sessions_with_counts.append(session)

        return {"sessions": sessions_with_counts}

    except Exception as e:
        logger.error(f"Get chat history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: str):
    """Get a specific chat session with messages"""
    try:
        # Get session (use admin to bypass RLS)
        session = supabase_admin.table("chat_sessions")\
            .select("*")\
            .eq("id", session_id)\
            .single()\
            .execute()

        if not session.data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get messages (use admin to bypass RLS)
        messages = supabase_admin.table("chat_messages")\
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


@router.delete("/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session and all its messages"""
    try:
        # Delete all messages first (cascade should handle this, but being explicit)
        supabase_admin.table("chat_messages")\
            .delete()\
            .eq("session_id", session_id)\
            .execute()

        # Delete the session
        result = supabase_admin.table("chat_sessions")\
            .delete()\
            .eq("id", session_id)\
            .execute()

        return {"success": True, "message": "Session deleted successfully"}

    except Exception as e:
        logger.error(f"Delete chat session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackCreate):
    """Submit feedback for a message"""
    try:
        feedback_data = {
            "message_id": feedback.message_id,
            "rating": feedback.rating
        }

        result = supabase_admin.table("chat_feedback").insert(feedback_data).execute()
        return {"success": True, "data": result.data[0]}

    except Exception as e:
        logger.error(f"Submit feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
