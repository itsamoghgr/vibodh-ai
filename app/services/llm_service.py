# -*- coding: utf-8 -*-
"""
LLM Service
Handles chat completions using Groq API (Llama 3.1)
"""

from app.core.config import settings
from typing import List, Dict, Any, Optional
from groq import Groq


class LLMService:
    def __init__(self):
        """Initialize Groq LLM service"""
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY must be set")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"  # Fast and capable
        self.max_tokens = 2048
        self.temperature = 0.7

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a chat completion response

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Generated response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=False
        )

        return response.choices[0].message.content

    def generate_answer_from_context(
        self,
        query: str,
        context_items: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate an answer to a query using provided context (RAG)

        Args:
            query: User's question
            context_items: List of relevant document chunks
            system_prompt: Optional custom system prompt

        Returns:
            Generated answer
        """
        # Format context
        context_text = "\n\n".join([
            f"[Source {i+1}] {item['content']}"
            for i, item in enumerate(context_items)
        ])

        # Default system prompt
        if not system_prompt:
            system_prompt = """You are Vibodh, an AI assistant that helps answer questions about a company's internal knowledge.

Your task is to answer questions based on the provided context from the company's Slack messages, documents, and other sources.

Guidelines:
- Always base your answer on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite sources when possible (e.g., "According to the Slack message...")
- Be concise but thorough
- If you're uncertain, express appropriate uncertainty
- Use a professional and helpful tone"""

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Context from company knowledge base:

{context_text}

---

Question: {query}

Please provide a helpful answer based on the context above."""}
        ]

        return self.generate_response(messages)

    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ):
        """
        Generate a streaming chat completion response (async)

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override

        Yields:
            Response chunks as they arrive
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Singleton instance
_llm_service = None


def get_llm_service() -> LLMService:
    """Get or create LLM service singleton"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
