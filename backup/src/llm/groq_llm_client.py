# src/llm/groq_llm_client.py

import os
from groq import AsyncGroq
from typing import List, Dict, Any
import asyncio
import random

class GroqLLMClient:
    _instance = None
    _clients = []  # List of AsyncGroq clients with different API keys
    _current_client_index = 0
    _failed_clients = set()  # Track failed clients
    _reset_counter = 0  # Counter to reset failed clients after all are exhausted
    
    # Primary model for speed and efficiency
    MODEL_NAME = "llama-3.1-8b-instant"
    # Fallback model for better accuracy if needed
    FALLBACK_MODEL = "gemma2-9b-it"

    def __new__(cls):
        """Ensures only one instance of GroqLLMClient is created (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(GroqLLMClient, cls).__new__(cls)
            cls._instance._initialize_clients()
        return cls._instance

    def _initialize_clients(self):
        """Initialize multiple Groq clients with different API keys."""
        print("Initializing Groq LLM clients with API key rotation...")
        
        # Get all GROQ API keys from environment variables
        api_keys = []
        for i in range(1, 11):  # GROQ_API_KEY_1 to GROQ_API_KEY_10
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key:
                api_keys.append(key)
        
        # Fallback to single key if numbered keys not found
        if not api_keys:
            single_key = os.getenv("GROQ_API_KEY")
            if single_key:
                api_keys.append(single_key)
        
        if not api_keys:
            raise ValueError(
                "No GROQ API keys found. Set GROQ_API_KEY_1 through GROQ_API_KEY_10 "
                "or a single GROQ_API_KEY environment variable."
            )
        
        # Initialize AsyncGroq clients
        for i, key in enumerate(api_keys):
            try:
                client = AsyncGroq(api_key=key)
                self._clients.append(client)
                print(f"✅ Groq client {i+1} initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Groq client {i+1}: {e}")
        
        if not self._clients:
            raise ValueError("Failed to initialize any Groq clients")
        
        print(f"🚀 Groq LLM system initialized with {len(self._clients)} API keys")
        print(f"📊 Primary model: {self.MODEL_NAME}")
        print(f"🔄 Fallback model: {self.FALLBACK_MODEL}")

    def _get_next_client(self):
        """Get the next available client using round-robin with failure tracking."""
        if len(self._failed_clients) >= len(self._clients):
            # All clients failed, reset and try again
            print("🔄 All API keys exhausted, resetting failure tracking...")
            self._failed_clients.clear()
            self._reset_counter += 1
            
            # Add exponential backoff delay after full rotation
            if self._reset_counter > 1:
                delay = min(2 ** (self._reset_counter - 1), 60)  # Max 60 seconds
                print(f"⏳ Applying {delay}s backoff delay...")
                # Note: In async context, you might want to use asyncio.sleep(delay)
        
        # Find next available client
        attempts = 0
        while attempts < len(self._clients):
            if self._current_client_index not in self._failed_clients:
                client = self._clients[self._current_client_index]
                current_index = self._current_client_index
                self._current_client_index = (self._current_client_index + 1) % len(self._clients)
                return client, current_index
            
            self._current_client_index = (self._current_client_index + 1) % len(self._clients)
            attempts += 1
        
        # If we get here, all clients are marked as failed
        raise RuntimeError("All Groq API keys are currently unavailable")

    def _mark_client_failed(self, client_index: int):
        """Mark a client as failed."""
        self._failed_clients.add(client_index)
        print(f"❌ Marked Groq API key {client_index + 1} as failed")

    async def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generates an answer to a query using the Groq LLM with API key rotation and fallback.
        
        Args:
            query (str): The user's question.
            context_chunks (List[str]): A list of retrieved text chunks from the document.

        Returns:
            str: The LLM's generated answer.
        """
        if not self._clients:
            raise RuntimeError("No Groq LLM clients available")

        context_string = "\n\n".join(context_chunks)

        # System prompt optimized for Bajaj Finserv domain
        system_prompt = (
            "You are a highly accurate assistant for Bajaj Finserv, specialized "
            "in providing precise answers based on the provided document context. "
            "Use the context to answer questions accurately and concisely. "
            "If specific information is not available in the context, clearly state what information "
            "is missing rather than making assumptions. "
            "Prioritize factual accuracy and provide direct, helpful answers in 1-3 sentences. "
            "Focus on the essential information that directly addresses the user's question."
        )

        user_prompt = (
            f"Answer the question using only the context below. Focus strictly on the question asked.\n\n"
            f"--- DOCUMENT CONTEXT ---\n{context_string}\n\n"
            f"--- QUESTION ---\n{query}\n\n"
            f"Your concise answer:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Try primary model first, then fallback model
        for model_name in [self.MODEL_NAME, self.FALLBACK_MODEL]:
            max_retries = len(self._clients)
            
            for attempt in range(max_retries):
                try:
                    client, client_index = self._get_next_client()
                    
                    print(f"🔄 Attempting API key {client_index + 1} with model {model_name}")
                    
                    chat_completion = await client.chat.completions.create(
                        messages=messages,
                        model=model_name,
                        temperature=0.0,  # Deterministic for consistency
                        max_tokens=150,   # Reduced from 300 for faster responses
                        top_p=1,
                        stop=None,
                        stream=False,
                    )
                    
                    if chat_completion.choices and chat_completion.choices[0].message.content:
                        answer = chat_completion.choices[0].message.content.strip()
                        print(f"✅ Success with API key {client_index + 1}, model {model_name}")
                        return answer
                    else:
                        print(f"⚠️ Empty response from API key {client_index + 1}")
                        continue
                        
                except Exception as e:
                    print(f"❌ Error with API key {client_index + 1}: {str(e)}")
                    
                    # Mark client as failed for rate limit or quota errors
                    error_str = str(e).lower()
                    if any(term in error_str for term in ['rate limit', 'quota', 'limit exceeded', 'too many requests']):
                        self._mark_client_failed(client_index)
                    
                    # Add small delay before retry
                    await asyncio.sleep(0.5)
                    continue
            
            print(f"⚠️ All API keys failed with model {model_name}, trying next model...")

        # If we get here, all models and clients failed
        return "Error: All Groq API keys and models are currently unavailable. Please try again later."
