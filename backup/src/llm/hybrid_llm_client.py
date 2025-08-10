# src/llm/hybrid_llm_client.py

import os
from groq import AsyncGroq
from mistralai import Mistral
from typing import List, Dict, Any
import asyncio

class HybridLLMClient:
    """
    Hybrid LLM client that uses Groq (free) as primary and Mistral (paid) as fallback.
    Best of both worlds: free tier for most requests, premium quality when needed.
    """
    _instance = None
    _groq_clients = []
    _mistral_client = None
    _failed_groq_clients = set()
    _current_groq_index = 0
    
    # Groq models (FREE)
    GROQ_MODELS = [
        "llama-3.1-8b-instant",    # Fast and free
        "gemma2-9b-it",           # Alternative free model
    ]
    
    # Mistral models (PAID - for premium fallback)
    MISTRAL_MODELS = [
        "open-mistral-nemo",       # Cheapest Mistral option
        "mistral-small-2503",      # Better quality
        "mistral-large-2411",      # Best quality (expensive)
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HybridLLMClient, cls).__new__(cls)
            cls._instance._initialize_clients()
        return cls._instance

    def _initialize_clients(self):
        """Initialize both Groq (free) and Mistral (paid) clients."""
        print("🔄 Initializing Hybrid LLM system (Groq + Mistral)...")
        
        # Initialize Groq clients (FREE)
        groq_keys = []
        for i in range(1, 11):
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key:
                groq_keys.append(key)
        
        if not groq_keys:
            single_key = os.getenv("GROQ_API_KEY")
            if single_key:
                groq_keys.append(single_key)
        
        for i, key in enumerate(groq_keys):
            try:
                client = AsyncGroq(api_key=key)
                self._groq_clients.append(client)
                print(f"✅ Groq client {i+1} initialized (FREE)")
            except Exception as e:
                print(f"❌ Failed to initialize Groq client {i+1}: {e}")
        
        # Initialize Mistral client (PAID)
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key:
            try:
                self._mistral_client = Mistral(api_key=mistral_key)
                print("✅ Mistral client initialized (PAID)")
            except Exception as e:
                print(f"❌ Failed to initialize Mistral client: {e}")
        
        print(f"🚀 Hybrid system ready: {len(self._groq_clients)} Groq (FREE) + {'1' if self._mistral_client else '0'} Mistral (PAID)")

    async def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate answer using hybrid approach:
        1. Try Groq first (FREE)
        2. Fallback to Mistral if Groq fails (PAID)
        """
        context_string = "\n\n".join(context_chunks)
        
        system_prompt = (
            "You are a highly accurate assistant for Bajaj Finserv, specialized "
            "in providing precise answers based on the provided document context. "
            "Use the context to answer questions accurately and concisely. "
            "If specific information is not available in the context, clearly state what information "
            "is missing rather than making assumptions. "
            "Prioritize factual accuracy and provide direct, helpful answers in 1-3 sentences. "
            "Focus on the essential information that directly addresses the user's question."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Answer the question using only the context below.\n\n--- DOCUMENT CONTEXT ---\n{context_string}\n\n--- QUESTION ---\n{query}\n\nYour concise answer:"}
        ]

        # Phase 1: Try Groq (FREE)
        groq_result = await self._try_groq(messages)
        if groq_result:
            print("✅ Response from Groq (FREE)")
            return groq_result

        # Phase 2: Fallback to Mistral (PAID)
        print("🔄 Groq exhausted, trying Mistral (PAID)...")
        mistral_result = await self._try_mistral(messages)
        if mistral_result:
            print("✅ Response from Mistral (PAID)")
            return mistral_result

        return "Error: All LLM services are currently unavailable."

    async def _try_groq(self, messages):
        """Try Groq models (FREE)."""
        if not self._groq_clients:
            return None
            
        for model_name in self.GROQ_MODELS:
            for attempt in range(len(self._groq_clients)):
                try:
                    if self._current_groq_index not in self._failed_groq_clients:
                        client = self._groq_clients[self._current_groq_index]
                        
                        response = await client.chat.completions.create(
                            messages=messages,
                            model=model_name,
                            temperature=0.0,
                            max_tokens=150,
                            top_p=1,
                        )
                        
                        if response.choices and response.choices[0].message.content:
                            return response.choices[0].message.content.strip()
                        
                except Exception as e:
                    print(f"❌ Groq error: {e}")
                    self._failed_groq_clients.add(self._current_groq_index)
                
                self._current_groq_index = (self._current_groq_index + 1) % len(self._groq_clients)
                await asyncio.sleep(0.5)
        
        return None

    async def _try_mistral(self, messages):
        """Try Mistral models (PAID)."""
        if not self._mistral_client:
            return None
            
        for model_name in self.MISTRAL_MODELS:
            try:
                print(f"🔄 Trying Mistral {model_name} (PAID)")
                response = await self._mistral_client.chat.complete_async(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=150,
                    top_p=1.0,
                )
                
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                    
            except Exception as e:
                print(f"❌ Mistral {model_name} error: {e}")
                await asyncio.sleep(0.5)
                continue
        
        return None
