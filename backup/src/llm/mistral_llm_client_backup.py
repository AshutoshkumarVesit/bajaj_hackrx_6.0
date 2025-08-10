# src/llm/mistral_llm_client.py

import os
from mistralai import Mistral
from typing import List, Dict, Any
import asyncio

class MistralLLMClient:
    _instance = None
    _client = None
    _failed_models = set()  # Track failed models
    _current_model_index = 0
    _reset_counter = 0  # Counter to reset failed models after all are exhausted
    
    # Model hierarchy for fallback strategy - UPDATED FOR AVAILABLE MODELS
    MODELS = [
        "mistral-large-2411",       # Primary: Best accuracy (Research License)
        "mistral-medium-2505",      # Fallback 1: Balanced performance 
        "mistral-small-2503",       # Fallback 2: Fast responses (Apache2)
        "open-mistral-nemo",        # Fallback 3: Multilingual (Apache2)
        "codestral-2501",          # Fallback 4: Code-optimized
        "open-codestral-mamba",    # Fallback 5: Mamba architecture (Apache2)
        "mathstral-7b",            # Fallback 6: Math-specialized (Apache2)
        "ministral-8b-2410",       # Fallback 7: Edge model (Research License)
        "ministral-3b-2410",       # Fallback 8: Smallest edge model
    ]
    
    # Embedding model for semantic tasks (if needed)
    EMBEDDING_MODEL = "mistral-embed"

    def __new__(cls):
        """Ensures only one instance of MistralLLMClient is created (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(MistralLLMClient, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        """Initialize Mistral client with single API key."""
        print("Initializing Mistral LLM client...")
        
        # Get Mistral API key from environment variable
        api_key = os.getenv("MISTRAL_API_KEY")
        
        if not api_key:
            raise ValueError(
                "No MISTRAL_API_KEY found. Set MISTRAL_API_KEY environment variable."
            )
        
        try:
            self._client = Mistral(api_key=api_key)
            print(f"✅ Mistral client initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize Mistral client: {e}")
        
        print(f"🚀 Mistral LLM system initialized")
        print(f"📊 Available models: {len(self.MODELS)}")
        print(f"🥇 Primary model: {self.MODELS[0]}")
        print(f"🔄 Total fallback models: {len(self.MODELS) - 1}")

    def _get_next_model(self):
        """Get the next available model using fallback strategy."""
        if len(self._failed_models) >= len(self.MODELS):
            # All models failed, reset and try again
            print("🔄 All models exhausted, resetting failure tracking...")
            self._failed_models.clear()
            self._current_model_index = 0
            self._reset_counter += 1
            
            # Add exponential backoff delay after full rotation
            if self._reset_counter > 1:
                delay = min(2 ** (self._reset_counter - 1), 30)  # Max 30 seconds
                print(f"⏳ Applying {delay}s backoff delay...")
                return None, delay  # Signal for backoff
        
        # Find next available model
        attempts = 0
        while attempts < len(self.MODELS):
            if self._current_model_index not in self._failed_models:
                model = self.MODELS[self._current_model_index]
                current_index = self._current_model_index
                self._current_model_index = (self._current_model_index + 1) % len(self.MODELS)
                return model, current_index
            
            self._current_model_index = (self._current_model_index + 1) % len(self.MODELS)
            attempts += 1
        
        # If we get here, all models are marked as failed
        raise RuntimeError("All Mistral models are currently unavailable")

    def _mark_model_failed(self, model_index: int):
        """Mark a model as failed."""
        self._failed_models.add(model_index)
        model_name = self.MODELS[model_index]
        print(f"❌ Marked model {model_name} as failed")

    async def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generates an answer to a query using Mistral LLM with model fallback strategy.
        
        Args:
            query (str): The user's question.
            context_chunks (List[str]): A list of retrieved text chunks from the document.

        Returns:
            str: The LLM's generated answer.
        """
        if not self._client:
            raise RuntimeError("No Mistral LLM client available")

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

        # Try models in fallback order
        max_retries = len(self.MODELS)
        
        for attempt in range(max_retries):
            try:
                model_result = self._get_next_model()
                
                # Handle backoff delay
                if model_result[0] is None and isinstance(model_result[1], (int, float)):
                    delay = model_result[1]
                    await asyncio.sleep(delay)
                    continue
                
                model_name, model_index = model_result
                
                print(f"🔄 Attempting model {model_name} (attempt {attempt + 1}/{max_retries})")
                
                # Create chat completion with Mistral API
                response = await self._client.chat.complete_async(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,  # Deterministic for consistency
                    max_tokens=150,   # Reduced for faster responses
                    top_p=1.0,
                )
                
                if response.choices and response.choices[0].message.content:
                    answer = response.choices[0].message.content.strip()
                    print(f"✅ Success with model {model_name}")
                    return answer
                else:
                    print(f"⚠️ Empty response from model {model_name}")
                    self._mark_model_failed(model_index)
                    continue
                    
            except Exception as e:
                print(f"❌ Error with model {model_name if 'model_name' in locals() else 'unknown'}: {str(e)}")
                
                # Mark model as failed for specific errors
                error_str = str(e).lower()
                if any(term in error_str for term in [
                    'rate limit', 'quota', 'limit exceeded', 'too many requests',
                    'model not found', 'invalid model', 'unavailable', 'overloaded'
                ]):
                    if 'model_index' in locals():
                        self._mark_model_failed(model_index)
                
                # Add small delay before retry
                await asyncio.sleep(0.5)
                continue

        # If we get here, all models failed
        return "Error: All Mistral models are currently unavailable. Please try again later."

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Mistral's embedding model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            print(f"🔄 Generating embeddings with {self.EMBEDDING_MODEL}")
            
            response = await self._client.embeddings.create_async(
                model=self.EMBEDDING_MODEL,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            print(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration."""
        return {
            "total_models": len(self.MODELS),
            "failed_models": len(self._failed_models),
            "current_primary": self.MODELS[0],
            "available_models": [model for i, model in enumerate(self.MODELS) if i not in self._failed_models],
            "failed_model_names": [self.MODELS[i] for i in self._failed_models],
            "reset_counter": self._reset_counter
        }
