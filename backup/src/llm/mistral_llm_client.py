# src/llm/mistral_llm_client.py
import os
import requests
import json
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class MistralLLMClient:
    """
    Mistral AI API client for text generation
    Optimized for multilingual document Q&A
    """
    
    def __init__(self):
        """Initialize Mistral client with API key and model configuration"""
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        
        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Model hierarchy (best to fallback)
        self.models = [
            "mistral-large-latest",      # Best for complex reasoning
            "mistral-medium-latest",     # Good balance
            "mistral-small-latest",      # Fast and efficient
            "open-mistral-7b"           # Fastest fallback
        ]
        self.current_model_index = 0
        
        logger.info("✅ Mistral LLM client initialized")
        logger.info(f"🎯 Primary model: {self.models[0]}")
    
    def _get_current_model(self) -> str:
        """Get current model for requests"""
        return self.models[self.current_model_index]
    
    def _rotate_model(self):
        """Rotate to next available model on failure"""
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        new_model = self._get_current_model()
        logger.info(f"🔄 Rotated to model: {new_model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
        """
        Generate response using Mistral API with automatic model fallback
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Creativity level (0.1 = focused, 1.0 = creative)
        
        Returns:
            Generated response text
        """
        # Try each model in sequence
        for attempt in range(len(self.models)):
            try:
                model = self._get_current_model()
                
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stream": False
                }
                
                logger.debug(f"🔄 Requesting {model} (attempt {attempt + 1})")
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=45  # Increased timeout for better reliability
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    logger.info(f"✅ Response generated using {model}")
                    return content.strip()
                
                elif response.status_code == 429:
                    # Rate limit - wait and try next model
                    logger.warning(f"⚠️ Rate limit hit for {model}")
                    time.sleep(2)
                    self._rotate_model()
                    continue
                
                elif response.status_code == 400:
                    # Bad request - try next model
                    logger.warning(f"⚠️ Bad request for {model}: {response.text[:200]}")
                    self._rotate_model()
                    continue
                
                elif response.status_code in [401, 403]:
                    # Auth error - fatal
                    logger.error(f"❌ Authentication error: {response.text}")
                    break
                
                else:
                    # Other HTTP error - try next model
                    logger.warning(f"⚠️ HTTP {response.status_code} for {model}")
                    self._rotate_model()
                    continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Timeout for {model}")
                self._rotate_model()
                continue
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"🔌 Connection error for {model}")
                self._rotate_model()
                continue
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"🌐 Request error for {model}: {e}")
                self._rotate_model()
                continue
                
            except json.JSONDecodeError as e:
                logger.warning(f"📄 JSON decode error for {model}: {e}")
                self._rotate_model()
                continue
                
            except Exception as e:
                logger.error(f"❌ Unexpected error for {model}: {e}")
                self._rotate_model()
                continue
        
        # If all models failed
        logger.error("❌ All Mistral models failed")
        return "I apologize, but I'm unable to generate a response at the moment due to service limitations. Please try again later."
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """
        Generate answer for a question given relevant contexts
        Optimized for document-based Q&A
        
        Args:
            question: User question
            contexts: List of relevant document chunks
            
        Returns:
            Generated answer based on provided contexts
        """
        try:
            # Prepare context
            if not contexts:
                context_text = "No relevant context available in the document."
            else:
                # Limit context to prevent token overflow
                context_chunks = contexts[:8]  # Use top 8 chunks
                context_text = "\n\n".join(context_chunks)
                
                # Truncate if too long (rough token estimation: 1 token ≈ 4 chars)
                max_context_chars = 6000  # ~1500 tokens for context
                if len(context_text) > max_context_chars:
                    context_text = context_text[:max_context_chars] + "\n\n[Context truncated...]"
            
            # Create optimized prompt for document Q&A
            prompt = f"""You are an expert document analysis assistant. Provide accurate, helpful answers based strictly on the provided document content.

INSTRUCTIONS:
1. Answer based ONLY on the provided document context
2. If the document doesn't contain relevant information, clearly state this
3. Be comprehensive but concise
4. Use specific details and quotes from the document when possible
5. Structure your answer clearly with bullet points or paragraphs as appropriate
6. If the question is in Malayalam, provide a helpful response (you may respond in English if the document content is in English)

DOCUMENT CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

            response = self.generate_response(prompt, max_tokens=1500, temperature=0.1)
            
            # Post-process response
            if not response or len(response.strip()) < 10:
                return "I couldn't generate a comprehensive answer based on the provided document content."
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            return f"I apologize, but I encountered an error while processing your question. Please try again."
    
    def test_connection(self) -> bool:
        """
        Test connection to Mistral API
        Returns True if successful, False otherwise
        """
        try:
            test_response = self.generate_response(
                "Respond with 'Connection test successful'", 
                max_tokens=10, 
                temperature=0
            )
            success = "successful" in test_response.lower()
            
            if success:
                logger.info("✅ Mistral API connection test passed")
            else:
                logger.warning("⚠️ Mistral API connection test failed")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Mistral API connection test failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about current model configuration"""
        return {
            "current_model": self._get_current_model(),
            "available_models": self.models,
            "model_index": self.current_model_index,
            "api_base": self.base_url
        }
