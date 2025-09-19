from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import requests
import json
import os

# Using transformers for LLM
from transformers import pipeline

class LLMService:
    """Local LLM service using Transformers pipeline"""
    
    def __init__(self, model_id: str = "google/flan-t5-small"):
        """Initialize the LLM service with a local model."""
        print(f"Initializing local LLM service with model: {model_id}")
        self.model = pipeline("text2text-generation", model=model_id, max_length=512)
        self.service_type = "local"
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate a response based on the query and context."""
        prompt = f"""
        You are a helpful assistant.

        Your task is to answer the user's question based on the provided context.
        Use only the information in the context to answer. If the answer is not in the
        context, say "I don't have enough information to answer this question."

        Context:
        {context}

        User Query: {query}

        Answer:"""
        
        # Generate response
        response = self.model(prompt, max_length=512)[0]["generated_text"]
        return response

class DeepSeekLLMService:
    """Service for generating responses using DeepSeek API"""
    
    def __init__(
        self, 
        api_key: str,
        model_name: str = "deepseek-chat",
        api_base: str = "https://api.deepseek.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """Initialize the DeepSeek LLM service.
        
        Args:
            api_key: DeepSeek API key
            model_name: Model identifier (e.g., "deepseek-chat", "deepseek-coder")
            api_base: Base URL for the API
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.service_type = "deepseek"
        
        # Prepare headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        print(f"Initializing DeepSeek LLM service with model: {model_name}")
        
        # Test API connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test the API connection."""
        try:
            # Make a simple test call
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 10
            }
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ DeepSeek API connection successful")
                return True
            else:
                print(f"⚠️ DeepSeek API test failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"⚠️ DeepSeek API connection test failed: {str(e)}")
            return False
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate a response based on the query and context using DeepSeek API.
        
        Args:
            query: User's question
            context: Retrieved context to ground the response
            
        Returns:
            Generated response text
        """
        prompt = self._create_prompt(query, context)
        
        try:
            # Prepare the API request
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that answers questions based on the provided context. Always be accurate and cite information from the context when possible."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            # Make the API call
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                
                # Extract the response content
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    return "Error: No response generated"
            else:
                error_msg = f"DeepSeek API Error: {response.status_code}"
                try:
                    error_detail = response.json()
                    if "error" in error_detail:
                        error_msg += f" - {error_detail['error'].get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text}"
                
                print(error_msg)
                return f"Error generating response: {response.status_code}"
                
        except requests.exceptions.Timeout:
            error_msg = "DeepSeek API request timed out"
            print(error_msg)
            return "Error: Request timed out"
            
        except requests.exceptions.ConnectionError:
            error_msg = "Failed to connect to DeepSeek API"
            print(error_msg)
            return "Error: Connection failed"
            
        except Exception as e:
            error_msg = f"Error calling DeepSeek API: {str(e)}"
            print(error_msg)
            return f"Error: {str(e)}"
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the DeepSeek model.
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        return f"""Please answer the following question based ONLY on the provided context. 
Be accurate and specific. If the answer cannot be determined from the context, respond with "I don't have enough information to answer this question."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

def create_llm_service(
    deepseek_api_key: Optional[str] = None,
    deepseek_model: str = "deepseek-chat",
    local_model: str = "google/flan-t5-small",
    temperature: float = 0.7,
    max_tokens: int = 1024
):
    """Factory function to create the appropriate LLM service.
    
    Args:
        deepseek_api_key: DeepSeek API key (if None, uses local model)
        deepseek_model: DeepSeek model name
        local_model: Local model name for fallback
        temperature: Temperature for DeepSeek model
        max_tokens: Max tokens for DeepSeek model
        
    Returns:
        LLM service instance (DeepSeek or local)
    """
    # Try to get API key from environment if not provided
    if not deepseek_api_key:
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if deepseek_api_key:
        try:
            return DeepSeekLLMService(
                api_key=deepseek_api_key,
                model_name=deepseek_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"Failed to initialize DeepSeek service: {str(e)}")
            print("Falling back to local LLM service...")
            return LLMService(model_id=local_model)
    else:
        print("No DeepSeek API key provided. Using local LLM service.")
        return LLMService(model_id=local_model)