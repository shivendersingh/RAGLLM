from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

# Using transformers for LLM
from transformers import pipeline

class LLMService:
    def __init__(self, model_id: str = "google/flan-t5-small"):
        """Initialize the LLM service with a model."""
        self.model = pipeline("text2text-generation", model=model_id, max_length=512)
    
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