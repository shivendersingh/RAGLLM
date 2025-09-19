import os
from typing import List
from langchain_core.documents import Document

# Try to use the new imports first, fall back to old ones if not available
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

class VectorStore:
    def __init__(self, persist_directory: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the vector store with the given embedding model."""
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"}
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        self.vectorstore.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search for the query."""
        return self.vectorstore.similarity_search(query, k=k)