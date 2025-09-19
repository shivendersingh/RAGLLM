import os
from typing import List, Dict, Any, Optional
import shutil
from .enhanced_pdf_processor import EnhancedPDFProcessor
from .vector_store import VectorStore
from .llm_service import LLMService, create_llm_service

class RAGSystem:
    def __init__(
        self, 
        pdf_dir: str = "data/pdfs", 
        vector_db_dir: str = "data/vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "google/flan-t5-small",
        deepseek_api_key: Optional[str] = None,
        deepseek_model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """Initialize the RAG system with all components.
        
        Args:
            pdf_dir: Directory for PDF storage
            vector_db_dir: Directory for vector database
            embedding_model: Model for generating embeddings
            llm_model: Local LLM model (used if no DeepSeek API key)
            deepseek_api_key: DeepSeek API key (optional)
            deepseek_model: DeepSeek model name
            temperature: Temperature for DeepSeek model
            max_tokens: Maximum tokens for DeepSeek model
        """
        self.pdf_dir = pdf_dir
        self.vector_db_dir = vector_db_dir
        
        # Create directories
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(vector_db_dir, exist_ok=True)
        
        # Initialize components with enhanced PDF processor
        self.pdf_processor = EnhancedPDFProcessor(upload_dir=pdf_dir)
        self.vector_store = VectorStore(
            persist_directory=vector_db_dir,
            embedding_model=embedding_model
        )
        
        # Initialize LLM service (DeepSeek or local)
        self.llm_service = create_llm_service(
            deepseek_api_key=deepseek_api_key,
            deepseek_model=deepseek_model,
            local_model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def process_pdf(self, file_path: str) -> bool:
        """Process a PDF file and add it to the vector store."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                return False
            
            print(f"Processing PDF: {file_path}")
            
            # Check if file is accessible (not locked by another process)
            try:
                with open(file_path, 'rb') as f:
                    f.read(1)  # Try to read one byte to check access
            except PermissionError as e:
                print(f"Cannot access file (it may be open in another application): {str(e)}")
                print("Please close the PDF file and try again.")
                return False
            
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.pdf_dir, filename)
            
            # Only copy if source and destination are different
            if os.path.abspath(file_path) != os.path.abspath(dest_path):
                try:
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied {filename} to {self.pdf_dir}")
                    process_path = dest_path
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not copy file ({str(e)}). Processing from original location.")
                    process_path = file_path
            else:
                process_path = file_path
            
            # Process PDF
            print("Extracting text from PDF...")
            documents = self.pdf_processor.process_pdf(process_path)
            
            if not documents:
                print("Warning: No text content extracted from PDF.")
                return False
            
            print(f"Extracted {len(documents)} text chunks from PDF.")
            
            # Add to vector store
            print("Adding documents to vector store...")
            self.vector_store.add_documents(documents)
            
            print(f"✓ Successfully processed {filename} and added to vector store.")
            return True
        
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            import traceback
            print("Full error details:")
            traceback.print_exc()
            return False
    
    def query(self, query_text: str, k: int = 4) -> Dict[str, Any]:
        """Query the system with a text query."""
        try:
            # Retrieve relevant documents
            results = self.vector_store.similarity_search(query_text, k=k)
            
            # Extract content and prepare context
            context = "\n\n".join([doc.page_content for doc in results])
            
            # Generate response
            response = self.llm_service.generate_response(query_text, context)
            
            return {
                "query": query_text,
                "response": response,
                "source_documents": results
            }
        
        except Exception as e:
            print(f"Error querying system: {str(e)}")
            return {
                "query": query_text,
                "response": f"Error: {str(e)}",
                "source_documents": []
            }
    
    def process_pdf_with_type(self, file_path: str, doc_type: str = "auto") -> bool:
        """Process a PDF file with a specific document type."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                return False
            
            print(f"Processing PDF: {file_path}")
            
            # Check if file is accessible (not locked by another process)
            try:
                with open(file_path, 'rb') as f:
                    f.read(1)  # Try to read one byte to check access
            except PermissionError as e:
                print(f"Cannot access file (it may be open in another application): {str(e)}")
                print("Please close the PDF file and try again.")
                return False
            
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.pdf_dir, filename)
            
            # Only copy if source and destination are different
            if os.path.abspath(file_path) != os.path.abspath(dest_path):
                try:
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied {filename} to {self.pdf_dir}")
                    process_path = dest_path
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not copy file ({str(e)}). Processing from original location.")
                    process_path = file_path
            else:
                process_path = file_path
            
            # Process PDF with specific document type
            print("Extracting text from PDF...")
            documents = self.pdf_processor.process_pdf(process_path, doc_type=doc_type)
            
            if not documents:
                print("Warning: No text content extracted from PDF.")
                return False
            
            print(f"Extracted {len(documents)} text chunks from PDF.")
            
            # Add to vector store
            print("Adding documents to vector store...")
            self.vector_store.add_documents(documents)
            
            print(f"✓ Successfully processed {filename} and added to vector store.")
            return True
        
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            import traceback
            print("Full error details:")
            traceback.print_exc()
            return False
    
    def process_multiple_pdfs(self, pdf_directory: str = None) -> Dict[str, bool]:
        """Process all PDF files in a directory."""
        if pdf_directory is None:
            pdf_directory = self.pdf_dir
        
        results = {}
        
        if not os.path.exists(pdf_directory):
            print(f"Directory {pdf_directory} does not exist.")
            return results
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_directory}")
            return results
        
        print(f"Found {len(pdf_files)} PDF files to process...")
        
        for pdf_file in pdf_files:
            file_path = os.path.join(pdf_directory, pdf_file)
            print(f"\nProcessing {pdf_file}...")
            results[pdf_file] = self.process_pdf(file_path)
        
        successful = sum(1 for success in results.values() if success)
        print(f"\n✓ Successfully processed {successful}/{len(pdf_files)} PDF files.")
        
        return results
    
    def process_pdf_with_collection(self, file_path: str, collection_name: str = "default") -> bool:
        """Process a PDF file and add it to a specific collection in the vector store."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                return False
            
            print(f"Processing PDF: {file_path} for collection: {collection_name}")
            
            # Check file access
            try:
                with open(file_path, 'rb') as f:
                    f.read(1)
            except PermissionError as e:
                print(f"Cannot access file: {str(e)}")
                return False
            
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.pdf_dir, filename)
            
            # Copy file if needed
            if os.path.abspath(file_path) != os.path.abspath(dest_path):
                try:
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied {filename} to {self.pdf_dir}")
                    process_path = dest_path
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not copy file ({str(e)}). Processing from original location.")
                    process_path = file_path
            else:
                process_path = file_path
            
            # Process PDF content
            print("Extracting text from PDF...")
            documents = self.pdf_processor.process_pdf(process_path)
            
            if not documents:
                print("Warning: No text content extracted from PDF.")
                return False
            
            print(f"Extracted {len(documents)} text chunks from PDF.")
            
            # Create or get collection-specific vector store
            vector_store = VectorStore(
                persist_directory=self.vector_db_dir,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                collection_name=collection_name
            )
            
            # Add to vector store
            print("Adding documents to vector store...")
            vector_store.add_documents(documents)
            
            # Update the main vector store reference if this is the default collection
            if collection_name == "default":
                self.vector_store = vector_store
            
            print(f"✓ Successfully processed {filename} and added to collection '{collection_name}'.")
            return True
        
        except Exception as e:
            print(f"Error processing PDF for collection {collection_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def query_collection(self, query_text: str, collection_name: str = "default", k: int = 4) -> Dict[str, Any]:
        """Query a specific collection with a text query."""
        try:
            print(f"Querying collection '{collection_name}' with: {query_text}")
            
            # Get or create collection-specific vector store
            vector_store = VectorStore(
                persist_directory=self.vector_db_dir,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                collection_name=collection_name
            )
            
            # Retrieve relevant documents
            results = vector_store.similarity_search(query_text, k=k)
            
            # Extract content and prepare context
            context = "\n\n".join([doc.page_content for doc in results])
            
            # Generate response
            response = self.llm_service.generate_response(query_text, context)
            
            return {
                "query": query_text,
                "response": response,
                "source_documents": results,
                "collection": collection_name
            }
        
        except Exception as e:
            print(f"Error querying collection {collection_name}: {str(e)}")
            return {
                "query": query_text,
                "response": f"Error: {str(e)}",
                "source_documents": [],
                "collection": collection_name
            }
    
    def process_pdf_replace_collection(self, file_path: str, collection_name: str = "default", use_master_collection: bool = False) -> bool:
        """Process PDF and COMPLETELY replace previous collection content with aggressive purging."""
        try:
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist.")
                return False
            
            print(f"🔄 Processing PDF with COMPLETE replacement for collection '{collection_name}': {file_path}")
            
            # Check file access
            try:
                with open(file_path, 'rb') as f:
                    f.read(1)
            except PermissionError as e:
                print(f"Cannot access file: {str(e)}")
                return False
            
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.pdf_dir, filename)
            
            # Copy file if needed
            if os.path.abspath(file_path) != os.path.abspath(dest_path):
                try:
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied {filename} to {self.pdf_dir}")
                    process_path = dest_path
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not copy file ({str(e)}). Processing from original location.")
                    process_path = file_path
            else:
                process_path = file_path
            
            # Step 1: Force delete old collection if it exists using direct ChromaDB access
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.vector_db_dir)
                collections = [c.name for c in client.list_collections()]
                if collection_name in collections:
                    print(f"🗑️ Forcefully deleting existing collection: {collection_name}")
                    client.delete_collection(collection_name)
                    print(f"✅ Collection {collection_name} completely purged")
            except Exception as e:
                print(f"Warning during force delete: {str(e)}")
            
            # Step 2: Process PDF content
            print("📄 Extracting text from PDF...")
            documents = self.pdf_processor.process_pdf(process_path)
            
            if not documents:
                print("Warning: No text content extracted from PDF.")
                return False
            
            print(f"✅ Extracted {len(documents)} text chunks from PDF.")
            
            # Step 3: Add file metadata to each document for better tracking
            import time
            current_time = time.time()
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['source_file'] = filename
                doc.metadata['upload_timestamp'] = current_time
                doc.metadata['collection'] = collection_name
            
            # Step 4: Create fresh vector store (this will create a new collection)
            vector_store = VectorStore(
                persist_directory=self.vector_db_dir,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                collection_name=collection_name,
                use_master_collection=use_master_collection
            )
            
            # Step 5: Add documents to the fresh collection
            print(f"📦 Adding {len(documents)} documents to fresh collection '{collection_name}'...")
            vector_store.add_documents(documents)
            
            # Step 6: Update the main vector store reference if this is the default collection
            if collection_name == "default":
                self.vector_store = vector_store
            
            print(f"🎉 Successfully REPLACED collection '{collection_name}' with {len(documents)} chunks from {filename}")
            return True
        
        except Exception as e:
            print(f"❌ Error processing PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def clear_collection(self, collection_name: str = "default") -> bool:
        """Clear all documents from a specific collection using aggressive deletion."""
        try:
            print(f"🗑️ Completely clearing collection '{collection_name}'...")
            
            # Create vector store for the collection
            vector_store = VectorStore(
                persist_directory=self.vector_db_dir,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                collection_name=collection_name
            )
            
            # Use aggressive deletion method
            success = vector_store.delete_all_documents()
            
            if success:
                print(f"✅ Successfully purged collection '{collection_name}'")
                # Update main vector store reference if clearing default
                if collection_name == "default":
                    self.vector_store = vector_store
            else:
                print(f"❌ Failed to purge collection '{collection_name}'")
            
            return success
        
        except Exception as e:
            print(f"❌ Error clearing collection {collection_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_collection_info(self, collection_name: str = "default") -> Dict[str, Any]:
        """Get detailed information about a specific collection."""
        try:
            # Create vector store for the collection
            vector_store = VectorStore(
                persist_directory=self.vector_db_dir,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                collection_name=collection_name
            )
            
            return vector_store.get_collection_info()
        
        except Exception as e:
            return {
                'collection_name': collection_name,
                'error': str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the current system state."""
        try:
            pdf_count = len([f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')])
        except:
            pdf_count = 0
        
        # Get collection info if possible
        collections_info = {}
        try:
            import chromadb
            client = chromadb.PersistentClient(self.vector_db_dir)
            collections = client.list_collections()
            for collection in collections:
                try:
                    coll = client.get_collection(collection.name)
                    count = coll.count()
                    collections_info[collection.name] = count
                except Exception as e:
                    collections_info[collection.name] = f"Error: {str(e)}"
        except Exception as e:
            collections_info = {"error": f"Could not access collection information: {str(e)}"}
        
        # Get LLM service information
        llm_info = {
            "service_type": getattr(self.llm_service, 'service_type', 'unknown'),
        }
        
        if hasattr(self.llm_service, 'model_name'):
            llm_info["model_name"] = self.llm_service.model_name
        elif hasattr(self.llm_service, 'model'):
            llm_info["model_name"] = getattr(self.llm_service.model, 'model', 'local_transformer')
        
        if hasattr(self.llm_service, 'temperature'):
            llm_info["temperature"] = self.llm_service.temperature
        if hasattr(self.llm_service, 'max_tokens'):
            llm_info["max_tokens"] = self.llm_service.max_tokens
        
        return {
            "pdf_directory": self.pdf_dir,
            "vector_db_directory": self.vector_db_dir,
            "pdf_files_count": pdf_count,
            "collections": collections_info,
            "chunk_size": getattr(self.pdf_processor, 'chunk_size', 600),
            "chunk_overlap": getattr(self.pdf_processor, 'chunk_overlap', 100),
            "document_types_supported": list(self.pdf_processor.DOCUMENT_TYPES.keys()),
            "llm_service": llm_info
        }