import os
from typing import List, Dict, Any, Optional
import shutil
from .enhanced_pdf_processor import EnhancedPDFProcessor
from .vector_store import VectorStore
from .llm_service import LLMService

class RAGSystem:
    def __init__(
        self, 
        pdf_dir: str = "data/pdfs", 
        vector_db_dir: str = "data/vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "google/flan-t5-small"
    ):
        """Initialize the RAG system with all components."""
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
        self.llm_service = LLMService(model_id=llm_model)
    
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
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the current system state."""
        try:
            pdf_count = len([f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')])
        except:
            pdf_count = 0
        
        return {
            "pdf_directory": self.pdf_dir,
            "vector_db_directory": self.vector_db_dir,
            "pdf_files_count": pdf_count,
            "chunk_size": getattr(self.pdf_processor, 'chunk_size', 600),
            "chunk_overlap": getattr(self.pdf_processor, 'chunk_overlap', 100),
            "document_types_supported": list(self.pdf_processor.DOCUMENT_TYPES.keys())
        }