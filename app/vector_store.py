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
    def __init__(self, persist_directory: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 collection_name: str = "default", reset_collection: bool = False,
                 use_master_collection: bool = False):
        """Initialize the vector store with the given embedding model and collection.
        
        Args:
            persist_directory: Directory for storing vector database
            embedding_model: Model name for embeddings
            collection_name: Name of collection to use
            reset_collection: If True, reset the collection before adding documents
            use_master_collection: If True, all documents are also added to the 'master' collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.use_master_collection = use_master_collection
        self.master_collection_name = "master"
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"}
        )
        
        # Handle collection reset if requested
        if reset_collection:
            self._force_delete_collection(collection_name)
        
        # Initialize vector store with collection name
        self.vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=self.embedding_function,
            collection_name=collection_name
        )
        
        # Print collection status for debugging
        try:
            doc_count = len(self.vectorstore._collection.get()['ids']) if hasattr(self.vectorstore, '_collection') else 0
            print(f"Vector store initialized: collection '{collection_name}', documents: {doc_count}")
        except:
            print(f"Vector store initialized: collection '{collection_name}'")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store and optionally sync to master collection."""
        try:
            # Add documents to the primary collection
            self.vectorstore.add_documents(documents)
            
            # Explicitly persist the primary collection data (newer ChromaDB doesn't need explicit persist)
            print(f"✓ Added {len(documents)} documents to collection '{self.collection_name}'")
            
            # Sync to master collection if enabled
            if self.use_master_collection and self.collection_name != self.master_collection_name:
                try:
                    # Add source collection metadata to documents for master collection
                    master_documents = []
                    for doc in documents:
                        # Create a copy of the document with source metadata
                        master_doc = Document(
                            page_content=doc.page_content,
                            metadata={
                                **doc.metadata,
                                'source_collection': self.collection_name
                            }
                        )
                        master_documents.append(master_doc)
                    
                    # Create master collection vector store
                    master_vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embedding_function,
                        collection_name=self.master_collection_name
                    )
                    
                    # Add documents to master collection
                    master_vectorstore.add_documents(master_documents)
                    
                    print(f"✓ Synced {len(master_documents)} documents to master collection '{self.master_collection_name}'")
                    
                except Exception as master_error:
                    print(f"⚠ Warning: Failed to sync to master collection: {str(master_error)}")
                    # Don't raise - master sync is optional
                
        except Exception as e:
            print(f"✗ Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search for the query."""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            print(f"Found {len(results)} similar documents for query in collection '{self.collection_name}'")
            return results
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the collection."""
        try:
            if hasattr(self.vectorstore, '_collection'):
                return len(self.vectorstore._collection.get()['ids'])
            return 0
        except:
            return 0
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            if hasattr(self.vectorstore, '_collection'):
                # Get all document IDs and delete them
                all_docs = self.vectorstore._collection.get()
                if all_docs['ids']:
                    self.vectorstore._collection.delete(ids=all_docs['ids'])
                    print(f"✓ Cleared collection '{self.collection_name}' - removed {len(all_docs['ids'])} documents")
                    return True
                else:
                    print(f"Collection '{self.collection_name}' is already empty")
                    return True
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
            return False
        return False
    
    def _force_delete_collection(self, collection_name: str) -> bool:
        """Internal method to forcefully delete a specific collection."""
        try:
            import chromadb
            
            # Get direct access to ChromaDB client
            client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Check if collection exists before deleting
            collections = [c.name for c in client.list_collections()]
            if collection_name in collections:
                print(f"🔄 Forcefully purging collection: {collection_name}")
                # Delete the collection completely
                client.delete_collection(collection_name)
                print(f"✅ Collection {collection_name} purged successfully")
            
            return True
        except Exception as e:
            print(f"❌ Error forcefully deleting collection {collection_name}: {str(e)}")
            return False

    def force_delete_collection(self) -> bool:
        """Forcefully delete and recreate the collection to ensure complete purging."""
        try:
            if self._force_delete_collection(self.collection_name):
                # Recreate the vectorstore with the new collection
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function,
                    collection_name=self.collection_name
                )
                print(f"✅ Created fresh collection: {self.collection_name}")
                return True
            return False
        except Exception as e:
            print(f"❌ Error forcefully resetting collection: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def reset_and_add_documents(self, documents: List[Document]) -> bool:
        """Clear the collection and add new documents using aggressive purging."""
        try:
            # First forcefully delete and recreate the collection
            if not self.force_delete_collection():
                print("Failed to forcefully reset collection")
                return False
            
            # Then add new documents
            self.add_documents(documents)
            print(f"✓ Successfully reset collection '{self.collection_name}' with {len(documents)} new documents")
            return True
        except Exception as e:
            print(f"Error resetting collection with documents: {str(e)}")
            return False
    
    def get_collection_info(self) -> dict:
        """Get detailed information about the collection."""
        try:
            info = {
                'collection_name': self.collection_name,
                'document_count': self.get_document_count(),
                'persist_directory': self.persist_directory
            }
            
            # Try to get additional metadata
            if hasattr(self.vectorstore, '_collection'):
                try:
                    all_docs = self.vectorstore._collection.get()
                    info['has_documents'] = len(all_docs['ids']) > 0
                    info['sample_metadata'] = all_docs['metadatas'][:3] if all_docs['metadatas'] else []
                except:
                    pass
                    
            return info
        except Exception as e:
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }
    
    def delete_all_documents(self) -> bool:
        """Delete all documents from the collection completely."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Check if collection exists
            collections = [c.name for c in client.list_collections()]
            if self.collection_name in collections:
                # Delete and recreate
                client.delete_collection(self.collection_name)
                client.create_collection(name=self.collection_name)
                print(f"✅ All documents deleted from collection {self.collection_name}")
                
                # Recreate vectorstore reference
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function,
                    collection_name=self.collection_name
                )
                return True
            else:
                print(f"⚠️ Collection {self.collection_name} does not exist")
                return False
        except Exception as e:
            print(f"❌ Error deleting documents: {str(e)}")
            import traceback
            traceback.print_exc()
            return False