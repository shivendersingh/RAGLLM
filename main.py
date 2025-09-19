import os
import sys
import argparse
from dotenv import load_dotenv
from app.rag_system import RAGSystem
import chromadb  # Add ChromaDB for collection management

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Enhanced RAG System CLI with DeepSeek Integration")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to process")
    parser.add_argument("--pdf-dir", type=str, help="Directory containing PDF files to process")
    parser.add_argument("--query", type=str, help="Query to ask the system")
    parser.add_argument("--doc-type", type=str, choices=["auto", "default", "academic", "legal", "technical", "presentation", "form"], 
                       default="auto", help="Document type for processing")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("--storage-pdf-dir", type=str, default="data/pdfs", help="Directory for PDF storage")
    parser.add_argument("--vector-db", type=str, default="data/vector_db", help="Directory for vector database")
    
    # Collection management parameters
    parser.add_argument("--collection", type=str, help="Specific collection to query (defaults to 'master' which contains web uploads)")
    parser.add_argument("--list-collections", action="store_true", help="List all available collections")
    parser.add_argument("--use-latest-web", action="store_true", help="Use the latest web session collection")
    
    # DeepSeek LLM parameters with environment variable defaults
    parser.add_argument("--deepseek-api-key", type=str, 
                       default=os.environ.get("DEEPSEEK_API_KEY"), 
                       help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--deepseek-model", type=str, 
                       default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"), 
                       help="DeepSeek model name (default: deepseek-chat)")
    parser.add_argument("--temperature", type=float, 
                       default=float(os.environ.get("DEEPSEEK_TEMPERATURE", 0.7)), 
                       help="Temperature for DeepSeek model (0.0-1.0, default: 0.7)")
    parser.add_argument("--max-tokens", type=int, 
                       default=int(os.environ.get("DEEPSEEK_MAX_TOKENS", 1024)), 
                       help="Maximum tokens for DeepSeek response (default: 1024)")
    
    args = parser.parse_args()
    
    # Get DeepSeek API key from argument or environment variable
    deepseek_api_key = args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
    
    # Initialize RAG system with custom directories and DeepSeek configuration
    rag_system = RAGSystem(
        pdf_dir=args.storage_pdf_dir,
        vector_db_dir=args.vector_db,
        deepseek_api_key=deepseek_api_key,
        deepseek_model=args.deepseek_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # List collections if requested
    if args.list_collections:
        try:
            client = chromadb.PersistentClient(path=args.vector_db)
            collections = client.list_collections()
            
            print("\n" + "="*60)
            print("AVAILABLE COLLECTIONS")
            print("="*60)
            if collections:
                for i, collection in enumerate(collections):
                    # Try to get document count
                    try:
                        coll = client.get_collection(collection.name)
                        count = coll.count()
                    except:
                        count = "unknown"
                    
                    print(f"{i+1:2d}. {collection.name} ({count} documents)")
                    
                    # If this is a web session, indicate it
                    if collection.name.startswith("user_"):
                        print(f"     (Web session collection)")
            else:
                print("No collections found")
            print("="*60)
            return 0
        except Exception as e:
            print(f"Error listing collections: {str(e)}")
            return 1
            
    # Determine which collection to use
    # Use master collection by default to access web uploads
    collection_name = "master"
    if args.collection:
        collection_name = args.collection
        print(f"Using specified collection: {collection_name}")
    elif args.use_latest_web:
        try:
            # Find the latest web session collection
            client = chromadb.PersistentClient(path=args.vector_db)
            collections = client.list_collections()
            web_collections = [c.name for c in collections if c.name.startswith("user_")]
            
            if web_collections:
                # Sort by name to get the most recent (this is a heuristic)
                collection_name = sorted(web_collections)[-1]
                print(f"Using latest web collection: {collection_name}")
            else:
                print("No web collections found, using master collection")
                collection_name = "master"
        except Exception as e:
            print(f"Error finding latest web collection: {str(e)}")
            collection_name = "master"
    
    # Show system info if requested
    if args.info:
        info = rag_system.get_system_info()
        print("\n" + "="*60)
        print("RAG SYSTEM INFORMATION")
        print("="*60)
        print(f"PDF Directory: {info['pdf_directory']}")
        print(f"PDF Files Count: {info['pdf_files_count']}")
        print(f"Vector DB Directory: {info['vector_db_directory']}")
        print(f"Document Types Supported: {', '.join(info['document_types_supported'])}")
        
        # Show collections with details
        print(f"\nCollections:")
        if 'collections' in info and info['collections']:
            for name, count in info['collections'].items():
                collection_type = "(Web session)" if name.startswith("user_") else "(Default/CLI)"
                print(f"  - {name}: {count} documents {collection_type}")
        else:
            print("  No collections information available")
        
        # Show LLM info
        if 'llm_service' in info and info['llm_service']:
            print(f"\nLLM Service:")
            llm = info['llm_service']
            print(f"  Type: {llm.get('service_type', 'Unknown')}")
            print(f"  Model: {llm.get('model_name', 'Unknown')}")
            if 'temperature' in llm:
                print(f"  Temperature: {llm.get('temperature')}")
            if 'max_tokens' in llm:
                print(f"  Max Tokens: {llm.get('max_tokens')}")
        
        print(f"\nCurrent Collection: {collection_name}")
        print("="*60)
    
    # Process single PDF if provided
    if args.pdf:
        print(f"Processing PDF: {args.pdf} into collection: {collection_name}")
        # Use collection-aware processing
        success = rag_system.process_pdf_replace_collection(args.pdf, collection_name)
        
        if success:
            print(f"✓ Successfully processed {args.pdf} into collection '{collection_name}'")
        else:
            print(f"✗ Failed to process {args.pdf}")
            return 1
    
    # Process multiple PDFs if directory provided
    if args.pdf_dir:
        results = rag_system.process_multiple_pdfs(args.pdf_dir)
        print("\n=== Processing Results ===")
        for file, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {file}")
        
        successful = sum(1 for success in results.values() if success)
        print(f"\nSummary: {successful}/{len(results)} files processed successfully")
    
    # Handle query if provided
    if args.query:
        print(f"Querying system with: {args.query}")
        print(f"Using collection: {collection_name}")
        # Use collection-aware querying
        result = rag_system.query_collection(args.query, collection_name)
        
        print("\n" + "="*50)
        print("QUERY RESULT")
        print("="*50)
        print(f"Query: {result['query']}")
        print(f"Collection: {collection_name}")
        print(f"\nResponse: {result['response']}")
        
        if result['source_documents']:
            print(f"\nSource Documents ({len(result['source_documents'])}):")
            for i, doc in enumerate(result['source_documents'][:3]):  # Show first 3
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc.page_content[:150]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    # Show relevant metadata
                    metadata_items = []
                    for key, value in doc.metadata.items():
                        if key in ['title', 'section', 'type', 'slide', 'field_name', 'source_file']:
                            metadata_items.append(f"{key}: {value}")
                    if metadata_items:
                        print(f"Metadata: {', '.join(metadata_items)}")
        else:
            print(f"\n⚠️  No source documents found in collection '{collection_name}'")
            print("💡 Try using --list-collections to see available collections")
            print("💡 Or use --use-latest-web to query the latest web session")
        print("="*50)
    
    # If no arguments provided, show help
    if not any([args.pdf, args.pdf_dir, args.query, args.info, args.list_collections]):
        parser.print_help()
        print("\n" + "="*60)
        print("EXAMPLE USAGE")
        print("="*60)
        print("Basic Operations:")
        print("  python main.py --pdf document.pdf --doc-type academic")
        print("  python main.py --pdf-dir /path/to/pdfs")
        print("  python main.py --query 'What is this document about?'")
        print("  python main.py --info")
        
        print("\nCollection Management:")
        print("  python main.py --list-collections")
        print("  python main.py --use-latest-web --query 'What is this about?'")
        print("  python main.py --collection user_abc123 --query 'Summarize this'")
        
        print("\nCombined Operations:")
        print("  python main.py --pdf doc.pdf --collection my_docs --query 'summarize'")
        print("  python main.py --deepseek-api-key sk-xxx --use-latest-web --query 'analyze'")
        print("="*60)
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())