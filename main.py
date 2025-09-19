import os
import sys
import argparse
from app.rag_system import RAGSystem

def main():
    parser = argparse.ArgumentParser(description="Enhanced RAG System CLI")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to process")
    parser.add_argument("--pdf-dir", type=str, help="Directory containing PDF files to process")
    parser.add_argument("--query", type=str, help="Query to ask the system")
    parser.add_argument("--doc-type", type=str, choices=["auto", "default", "academic", "legal", "technical", "presentation", "form"], 
                       default="auto", help="Document type for processing")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("--storage-pdf-dir", type=str, default="data/pdfs", help="Directory for PDF storage")
    parser.add_argument("--vector-db", type=str, default="data/vector_db", help="Directory for vector database")
    
    args = parser.parse_args()
    
    # Initialize RAG system with custom directories if provided
    rag_system = RAGSystem(
        pdf_dir=args.storage_pdf_dir,
        vector_db_dir=args.vector_db
    )
    
    # Show system info if requested
    if args.info:
        info = rag_system.get_system_info()
        print("\n=== RAG System Information ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        print("=" * 30)
    
    # Process single PDF if provided
    if args.pdf:
        print(f"Processing PDF: {args.pdf}")
        if args.doc_type != "auto":
            success = rag_system.process_pdf_with_type(args.pdf, doc_type=args.doc_type)
        else:
            success = rag_system.process_pdf(args.pdf)
        
        if success:
            print(f"✓ Successfully processed {args.pdf}")
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
        result = rag_system.query(args.query)
        
        print("\n" + "="*50)
        print("QUERY RESULT")
        print("="*50)
        print(f"Query: {result['query']}")
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
                        if key in ['title', 'section', 'type', 'slide', 'field_name']:
                            metadata_items.append(f"{key}: {value}")
                    if metadata_items:
                        print(f"Metadata: {', '.join(metadata_items)}")
        print("="*50)
    
    # If no arguments provided, show help
    if not any([args.pdf, args.pdf_dir, args.query, args.info]):
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py --pdf document.pdf --doc-type academic")
        print("  python main.py --pdf-dir /path/to/pdfs")
        print("  python main.py --query 'What is this document about?'")
        print("  python main.py --info")
        print("  python main.py --pdf doc.pdf --query 'summarize this'")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())