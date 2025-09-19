import os
import uuid
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

# Global variable to hold RAG system (lazy initialization)
rag_system = None

def get_rag_system():
    """Initialize RAG system on first use to avoid startup import issues."""
    global rag_system
    if rag_system is None:
        print("Initializing RAG system...")
        from app.rag_system import RAGSystem
        rag_system = RAGSystem(
            pdf_dir=os.environ.get("PDF_DIR", "data/pdfs"),
            vector_db_dir=os.environ.get("VECTOR_DB_DIR", "data/vector_db"),
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"),
            deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=float(os.environ.get("DEEPSEEK_TEMPERATURE", 0.7)),
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", 1024))
        )
        print("RAG system initialized successfully!")
    return rag_system

@app.route('/')
def index():
    """Render the chatbot interface."""
    # Generate a session ID if not present
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['uploaded_pdfs'] = []
    
    return render_template('index.html', 
                          session_id=session['session_id'],
                          uploaded_pdfs=session.get('uploaded_pdfs', []))

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'message': 'Only PDF files are allowed'}), 400
        
        # Initialize RAG system if needed
        rag_sys = get_rag_system()
        
        # Use session-specific collection name
        session_id = session.get('session_id', 'default')
        collection_name = f"user_{session_id}"
        
        # Save the file temporarily
        temp_path = os.path.join('temp', f"{session_id}_{file.filename}")
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        # REPLACE the collection with the new PDF (not accumulate)
        # Enable master collection sync so CLI can access web uploads
        success = rag_sys.process_pdf_replace_collection(temp_path, collection_name, use_master_collection=True)
        
        # Replace session's uploaded PDFs list with just this file
        if success:
            session['uploaded_pdfs'] = [file.filename]  # Replace list with single file
            session.modified = True
            print(f"✓ Replaced collection '{collection_name}' with {file.filename}")
        else:
            print(f"✗ Failed to replace collection '{collection_name}' with {file.filename}")
        
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'success': success,
            'message': 'PDF processed successfully' if success else 'Failed to process PDF'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat queries."""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'success': False, 'message': 'No query provided'}), 400
        
        query = data['query']
        
        # Initialize RAG system if needed
        rag_sys = get_rag_system()
        
        # Use session-specific collection name
        session_id = session.get('session_id', 'default')
        collection_name = f"user_{session_id}"
        
        # Process the query through RAG system with specific collection
        result = rag_sys.query_collection(query, collection_name)
        
        # Format sources for display
        sources = []
        for i, doc in enumerate(result.get('source_documents', [])):
            source = {
                'content': doc.page_content[:150] + '...' if len(doc.page_content) > 150 else doc.page_content,
                'metadata': doc.metadata
            }
            sources.append(source)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'sources': sources,
            'collection': result.get('collection', collection_name)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/system_info', methods=['GET'])
def system_info():
    """Get system information."""
    try:
        # Initialize RAG system if needed
        rag_sys = get_rag_system()
        info = rag_sys.get_system_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/clear_documents', methods=['POST'])
def clear_documents():
    """Completely purge user's documents from vector DB with aggressive clearing."""
    try:
        # Initialize RAG system if needed
        rag_sys = get_rag_system()
        
        # Get session ID
        session_id = session.get('session_id', 'default')
        collection_name = f"user_{session_id}"
        
        print(f"🗑️ Starting complete document purge for session: {session_id}")
        
        # Purge the collection from ChromaDB using direct access
        import chromadb
        try:
            client = chromadb.PersistentClient(path=rag_sys.vector_db_dir)
            
            # Check if collection exists
            collections = [c.name for c in client.list_collections()]
            if collection_name in collections:
                # Delete collection completely
                client.delete_collection(collection_name)
                print(f"✅ Collection {collection_name} purged completely from ChromaDB")
                success = True
            else:
                print(f"⚠️ Collection {collection_name} not found in ChromaDB")
                success = True  # Consider it successful if it doesn't exist
        except Exception as chroma_error:
            print(f"❌ ChromaDB direct purge failed: {str(chroma_error)}")
            # Fallback to RAG system clear method
            success = rag_sys.clear_collection(collection_name)
        
        # Clear session's uploaded PDFs list regardless
        session['uploaded_pdfs'] = []
        session.modified = True
        
        print(f"{'✅' if success else '❌'} Document purge {'completed' if success else 'failed'} for session {session_id}")
        
        return jsonify({
            'success': success,
            'message': 'All documents cleared successfully' if success else 'Failed to clear documents'
        })
    except Exception as e:
        print(f"❌ Error clearing documents: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error clearing documents: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Disable reloader to avoid torch import issues on Windows
    app.run(debug=True, port=5000, use_reloader=False, host='0.0.0.0')