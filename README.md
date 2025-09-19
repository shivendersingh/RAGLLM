# RAG (Retrieval-Augmented Generation) System

A complete backend RAG system that processes PDF documents, stores embeddings in a vector database, and provides AI-powered question answering based on document content.

## Features

- **PDF Processing**: Extract and chunk text from PDF documents
- **Vector Storage**: Store document embeddings using ChromaDB
- **Semantic Search**: Find relevant document sections using similarity search
- **LLM Integration**: Generate contextual responses using Hugging Face transformers
- **CLI Interface**: Easy-to-use command line interface

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
RagLLM/
├── app/
│   ├── __init__.py
│   ├── pdf_processor.py     # PDF loading and chunking
│   ├── vector_store.py      # Vector database management
│   ├── llm_service.py       # Language model service
│   └── rag_system.py        # Main RAG system
├── data/
│   ├── pdfs/               # Processed PDF files
│   └── vector_db/          # Vector database storage
├── requirements.txt        # Python dependencies
├── main.py                 # CLI entry point
├── test.py                 # Test suite
└── README.md              # This file
```

## Usage

### Command Line Interface

1. **Process a PDF document**:
   ```bash
   python main.py --pdf "path/to/your/document.pdf"
   ```

2. **Query the system**:
   ```bash
   python main.py --query "What is the main topic of the document?"
   ```

3. **Process PDF and query in one command**:
   ```bash
   python main.py --pdf "document.pdf" --query "What are the key findings?"
   ```

4. **Use custom directories**:
   ```bash
   python main.py --pdf-dir "custom/pdf/dir" --vector-db "custom/vector/dir" --pdf "document.pdf"
   ```

### Python API

```python
from app.rag_system import RAGSystem

# Initialize the system
rag = RAGSystem()

# Process a PDF
success = rag.process_pdf("path/to/document.pdf")

# Query the system
result = rag.query("What is artificial intelligence?")
print(result["response"])
```

## Testing

Run the test suite to validate the implementation:

```bash
python test.py
```

The test suite includes:
- Component initialization tests
- PDF processing validation
- Query functionality testing
- End-to-end workflow verification
- Error handling validation

## How It Works

1. **PDF Processing**: 
   - PDFs are loaded using PyPDFLoader
   - Text is split into chunks using RecursiveCharacterTextSplitter
   - Chunks are processed and prepared for embedding

2. **Vector Storage**:
   - Text chunks are converted to embeddings using sentence-transformers
   - Embeddings are stored in ChromaDB vector database
   - Database persists to disk for reuse

3. **Query Processing**:
   - User queries are converted to embeddings
   - Similarity search finds most relevant document chunks
   - Retrieved chunks serve as context for the LLM

4. **Response Generation**:
   - LLM (Flan-T5) generates responses based on context
   - Responses are grounded in the retrieved document content
   - Source documents are provided for transparency

## Configuration

### Models Used

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: `google/flan-t5-small`

### Customization

You can customize the models by modifying the RAGSystem initialization:

```python
rag = RAGSystem(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    llm_model="google/flan-t5-base"
)
```

### Chunking Parameters

Modify chunking behavior in PDFProcessor:

```python
processor = PDFProcessor(
    upload_dir="pdfs",
    chunk_size=800,    # Larger chunks
    chunk_overlap=150  # More overlap
)
```

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Error Handling

The system includes comprehensive error handling for:
- Missing or invalid PDF files
- Vector database connection issues
- LLM generation failures
- Memory constraints

## Performance Considerations

- **CPU vs GPU**: Models run on CPU by default (set `device="cuda"` for GPU)
- **Memory**: Larger models require more RAM
- **Storage**: Vector database grows with document corpus
- **Speed**: First query after startup takes longer due to model loading

## Limitations

- Currently supports PDF files only
- LLM responses limited to context window size
- No real-time document updates (requires reprocessing)
- Single-user system (no concurrent access handling)

## Future Enhancements

- Support for additional file formats (Word, text, etc.)
- Web interface
- Multi-user support
- Larger language models
- Real-time document indexing
- Advanced query filtering