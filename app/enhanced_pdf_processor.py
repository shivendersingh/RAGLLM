import os
import re
from typing import List, Dict, Any, Optional, Union, Callable
import traceback
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document

# Try to import UnstructuredPDFLoader, fall back if not available
try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("Warning: UnstructuredPDFLoader not available. Some advanced features will be disabled.")

# Try to import MarkdownHeaderTextSplitter, fall back if not available
try:
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    MARKDOWN_SPLITTER_AVAILABLE = True
except ImportError:
    MARKDOWN_SPLITTER_AVAILABLE = False
    print("Warning: MarkdownHeaderTextSplitter not available. Some advanced features will be disabled.")

class EnhancedPDFProcessor:
    """Enhanced PDF processor with multiple chunking strategies"""
    
    # Document types with corresponding processors
    DOCUMENT_TYPES = {
        "default": "process_default",
        "academic": "process_academic",
        "legal": "process_legal",
        "technical": "process_technical",
        "presentation": "process_presentation",
        "form": "process_form"
    }
    
    def __init__(
        self, 
        upload_dir: str,
        chunk_size: int = 600, 
        chunk_overlap: int = 100,
        enable_layout_analysis: bool = True,
        extraction_mode: str = "auto"
    ):
        """Initialize the enhanced PDF processor.
        
        Args:
            upload_dir: Directory to store uploaded PDFs
            chunk_size: Default chunk size for text splitting
            chunk_overlap: Default chunk overlap for text splitting
            enable_layout_analysis: Whether to analyze document layout
            extraction_mode: PDF extraction mode ('auto', 'text', 'layout')
        """
        self.upload_dir = upload_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_layout_analysis = enable_layout_analysis
        self.extraction_mode = extraction_mode
        
        # Create directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)
        
        # Initialize text splitters
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        
        # Try to initialize token splitter
        try:
            self.token_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size // 4,  # Adjust for tokens vs chars
                chunk_overlap=self.chunk_overlap // 4
            )
        except Exception:
            print("Warning: TokenTextSplitter not available, using character splitter as fallback")
            self.token_splitter = self.char_splitter
    
    def process_pdf(self, file_path: str, doc_type: str = "auto", **kwargs) -> List[Document]:
        """Process a PDF file with the appropriate strategy based on document type.
        
        This method maintains compatibility with the original interface while adding
        enhanced functionality.
        
        Args:
            file_path: Path to the PDF file
            doc_type: Document type ('auto', 'default', 'academic', 'legal', etc.)
            **kwargs: Additional processing options
            
        Returns:
            List of Document objects with text chunks and metadata
        """
        try:
            # Auto-detect document type if set to auto
            if doc_type == "auto":
                doc_type = self._detect_document_type(file_path)
            
            print(f"Processing PDF as '{doc_type}' document type...")
            
            # Use the appropriate processing method based on document type
            if doc_type in self.DOCUMENT_TYPES:
                processor_method = getattr(self, self.DOCUMENT_TYPES[doc_type])
                return processor_method(file_path, **kwargs)
            else:
                # Fall back to default processor
                return self.process_default(file_path, **kwargs)
                
        except Exception as e:
            print(f"Error processing PDF with enhanced processor: {str(e)}")
            print("Falling back to legacy processing method...")
            # Fall back to original implementation for compatibility
            return self._legacy_process_pdf(file_path)
    
    def _legacy_process_pdf(self, file_path: str) -> List[Document]:
        """Legacy PDF processing method for backward compatibility"""
        print("Using legacy PDF processing method...")
        # This is the original implementation
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        split_pages = text_splitter.split_documents(pages)
        return split_pages
    
    def _detect_document_type(self, file_path: str) -> str:
        """Detect document type based on content and structure analysis"""
        try:
            # Load first few pages for analysis
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            if not pages:
                return "default"
            
            # Analyze first 3 pages or all pages if fewer than 3
            analysis_pages = min(3, len(pages))
            text = " ".join([p.page_content for p in pages[:analysis_pages]])
            text_lower = text.lower()
            
            # Check for academic paper indicators
            academic_indicators = ["abstract", "keywords", "introduction", "methodology", 
                                 "literature review", "references", "doi", "journal", 
                                 "citation", "bibliography", "publication"]
            if sum(1 for indicator in academic_indicators if indicator in text_lower) >= 2:
                return "academic"
            
            # Check for legal document indicators
            legal_indicators = ["agreement", "contract", "terms", "party", "parties", 
                              "clause", "section", "law", "rights", "herein", 
                              "provisions", "pursuant", "legal", "whereas", "attorney"]
            if sum(1 for indicator in legal_indicators if indicator in text_lower) >= 3:
                return "legal"
            
            # Check for technical document indicators
            technical_indicators = ["figure", "table", "algorithm", "implementation", 
                                  "documentation", "specification", "api", "interface",
                                  "installation", "configuration", "code", "function"]
            if sum(1 for indicator in technical_indicators if indicator in text_lower) >= 2:
                return "technical"
            
            # Check for presentation indicators
            presentation_indicators = ["slide", "presentation", "agenda"]
            if (len(text) < 1500 and  # Short pages typical of presentations
                any(indicator in text_lower for indicator in presentation_indicators)):
                return "presentation"
            
            # Check for form indicators
            form_indicators = ["form", "fill", "complete", "signature", "sign", "field", "checkbox"]
            if sum(1 for indicator in form_indicators if indicator in text_lower) >= 2:
                return "form"
            
            # Default to standard processing
            return "default"
            
        except Exception as e:
            print(f"Error detecting document type: {str(e)}")
            return "default"
    
    def process_default(self, file_path: str, **kwargs) -> List[Document]:
        """Process standard documents with balanced chunking"""
        try:
            print("Processing with default strategy...")
            # Load PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            if not pages:
                print("No pages loaded from PDF")
                return []
            
            # Try to identify structure
            has_clear_sections = self._has_section_headers(pages)
            
            if has_clear_sections:
                print("Document has clear sections, extracting sections first...")
                # Extract by sections first
                sections = self.extract_section_blocks(pages)
                # Then split sections into manageable chunks
                return self.sentence_splitter.split_documents(sections)
            else:
                print("No clear sections detected, using sentence-aware splitting...")
                # Use sentence-aware splitting for better semantic boundaries
                return self.sentence_splitter.split_documents(pages)
                
        except Exception as e:
            print(f"Error in default processing: {str(e)}")
            return self._legacy_process_pdf(file_path)
    
    def process_academic(self, file_path: str, **kwargs) -> List[Document]:
        """Process academic papers with section awareness"""
        try:
            print("Processing with academic paper strategy...")
            
            # Try structured loader if available, otherwise use PyPDF
            if UNSTRUCTURED_AVAILABLE:
                try:
                    loader = UnstructuredPDFLoader(file_path, strategy="fast")
                    elements = loader.load()
                    print(f"Loaded {len(elements)} elements with UnstructuredPDFLoader")
                except Exception as e:
                    print(f"UnstructuredPDFLoader failed: {str(e)}, falling back to PyPDFLoader")
                    loader = PyPDFLoader(file_path)
                    elements = loader.load_and_split()
            else:
                loader = PyPDFLoader(file_path)
                elements = loader.load_and_split()
            
            if not elements:
                return self.process_default(file_path)
            
            # If markdown splitter is available, use section-based splitting
            if MARKDOWN_SPLITTER_AVAILABLE:
                # Define academic paper sections
                headers_to_split_on = [
                    ("# ", "Header 1"),
                    ("## ", "Header 2"),
                    ("### ", "Header 3"),
                    ("#### ", "Header 4"),
                ]
                
                # Convert to markdown-like format for section splitting
                markdown_text = ""
                for doc in elements:
                    text = doc.page_content
                    # Identify potential headers
                    lines = text.split("\n")
                    for line in lines:
                        line = line.strip()
                        if re.match(r'^([A-Z][A-Za-z0-9\-\(\) ]{3,})$', line):
                            markdown_text += f"# {line}\n\n"
                        elif re.match(r'^(\d+\.\d*[\.\s]*[A-Z][A-Za-z0-9\-\(\) ]{2,})$', line):
                            markdown_text += f"## {line}\n\n"
                        else:
                            markdown_text += f"{line}\n"
                
                # Split on headers
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                md_header_splits = markdown_splitter.split_text(markdown_text)
                
                # Further split long sections
                return self.sentence_splitter.split_documents(md_header_splits)
            else:
                # Fall back to section extraction
                print("MarkdownHeaderTextSplitter not available, using section extraction...")
                sections = self.extract_section_blocks(elements)
                return self.sentence_splitter.split_documents(sections)
            
        except Exception as e:
            print(f"Error in academic processing: {str(e)}")
            return self.process_default(file_path)
    
    def process_legal(self, file_path: str, **kwargs) -> List[Document]:
        """Process legal documents with clause and section awareness"""
        try:
            print("Processing with legal document strategy...")
            # Load PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            if not pages:
                return []
            
            documents = []
            section_pattern = r'(?:Section|SECTION|Article|ARTICLE)\s+(\d+(?:\.\d+)?)'
            
            # Process page by page
            buffer = ""
            current_section = "General"
            
            for page in pages:
                text = page.page_content
                lines = text.split("\n")
                
                for line in lines:
                    line = line.strip()
                    # Check for section headers
                    section_match = re.search(section_pattern, line)
                    if section_match or re.match(r'^(\d+\.\d+)\s+[A-Z]', line):
                        # Save previous section
                        if buffer and buffer.strip():
                            documents.append(Document(
                                page_content=buffer.strip(),
                                metadata={"section": current_section, "type": "legal", **page.metadata}
                            ))
                        
                        current_section = line
                        buffer = current_section + "\n"
                    else:
                        if line:  # Skip empty lines
                            buffer += line + "\n"
            
            # Add final section
            if buffer and buffer.strip():
                documents.append(Document(
                    page_content=buffer.strip(),
                    metadata={"section": current_section, "type": "legal", **pages[-1].metadata}
                ))
            
            # If no sections were found, fall back to default processing
            if not documents:
                print("No legal sections detected, falling back to default processing...")
                return self.process_default(file_path)
            
            # Use a more conservative chunking for legal text to preserve context
            return self.sentence_splitter.split_documents(documents)
            
        except Exception as e:
            print(f"Error in legal processing: {str(e)}")
            return self.process_default(file_path)
    
    def process_technical(self, file_path: str, **kwargs) -> List[Document]:
        """Process technical documents preserving code blocks and tables"""
        try:
            print("Processing with technical document strategy...")
            
            # Try structured loader if available
            if UNSTRUCTURED_AVAILABLE:
                try:
                    loader = UnstructuredPDFLoader(file_path, strategy="fast")
                    elements = loader.load()
                except Exception as e:
                    print(f"UnstructuredPDFLoader failed: {str(e)}, falling back to PyPDFLoader")
                    loader = PyPDFLoader(file_path)
                    elements = loader.load_and_split()
            else:
                loader = PyPDFLoader(file_path)
                elements = loader.load_and_split()
            
            if not elements:
                return []
            
            processed_docs = []
            
            # Identify and preserve code blocks and tables
            for doc in elements:
                text = doc.page_content
                
                # Check for code blocks (indented or monospaced text sections)
                code_block_patterns = [
                    r'```[\s\S]*?```',  # Markdown code blocks
                    r'`[^`\n]+`',       # Inline code
                    r'^[ ]{4,}[^\s].*$',  # Indented code (4+ spaces)
                ]
                
                # Process text and identify code blocks
                has_code = False
                for pattern in code_block_patterns:
                    if re.search(pattern, text, re.MULTILINE):
                        has_code = True
                        break
                
                if has_code:
                    # Split on code blocks
                    code_pattern = r'(```[\s\S]*?```|`[^`\n]+`|^[ ]{4,}[^\s].*$)'
                    parts = re.split(code_pattern, text, flags=re.MULTILINE)
                    
                    for part in parts:
                        if part and part.strip():
                            is_code = any(re.match(pattern, part, re.MULTILINE) for pattern in code_block_patterns)
                            
                            processed_docs.append(Document(
                                page_content=part.strip(),
                                metadata={"type": "code" if is_code else "text", **doc.metadata}
                            ))
                else:
                    # Regular text
                    processed_docs.append(Document(
                        page_content=text,
                        metadata={"type": "text", **doc.metadata}
                    ))
            
            # If no processed docs, fall back to default
            if not processed_docs:
                return self.process_default(file_path)
            
            # Split regular text while preserving code blocks
            result_docs = []
            for doc in processed_docs:
                if doc.metadata.get("type") == "code":
                    # Don't split code blocks
                    result_docs.append(doc)
                else:
                    # Split text normally
                    chunks = self.sentence_splitter.split_documents([doc])
                    result_docs.extend(chunks)
                    
            return result_docs
            
        except Exception as e:
            print(f"Error in technical processing: {str(e)}")
            return self.process_default(file_path)
    
    def process_presentation(self, file_path: str, **kwargs) -> List[Document]:
        """Process presentation slides as individual chunks"""
        try:
            print("Processing with presentation strategy...")
            # Load PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            if not pages:
                return []
            
            # For presentations, treat each page as a separate slide/chunk
            result = []
            for i, page in enumerate(pages):
                # Add slide number to metadata
                metadata = {**page.metadata, "slide": i+1, "type": "presentation"}
                result.append(Document(page_content=page.page_content, metadata=metadata))
            
            print(f"Processed {len(result)} slides")
            return result
            
        except Exception as e:
            print(f"Error in presentation processing: {str(e)}")
            return self._legacy_process_pdf(file_path)
    
    def process_form(self, file_path: str, **kwargs) -> List[Document]:
        """Process forms with field detection"""
        try:
            print("Processing with form strategy...")
            
            # Try structured loader if available
            if UNSTRUCTURED_AVAILABLE:
                try:
                    loader = UnstructuredPDFLoader(file_path, strategy="fast")
                    elements = loader.load()
                except Exception as e:
                    print(f"UnstructuredPDFLoader failed: {str(e)}, falling back to PyPDFLoader")
                    loader = PyPDFLoader(file_path)
                    elements = loader.load_and_split()
            else:
                loader = PyPDFLoader(file_path)
                elements = loader.load_and_split()
            
            if not elements:
                return []
            
            # Identify form fields
            form_fields = []
            field_patterns = [
                r'(?:^|\n)([A-Za-z][A-Za-z\s]+):[\s]*(.*)(?:\n|$)',  # Label: Value
                r'([A-Za-z][A-Za-z\s]+)[\s]*\[([^\]]*)\]',          # Label [Value]
                r'([A-Za-z][A-Za-z\s]+)[\s]*_+[\s]*([^\n]*)',       # Label _____ Value
            ]
            
            for doc in elements:
                text = doc.page_content
                
                for pattern in field_patterns:
                    fields = re.finditer(pattern, text)
                    
                    for match in fields:
                        field_name = match.group(1).strip()
                        field_value = match.group(2).strip() if len(match.groups()) > 1 else ""
                        
                        if field_name and len(field_name) > 2:  # Valid field name
                            form_fields.append(Document(
                                page_content=f"{field_name}: {field_value}",
                                metadata={"field_name": field_name, "field_value": field_value, 
                                        "type": "form_field", **doc.metadata}
                            ))
            
            # If form fields were identified, return them
            if form_fields:
                print(f"Identified {len(form_fields)} form fields")
                return form_fields
            
            # Otherwise fall back to default processing
            print("No form fields detected, falling back to default processing...")
            return self.process_default(file_path)
            
        except Exception as e:
            print(f"Error in form processing: {str(e)}")
            return self.process_default(file_path)
    
    def extract_section_blocks(self, pages: List[Document]) -> List[Document]:
        """Extract sections from document pages based on title patterns.
        
        This method maintains compatibility with the original implementation
        while enhancing section detection.
        """
        sections = []
        buffer = ""
        current_title = "General"
        
        # Enhanced section header patterns
        section_patterns = [
            r'^([A-Z][A-Z\s]{4,})$',  # ALL CAPS HEADERS
            r'^([A-Z][a-zA-Z0-9\-\(\) ]{4,})$',  # Capitalized headers
            r'^(\d+\.\s+[A-Z][a-zA-Z0-9\-\(\) ]{2,})$',  # 1. Section Title
            r'^(Chapter\s+\d+[:\.\s]+[A-Z][a-zA-Z0-9\-\(\) ]{2,})$',  # Chapter 1: Title
            r'^(Section\s+\d+[:\.\s]+[A-Z][a-zA-Z0-9\-\(\) ]{2,})$',  # Section 1: Title
            r'^(CHAPTER\s+\d+[:\.\s]+[A-Z][a-zA-Z0-9\-\(\) ]{2,})$',  # CHAPTER 1: TITLE
            r'^(SECTION\s+\d+[:\.\s]+[A-Z][a-zA-Z0-9\-\(\) ]{2,})$',  # SECTION 1: TITLE
        ]
        
        for page in pages:
            lines = page.page_content.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                is_section_header = any(re.match(pattern, line) for pattern in section_patterns)
                
                if is_section_header:
                    # Save previous section
                    if buffer and buffer.strip():
                        sections.append(Document(
                            page_content=buffer.strip(),
                            metadata={"title": current_title, **page.metadata}
                        ))
                    
                    current_title = line
                    buffer = current_title + "\n"
                else:
                    buffer += line + "\n"
        
        # Add the last section
        if buffer and buffer.strip():
            last_page_metadata = pages[-1].metadata if pages else {}
            sections.append(Document(
                page_content=buffer.strip(),
                metadata={"title": current_title, **last_page_metadata}
            ))
        
        print(f"Extracted {len(sections)} sections")
        return sections
    
    def _has_section_headers(self, pages: List[Document]) -> bool:
        """Check if document has clear section headers"""
        section_patterns = [
            r'^([A-Z][A-Z\s]{4,})$',
            r'^([A-Z][a-zA-Z0-9\-\(\) ]{4,})$',
            r'^(\d+\.\s+[A-Z][a-zA-Z0-9\-\(\) ]{2,})$',
            r'^(Chapter\s+\d+[:\.\s]+[A-Z][a-zA-Z0-9\-\(\) ]{2,})$',
            r'^(Section\s+\d+[:\.\s]+[A-Z][a-zA-Z0-9\-\(\) ]{2,})$',
        ]
        
        section_count = 0
        
        for page in pages:
            lines = page.page_content.strip().split("\n")
            for line in lines:
                line = line.strip()
                if any(re.match(pattern, line) for pattern in section_patterns):
                    section_count += 1
                    if section_count >= 2:  # At least 2 sections to consider structured
                        return True
        
        return False