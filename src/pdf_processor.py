"""
PDF processing module for text extraction, chunking, and preprocessing.
"""
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib

from pypdf import PdfReader
from loguru import logger
import nltk
from tqdm import tqdm

from config import config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

nltk.download('punkt_tab')

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a PDF document."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    page_number: int
    pdf_name: str
    start_char: int
    end_char: int


@dataclass
class PDFDocument:
    """Represents a processed PDF document."""
    filename: str
    content: str
    metadata: Dict[str, Any]
    chunks: List[DocumentChunk]
    file_path: Path
    file_hash: str


class SimpleTextSplitter:
    """Simple text splitter to replace LangChain's RecursiveCharacterTextSplitter."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to find a good split point using separators
            best_split = end
            for separator in self.separators:
                # Look for separator within the overlap region
                search_start = max(start, end - self.chunk_overlap)
                sep_pos = text.rfind(separator, search_start, end)
                if sep_pos != -1:
                    best_split = sep_pos + len(separator)
                    break
            
            chunks.append(text[start:best_split])
            
            # Calculate next start position with overlap
            start = max(start + 1, best_split - self.chunk_overlap)
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]


class PDFProcessor:
    """Handles PDF text extraction, chunking, and preprocessing."""
    
    def __init__(self):
        self.text_splitter = SimpleTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_PDFS)
        
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                
                # Extract metadata
                metadata = {
                    'filename': pdf_path.name,
                    'file_path': str(pdf_path),
                    'num_pages': len(reader.pages),
                    'file_size': pdf_path.stat().st_size,
                    'creation_time': pdf_path.stat().st_ctime,
                    'modification_time': pdf_path.stat().st_mtime
                }
                
                # Add PDF metadata if available
                if reader.metadata:
                    pdf_meta = reader.metadata
                    metadata.update({
                        'title': pdf_meta.get('/Title', ''),
                        'author': pdf_meta.get('/Author', ''),
                        'subject': pdf_meta.get('/Subject', ''),
                        'creator': pdf_meta.get('/Creator', ''),
                        'producer': pdf_meta.get('/Producer', ''),
                        'creation_date': pdf_meta.get('/CreationDate', ''),
                        'modification_date': pdf_meta.get('/ModDate', '')
                    })
                
                # Extract text from all pages
                text_content = ""
                page_texts = []
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            page_texts.append((page_num + 1, page_text))
                            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1} of {pdf_path}: {e}")
                        continue
                
                metadata['page_texts'] = page_texts
                
                if not text_content.strip():
                    logger.warning(f"No text content extracted from {pdf_path}")
                    return "", metadata
                
                # Clean the text
                text_content = self.clean_text(text_content)
                
                logger.info(f"Successfully extracted {len(text_content)} characters from {pdf_path}")
                return text_content, metadata
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[\.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s+', r'\1 ', text)
        
        # Remove extra newlines and spaces
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        
        return text.strip()
    
    def create_chunks(self, document: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split document into chunks with metadata.
        
        Args:
            document: Full document text
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(document)
        
        current_pos = 0
        page_texts = metadata.get('page_texts', [])
        
        for i, chunk_text in enumerate(text_chunks):
            # Find which page this chunk belongs to
            chunk_page = self._find_chunk_page(chunk_text, page_texts)
            
            # Create unique chunk ID
            chunk_id = hashlib.md5(
                f"{metadata['filename']}_{i}_{chunk_text[:50]}".encode()
            ).hexdigest()
            
            # Find character positions
            start_char = document.find(chunk_text, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(chunk_text)
            current_pos = end_char
            
            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'chunk_length': len(chunk_text),
                'page_number': chunk_page
            }
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                page_number=chunk_page,
                pdf_name=metadata['filename'],
                start_char=start_char,
                end_char=end_char
            )
            
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for {metadata['filename']}")
        return chunks
    
    def _find_chunk_page(self, chunk_text: str, page_texts: List[Tuple[int, str]]) -> int:
        """Find which page a chunk belongs to."""
        chunk_start = chunk_text[:100]  # First 100 chars for matching
        
        for page_num, page_text in page_texts:
            if chunk_start in page_text:
                return page_num
        
        return 1  # Default to page 1 if not found
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file for change detection."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def process_pdf(self, pdf_path: Path) -> Optional[PDFDocument]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDFDocument object or None if processing failed
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Calculate file hash
            file_hash = self.calculate_file_hash(pdf_path)
            
            # Extract text and metadata
            content, metadata = self.extract_text_from_pdf(pdf_path)
            
            if not content:
                logger.warning(f"No content extracted from {pdf_path}")
                return None
            
            # Create chunks
            chunks = self.create_chunks(content, metadata)
            
            if not chunks:
                logger.warning(f"No chunks created for {pdf_path}")
                return None
            
            # Create PDF document
            pdf_doc = PDFDocument(
                filename=pdf_path.name,
                content=content,
                metadata=metadata,
                chunks=chunks,
                file_path=pdf_path,
                file_hash=file_hash
            )
            
            logger.success(f"Successfully processed {pdf_path}: {len(chunks)} chunks")
            return pdf_doc
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return None
    
    async def process_pdfs_batch(self, pdf_paths: List[Path]) -> List[PDFDocument]:
        """
        Process multiple PDF files concurrently.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of processed PDFDocument objects
        """
        processed_docs = []
        
        # Process PDFs in batches to avoid overwhelming the system
        for i in range(0, len(pdf_paths), config.BATCH_SIZE):
            batch = pdf_paths[i:i + config.BATCH_SIZE]
            logger.info(f"Processing batch {i//config.BATCH_SIZE + 1}: {len(batch)} PDFs")
            
            # Use thread pool for CPU-bound PDF processing
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(self.executor, self.process_pdf, pdf_path)
                for pdf_path in batch
            ]
            
            # Wait for batch completion with progress bar
            batch_results = []
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), 
                           desc=f"Processing batch {i//config.BATCH_SIZE + 1}"):
                result = await task
                if result:
                    batch_results.append(result)
            
            processed_docs.extend(batch_results)
            
            # Small delay between batches to prevent system overload
            if i + config.BATCH_SIZE < len(pdf_paths):
                await asyncio.sleep(1)
        
        logger.info(f"Successfully processed {len(processed_docs)} out of {len(pdf_paths)} PDFs")
        return processed_docs
    
    def discover_pdfs(self, directory: Path) -> List[Path]:
        """
        Discover all PDF files in a directory.
        
        Args:
            directory: Directory to search for PDFs
            
        Returns:
            List of PDF file paths
        """
        pdf_files = []
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return pdf_files
        
        # Search for PDF files recursively
        for pdf_path in directory.rglob("*.pdf"):
            if pdf_path.is_file():
                pdf_files.append(pdf_path)
        
        logger.info(f"Discovered {len(pdf_files)} PDF files in {directory}")
        return sorted(pdf_files)
    
    async def process_directory(self, directory: Path) -> List[PDFDocument]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory: Directory containing PDF files
            
        Returns:
            List of processed PDFDocument objects
        """
        pdf_paths = self.discover_pdfs(directory)
        
        if not pdf_paths:
            logger.warning(f"No PDF files found in {directory}")
            return []
        
        return await self.process_pdfs_batch(pdf_paths)
    
    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
