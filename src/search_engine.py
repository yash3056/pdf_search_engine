"""
Hybrid search engine combining vector search and keyword search.
"""
import asyncio
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from loguru import logger
from cachetools import TTLCache
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from config import config
from src.pdf_processor import PDFDocument, DocumentChunk

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)


@dataclass
class SearchResult:
    """Represents a search result."""
    pdf_name: str
    content: str
    page_number: int
    confidence_score: float
    chunk_id: str
    metadata: Dict[str, Any]
    highlight_snippet: str = ""
    vector_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0


class HybridSearchEngine:
    """Hybrid search engine combining vector and keyword search."""
    
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.bm25_index = None
        self.document_chunks: List[DocumentChunk] = []
        self.chunk_lookup: Dict[str, DocumentChunk] = {}
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Cache for search results
        self.search_cache = TTLCache(maxsize=config.CACHE_SIZE, ttl=config.CACHE_TTL)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_chromadb()
        
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.success("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            logger.info("Initializing ChromaDB...")
            
            # Create ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=config.CHROMA_PERSIST_DIR,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=config.COLLECTION_NAME
                )
                logger.info(f"Found existing collection: {config.COLLECTION_NAME}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=config.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {config.COLLECTION_NAME}")
            
            logger.success("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 indexing.
        
        Args:
            text: Input text
            
        Returns:
            List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stop words and non-alphabetic tokens
        tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token.isalpha() and token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        try:
            if not self.embedding_model:
                raise RuntimeError("Embedding model not initialized")
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    async def index_documents(self, pdf_documents: List[PDFDocument]) -> bool:
        """
        Index PDF documents for search.
        
        Args:
            pdf_documents: List of PDFDocument objects
            
        Returns:
            True if indexing successful
        """
        try:
            logger.info(f"Starting fresh indexing of {len(pdf_documents)} PDF documents...")
            
            # Clear existing indexes first
            self.clear_existing_index()
            
            # Collect all chunks
            all_chunks = []
            for pdf_doc in pdf_documents:
                all_chunks.extend(pdf_doc.chunks)
            
            if not all_chunks:
                logger.warning("No chunks to index")
                return False
            
            self.document_chunks = all_chunks
            self.chunk_lookup = {chunk.chunk_id: chunk for chunk in all_chunks}
            
            logger.info(f"Indexing {len(all_chunks)} document chunks...")
            
            # Index in ChromaDB (vector search)
            await self._index_vector_search(all_chunks)
            
            # Index in BM25 (keyword search)
            await self._index_keyword_search(all_chunks)
            
            # Save index metadata
            self._save_index_metadata(pdf_documents)
            
            logger.success(f"Successfully indexed {len(all_chunks)} chunks from {len(pdf_documents)} PDFs")
            return True
            
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            return False
    
    async def _index_vector_search(self, chunks: List[DocumentChunk]):
        """Index chunks in ChromaDB for vector search."""
        try:
            logger.info("Creating vector embeddings...")
            
            # Prepare data for ChromaDB
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            
            # Create embeddings in batches
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                batch_embeddings = self.create_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings.tolist())
                logger.info(f"Created embeddings for batch {i//batch_size + 1}/{math.ceil(len(chunk_texts)/batch_size)}")
            
            # Prepare metadata
            metadatas = []
            for chunk in chunks:
                metadata = {
                    'pdf_name': chunk.pdf_name,
                    'page_number': chunk.page_number,
                    'chunk_index': chunk.metadata.get('chunk_index', 0),
                    'file_path': chunk.metadata.get('file_path', ''),
                    'chunk_length': len(chunk.content)
                }
                metadatas.append(metadata)
            
            # Clear existing collection
            try:
                if self.collection:
                    self.collection.delete()
                    logger.info("Cleared existing vector index")
            except Exception:
                pass
            
            # Add documents to ChromaDB in batches
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                
                if self.collection:
                    self.collection.add(
                        embeddings=all_embeddings[i:end_idx],
                        documents=chunk_texts[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                        ids=chunk_ids[i:end_idx]
                    )
                    logger.info(f"Indexed vector batch {i//batch_size + 1}/{math.ceil(len(chunks)/batch_size)}")
            
            logger.success("Vector indexing completed")
            
        except Exception as e:
            logger.error(f"Error in vector indexing: {e}")
            raise
    
    async def _index_keyword_search(self, chunks: List[DocumentChunk]):
        """Index chunks for BM25 keyword search."""
        try:
            logger.info("Creating BM25 keyword index...")
            
            # Preprocess all chunk texts
            processed_texts = []
            for chunk in chunks:
                processed_tokens = self.preprocess_text(chunk.content)
                processed_texts.append(processed_tokens)
            
            # Create BM25 index
            self.bm25_index = BM25Okapi(processed_texts)
            
            # Save BM25 index
            bm25_path = Path(config.CACHE_DIR) / "bm25_index.pkl"
            with open(bm25_path, 'wb') as f:
                pickle.dump(self.bm25_index, f)
            
            logger.success("BM25 keyword indexing completed")
            
        except Exception as e:
            logger.error(f"Error in keyword indexing: {e}")
            raise
    
    def _save_index_metadata(self, pdf_documents: List[PDFDocument]):
        """Save index metadata for future reference."""
        try:
            metadata = {
                'total_documents': len(pdf_documents),
                'total_chunks': len(self.document_chunks),
                'pdf_files': [
                    {
                        'filename': doc.filename,
                        'file_hash': doc.file_hash,
                        'chunk_count': len(doc.chunks),
                        'file_path': str(doc.file_path)
                    }
                    for doc in pdf_documents
                ],
                'config': config.get_search_params(),
                'model_name': config.EMBEDDING_MODEL
            }
            
            metadata_path = Path(config.CACHE_DIR) / "index_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved index metadata to {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save index metadata: {e}")
    
    def load_existing_index(self) -> bool:
        """Load existing index from cache."""
        try:
            # Check if ChromaDB collection exists and has data
            if not self.collection or self.collection.count() == 0:
                logger.info("No existing vector index found")
                return False
            
            # Load BM25 index
            bm25_path = Path(config.CACHE_DIR) / "bm25_index.pkl"
            if not bm25_path.exists():
                logger.info("No existing BM25 index found")
                return False
            
            with open(bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            
            # Load index metadata
            metadata_path = Path(config.CACHE_DIR) / "index_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded index with {metadata['total_chunks']} chunks from {metadata['total_documents']} documents")
            
            logger.success("Successfully loaded existing search index")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            return False
    
    def _create_cache_key(self, query: str, max_results: int) -> str:
        """Create cache key for search query."""
        return hashlib.md5(f"{query}_{max_results}".encode()).hexdigest()
    
    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if max_results is None:
            max_results = config.MAX_RESULTS
        
        # Check cache first
        cache_key = self._create_cache_key(query, max_results)
        if cache_key in self.search_cache:
            logger.debug(f"Returning cached results for query: {query}")
            return self.search_cache[cache_key]
        
        try:
            logger.info(f"Performing hybrid search for: '{query}'")
            
            # Perform vector search and keyword search in parallel
            vector_task = asyncio.create_task(self._vector_search(query, max_results * 2))
            keyword_task = asyncio.create_task(self._keyword_search(query, max_results * 2))
            
            vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)
            
            # Combine and rank results
            combined_results = self._combine_results(
                vector_results, keyword_results, query, max_results
            )
            
            # Add to cache
            self.search_cache[cache_key] = combined_results
            
            logger.info(f"Search completed: {len(combined_results)} results for '{query}'")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    async def _vector_search(self, query: str, max_results: int) -> List[Tuple[str, float]]:
        """Perform vector similarity search."""
        try:
            if not self.collection:
                return []
            
            # Create query embedding
            if not self.embedding_model:
                raise RuntimeError("Embedding model not initialized")
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max_results, self.collection.count())
            )
            
            # Extract chunk IDs and scores
            vector_results = []
            if results['ids'] and results['distances']:
                for chunk_id, distance in zip(results['ids'][0], results['distances'][0]):
                    # Convert distance to similarity score (ChromaDB returns cosine distance)
                    similarity = 1 - distance
                    vector_results.append((chunk_id, similarity))
            
            logger.debug(f"Vector search found {len(vector_results)} results")
            return vector_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    async def _keyword_search(self, query: str, max_results: int) -> List[Tuple[str, float]]:
        """Perform BM25 keyword search."""
        try:
            if not self.bm25_index:
                return []
            
            # Preprocess query
            query_tokens = self.preprocess_text(query)
            
            if not query_tokens:
                return []
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top results
            scored_chunks = [
                (self.document_chunks[i].chunk_id, scores[i])
                for i in range(len(scores))
                if scores[i] > 0
            ]
            
            # Sort by score and take top results
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            keyword_results = scored_chunks[:max_results]
            
            logger.debug(f"Keyword search found {len(keyword_results)} results")
            return keyword_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _combine_results(
        self, 
        vector_results: List[Tuple[str, float]], 
        keyword_results: List[Tuple[str, float]], 
        query: str, 
        max_results: int
    ) -> List[SearchResult]:
        """Combine and rank vector and keyword search results."""
        try:
            # Normalize scores
            vector_scores = self._normalize_scores([score for _, score in vector_results])
            keyword_scores = self._normalize_scores([score for _, score in keyword_results])
            
            # Create score dictionaries
            vector_score_dict = {
                chunk_id: score 
                for (chunk_id, _), score in zip(vector_results, vector_scores)
            }
            keyword_score_dict = {
                chunk_id: score 
                for (chunk_id, _), score in zip(keyword_results, keyword_scores)
            }
            
            # Get all unique chunk IDs
            all_chunk_ids = set(vector_score_dict.keys()) | set(keyword_score_dict.keys())
            
            # Calculate combined scores
            combined_scores = []
            for chunk_id in all_chunk_ids:
                if chunk_id not in self.chunk_lookup:
                    continue
                
                vector_score = vector_score_dict.get(chunk_id, 0.0)
                keyword_score = keyword_score_dict.get(chunk_id, 0.0)
                
                # Weighted combination
                combined_score = (
                    config.VECTOR_WEIGHT * vector_score + 
                    config.KEYWORD_WEIGHT * keyword_score
                )
                
                combined_scores.append((chunk_id, combined_score, vector_score, keyword_score))
            
            # Sort by combined score
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Create search results
            search_results = []
            seen_chunks = set()
            
            for chunk_id, combined_score, vector_score, keyword_score in combined_scores[:max_results]:
                if chunk_id in seen_chunks:
                    continue
                
                chunk = self.chunk_lookup[chunk_id]
                
                # Create highlight snippet
                highlight_snippet = self._create_highlight_snippet(chunk.content, query)
                
                result = SearchResult(
                    pdf_name=chunk.pdf_name,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    confidence_score=combined_score,
                    chunk_id=chunk_id,
                    metadata=chunk.metadata,
                    highlight_snippet=highlight_snippet,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    combined_score=combined_score
                )
                
                search_results.append(result)
                seen_chunks.add(chunk_id)
            
            # Deduplicate results from same page/document if needed
            search_results = self._deduplicate_results(search_results)
            
            return search_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return []
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _create_highlight_snippet(self, content: str, query: str, snippet_length: int = 200) -> str:
        """Create a highlighted snippet around query matches."""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Find query position
        query_pos = content_lower.find(query_lower)
        
        if query_pos == -1:
            # If exact query not found, try individual words
            query_words = query_lower.split()
            best_pos = -1
            for word in query_words:
                word_pos = content_lower.find(word)
                if word_pos != -1:
                    best_pos = word_pos
                    break
            query_pos = best_pos
        
        if query_pos == -1:
            # Return beginning of content if no match found
            return content[:snippet_length] + "..." if len(content) > snippet_length else content
        
        # Calculate snippet boundaries
        start = max(0, query_pos - snippet_length // 2)
        end = min(len(content), start + snippet_length)
        
        # Adjust start if we're at the end
        if end == len(content):
            start = max(0, end - snippet_length)
        
        snippet = content[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results from same document/page."""
        seen = set()
        deduplicated = []
        
        for result in results:
            # Create a key based on PDF name and page number
            key = f"{result.pdf_name}_{result.page_number}"
            
            if key not in seen:
                deduplicated.append(result)
                seen.add(key)
        
        return deduplicated
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index."""
        try:
            stats = {
                'total_chunks': len(self.document_chunks),
                'vector_index_count': self.collection.count() if self.collection else 0,
                'bm25_index_ready': self.bm25_index is not None,
                'cache_size': len(self.search_cache),
                'embedding_model': config.EMBEDDING_MODEL
            }
            
            # Load metadata if available
            metadata_path = Path(config.CACHE_DIR) / "index_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                stats.update({
                    'total_documents': metadata.get('total_documents', 0),
                    'pdf_files': metadata.get('pdf_files', [])
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the search cache."""
        self.search_cache.clear()
        logger.info("Search cache cleared")
    
    def clear_existing_index(self):
        """Clear all existing indexes to start fresh."""
        try:
            logger.info("🗑️ Clearing existing indexes...")
            
            # Clear in-memory data
            self.document_chunks = []
            self.chunk_lookup = {}
            self.search_cache.clear()
            
            # Clear ChromaDB collection
            if self.collection:
                try:
                    # Delete all documents in the collection
                    collection_count = self.collection.count()
                    if collection_count > 0:
                        logger.info(f"🗑️ Clearing {collection_count} documents from vector index...")
                        self.collection.delete()  # Delete all documents
                        logger.info("✅ Vector index cleared")
                    else:
                        logger.info("🔍 Vector index already empty")
                except Exception as e:
                    logger.warning(f"Could not clear vector index: {e}")
            
            # Clear BM25 index
            self.bm25_index = None
            bm25_path = Path(config.CACHE_DIR) / "bm25_index.pkl"
            if bm25_path.exists():
                bm25_path.unlink()
                logger.info("🗑️ BM25 index file deleted")
            
            # Clear metadata
            metadata_path = Path(config.CACHE_DIR) / "index_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info("🗑️ Index metadata deleted")
            
            logger.success("✅ All existing indexes cleared")
            
        except Exception as e:
            logger.error(f"Error clearing indexes: {e}")
