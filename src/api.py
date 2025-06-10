"""
FastAPI application for the PDF search engine.
"""
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from loguru import logger
import uvicorn

from config import config
from src.pdf_processor import PDFProcessor
from src.search_engine import HybridSearchEngine, SearchResult


# Pydantic models
class SearchQuery(BaseModel):
    """Search query model."""
    query: str = Field(..., min_length=config.MIN_QUERY_LENGTH, max_length=config.MAX_QUERY_LENGTH)
    max_results: Optional[int] = Field(default=config.MAX_RESULTS, ge=1, le=50)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time: float
    cached: bool = False


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    index_stats: Dict[str, Any]


class IndexResponse(BaseModel):
    """Index response model."""
    status: str
    message: str
    indexed_documents: int
    indexed_chunks: int


# Global instances
search_engine: Optional[HybridSearchEngine] = None
pdf_processor: Optional[PDFProcessor] = None
app_startup_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global search_engine, pdf_processor, app_startup_time
    
    # Startup
    logger.info("Starting PDF search engine API...")
    app_startup_time = time.time()
    
    try:
        # Initialize components
        search_engine = HybridSearchEngine()
        pdf_processor = PDFProcessor()
        
        # Try to load existing index
        if not search_engine.load_existing_index():
            logger.info("🔄 No existing index found, starting automatic indexing...")
            
            # Auto-index PDFs on startup
            try:
                # Check if data directory exists and has PDFs
                if config.DATA_DIR.exists():
                    pdf_documents = await pdf_processor.process_directory(config.DATA_DIR)
                    
                    if pdf_documents:
                        logger.info(f"📚 Found {len(pdf_documents)} PDF documents, indexing...")
                        success = await search_engine.index_documents(pdf_documents)
                        
                        if success:
                            total_chunks = sum(len(doc.chunks) for doc in pdf_documents)
                            logger.success(f"✅ Auto-indexed {len(pdf_documents)} PDFs with {total_chunks} chunks")
                            logger.info("🔍 Search engine is now ready for queries!")
                        else:
                            logger.error("❌ Failed to auto-index documents")
                    else:
                        logger.warning("⚠️ No PDF documents found in data directory")
                        logger.info(f"📁 Place PDF files in: {config.DATA_DIR}")
                        logger.info("💡 Then restart the API or call POST /index to index them")
                else:
                    logger.warning(f"⚠️ Data directory does not exist: {config.DATA_DIR}")
                    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
                    logger.info(f"📁 Created data directory: {config.DATA_DIR}")
                    logger.info("💡 Add PDF files and restart the API or call POST /index")
                    
            except Exception as e:
                logger.error(f"❌ Error during auto-indexing: {e}")
                # Don't fail startup if indexing fails
        else:
            logger.info("✅ Loaded existing search index")
            logger.info("🔍 Search engine is ready for queries!")
        
        logger.success("PDF search engine API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down PDF search engine API...")


# Create FastAPI app
app = FastAPI(
    title="PDF Search Engine API",
    description="Hybrid search engine for PDF documents using vector and keyword search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_search_engine() -> HybridSearchEngine:
    """Dependency to get search engine instance."""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    return search_engine


def get_pdf_processor() -> PDFProcessor:
    """Dependency to get PDF processor instance."""
    if pdf_processor is None:
        raise HTTPException(status_code=503, detail="PDF processor not initialized")
    return pdf_processor


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "PDF Search Engine API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(engine: HybridSearchEngine = Depends(get_search_engine)):
    """Health check endpoint."""
    try:
        index_stats = engine.get_index_stats()
        
        return HealthResponse(
            status="healthy",
            timestamp=time.time(),
            index_stats=index_stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/search", response_model=SearchResponse)
async def search_pdfs(
    search_query: SearchQuery,
    engine: HybridSearchEngine = Depends(get_search_engine)
):
    """
    Search PDF documents using hybrid vector and keyword search.
    
    Args:
        search_query: Search query parameters
        
    Returns:
        Search results with metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Search request: '{search_query.query}' (max_results: {search_query.max_results})")
        
        # Check if index exists
        index_stats = engine.get_index_stats()
        if index_stats.get('total_chunks', 0) == 0:
            raise HTTPException(
                status_code=503, 
                detail="Search index is empty. Please index some PDF documents first."
            )
        
        # Perform search with timeout
        try:
            max_results_val = search_query.max_results if search_query.max_results is not None else config.MAX_RESULTS
            search_results = await asyncio.wait_for(
                engine.search(search_query.query, max_results_val),
                timeout=config.SEARCH_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Search request timed out")
        
        # Format results
        formatted_results = []
        for result in search_results:
            formatted_result = {
                "pdf_name": result.pdf_name,
                "content": result.content,
                "page_number": result.page_number,
                "confidence_score": round(result.confidence_score, 4),
                "chunk_id": result.chunk_id,
                "highlight_snippet": result.highlight_snippet,
                "vector_score": round(result.vector_score, 4),
                "keyword_score": round(result.keyword_score, 4),
                "combined_score": round(result.combined_score, 4),
                "metadata": {
                    "file_path": result.metadata.get("file_path", ""),
                    "chunk_index": result.metadata.get("chunk_index", 0),
                    "chunk_length": result.metadata.get("chunk_length", 0)
                }
            }
            formatted_results.append(formatted_result)
        
        search_time = time.time() - start_time
        
        # Check if results were cached
        cache_key = engine._create_cache_key(search_query.query, max_results_val)
        cached = cache_key in engine.search_cache
        
        logger.info(f"Search completed in {search_time:.3f}s: {len(formatted_results)} results")
        
        return SearchResponse(
            query=search_query.query,
            results=formatted_results,
            total_results=len(formatted_results),
            search_time=search_time,
            cached=cached
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search", response_model=SearchResponse)
async def search_pdfs_get(
    q: str = Query(..., min_length=config.MIN_QUERY_LENGTH, max_length=config.MAX_QUERY_LENGTH),
    max_results: int = Query(default=config.MAX_RESULTS, ge=1, le=50),
    engine: HybridSearchEngine = Depends(get_search_engine)
):
    """
    Search PDF documents using GET method for simple queries.
    
    Args:
        q: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Search results with metadata
    """
    search_query = SearchQuery(query=q, max_results=max_results)
    return await search_pdfs(search_query, engine)


@app.post("/index", response_model=IndexResponse)
async def index_pdfs(
    background_tasks: BackgroundTasks,
    processor: PDFProcessor = Depends(get_pdf_processor),
    engine: HybridSearchEngine = Depends(get_search_engine)
):
    """
    Index PDF documents from the data directory.
    
    Returns:
        Indexing status and statistics
    """
    try:
        logger.info("Starting PDF indexing...")
        
        # Check if data directory exists
        if not config.DATA_DIR.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Data directory not found: {config.DATA_DIR}"
            )
        
        # Process PDFs
        pdf_documents = await processor.process_directory(config.DATA_DIR)
        
        if not pdf_documents:
            return IndexResponse(
                status="warning",
                message="No PDF documents found to index",
                indexed_documents=0,
                indexed_chunks=0
            )
        
        # Index documents
        success = await engine.index_documents(pdf_documents)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to index documents")
        
        # Calculate statistics
        total_chunks = sum(len(doc.chunks) for doc in pdf_documents)
        
        logger.success(f"Successfully indexed {len(pdf_documents)} documents with {total_chunks} chunks")
        
        return IndexResponse(
            status="success",
            message=f"Successfully indexed {len(pdf_documents)} PDF documents",
            indexed_documents=len(pdf_documents),
            indexed_chunks=total_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.get("/index/stats", response_model=Dict[str, Any])
async def get_index_stats(engine: HybridSearchEngine = Depends(get_search_engine)):
    """Get current index statistics."""
    try:
        stats = engine.get_index_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get index statistics")


@app.delete("/cache")
async def clear_cache(engine: HybridSearchEngine = Depends(get_search_engine)):
    """Clear the search cache."""
    try:
        engine.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@app.get("/docs-count")
async def get_docs_count():
    """Get count of PDF files in data directory."""
    try:
        if not config.DATA_DIR.exists():
            return {"count": 0, "message": "Data directory not found"}
        
        pdf_files = list(config.DATA_DIR.rglob("*.pdf"))
        return {"count": len(pdf_files), "files": [f.name for f in pdf_files]}
        
    except Exception as e:
        logger.error(f"Error counting documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to count documents")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    # Configure logging
    logger.add(
        config.LOG_FILE,
        rotation="1 day",
        retention="7 days",
        level=config.LOG_LEVEL
    )
    
    # Run the application
    uvicorn.run(
        "src.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        log_level=config.LOG_LEVEL.lower(),
        reload=False
    )
