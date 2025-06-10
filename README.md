# PDF Search Engine

A production-ready hybrid search engine for PDF documents that combines vector search and keyword search to find the most relevant content for user queries.

## Features

- **Hybrid Search**: Combines vector similarity search (30%) with BM25 keyword search (70%)
- **Advanced PDF Processing**: Extracts text, handles metadata, and creates intelligent document chunks
- **High Performance**: Uses ChromaDB with HNSW algorithm for fast vector search
- **Scalable Architecture**: FastAPI backend with Streamlit frontend
- **Caching**: Built-in result caching for improved performance
- **Production Ready**: Docker support, health checks, and comprehensive logging

## Technology Stack

- **Backend**: FastAPI, Python 3.9+
- **Vector Search**: ChromaDB, Sentence-Transformers (all-MiniLM-L6-v2)
- **Keyword Search**: BM25 (rank-bm25)
- **PDF Processing**: pypdf
- **Frontend**: Streamlit
- **Deployment**: Docker, Docker Compose

## Quick Start

### Option 1: Docker (Recommended)

1. **Clone and prepare**:
   ```bash
   git clone <repository>
   cd pdf_search_engine
   mkdir data
   # Copy your PDF files to the data/ directory
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Access the application**:
   - **Web Interface**: http://localhost:8501
   - **API Documentation**: http://localhost:8000/docs
   - **API Endpoint**: http://localhost:8000

### Option 2: Local Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**:
   ```bash
   mkdir data
   # Copy your PDF files to the data/ directory
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

## API Usage

### Search PDFs
```bash
# GET request
curl "http://localhost:8000/search?q=machine%20learning&max_results=10"

# POST request
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "max_results": 10}'
```

### Index Documents
```bash
curl -X POST "http://localhost:8000/index"
```

### Health Check
```bash
curl "http://localhost:8000/health"
```

## Configuration

Edit `config.py` to customize:

- **Search weights**: Adjust vector vs keyword search balance
- **Chunk parameters**: Modify chunk size and overlap
- **Model settings**: Change embedding model
- **Performance settings**: Adjust cache size, timeouts, etc.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │  Search Engine  │
│                 │◄──►│                 │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                        ┌─────────────────┐    ┌─────────────────┐
                        │   PDF Processor │    │    ChromaDB     │
                        │                 │    │   (Vectors)     │
                        └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   BM25 Index    │
                                               │   (Keywords)    │
                                               └─────────────────┘
```

## File Structure

```
pdf_search_engine/
├── src/
│   ├── pdf_processor.py      # PDF text extraction and chunking
│   ├── search_engine.py      # Hybrid search implementation
│   ├── api.py               # FastAPI application
│   └── ui.py                # Streamlit interface
├── data/                    # PDF documents (place your PDFs here)
├── cache/                   # Search index and cache
├── logs/                    # Application logs
├── main.py                  # Application entry point
├── config.py                # Configuration settings
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
└── README.md               # This file
```

## Performance Specifications

- **Search Speed**: < 2 seconds for most queries
- **Concurrent Users**: Supports 10+ concurrent users
- **Document Capacity**: Optimized for 1000+ PDF documents
- **Memory Usage**: Efficient processing for 4GB RAM systems
- **Caching**: LRU cache with configurable TTL

## Search Algorithm

The hybrid search combines:

1. **Vector Search (30%)**:
   - Uses Sentence-BERT embeddings
   - Cosine similarity matching
   - Semantic understanding

2. **Keyword Search (70%)**:
   - BM25 algorithm
   - Term frequency analysis
   - Exact keyword matching

3. **Result Fusion**:
   - Weighted score combination
   - Deduplication
   - Relevance ranking

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ main.py config.py
flake8 src/ main.py config.py
```

### Adding New Features

1. **Custom Models**: Modify `config.py` to use different embedding models
2. **New Processors**: Extend `PDFProcessor` for additional file types
3. **Custom Scoring**: Implement custom ranking algorithms in `SearchEngine`

## Monitoring and Logging

- **Logs**: Stored in `logs/pdf_search.log` with rotation
- **Health Checks**: `/health` endpoint provides system status
- **Metrics**: Index statistics and search performance metrics
- **Error Handling**: Comprehensive error reporting and recovery

## Troubleshooting

### Common Issues

1. **No search results**:
   - Check if PDFs are indexed: `curl http://localhost:8000/index/stats`
   - Reindex documents: `curl -X POST http://localhost:8000/index`

2. **API not responding**:
   - Check logs: `docker-compose logs pdf-search-engine`
   - Verify health: `curl http://localhost:8000/health`

3. **Out of memory**:
   - Reduce `CHUNK_SIZE` in config.py
   - Lower `MAX_CONCURRENT_PDFS`
   - Increase Docker memory limits

### Performance Tuning

1. **For large document collections**:
   - Increase `CACHE_SIZE`
   - Adjust `BATCH_SIZE` for indexing
   - Consider distributed deployment

2. **For faster searches**:
   - Increase `VECTOR_WEIGHT` for semantic search
   - Reduce `MAX_RESULTS`
   - Enable Redis caching

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review application logs
- Open an issue with detailed information
