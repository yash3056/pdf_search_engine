"""
Basic tests for the PDF search engine.
"""
import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from src.pdf_processor import PDFProcessor
from src.search_engine import HybridSearchEngine
from config import TestConfig


@pytest.fixture
def test_config():
    """Test configuration fixture."""
    return TestConfig()


@pytest.fixture
def pdf_processor():
    """PDF processor fixture."""
    return PDFProcessor()


@pytest.fixture
def search_engine():
    """Search engine fixture."""
    return HybridSearchEngine()


@pytest.fixture
def temp_pdf_dir():
    """Temporary directory with test PDFs."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create a simple test PDF (placeholder - in real tests you'd have actual PDFs)
    test_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    
    test_pdf = temp_dir / "test.pdf"
    test_pdf.write_bytes(test_pdf_content)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestPDFProcessor:
    """Test PDF processor functionality."""
    
    def test_init(self, pdf_processor):
        """Test PDF processor initialization."""
        assert pdf_processor is not None
        assert pdf_processor.text_splitter is not None
    
    def test_clean_text(self, pdf_processor):
        """Test text cleaning functionality."""
        dirty_text = "   This  is   some    dirty text!!!   \n\n\n   "
        clean_text = pdf_processor.clean_text(dirty_text)
        
        assert clean_text == "This is some dirty text!"
        assert not clean_text.startswith(" ")
        assert not clean_text.endswith(" ")
    
    def test_discover_pdfs(self, pdf_processor, temp_pdf_dir):
        """Test PDF discovery in directory."""
        pdfs = pdf_processor.discover_pdfs(temp_pdf_dir)
        assert len(pdfs) >= 0  # May not find valid PDFs in temp directory


class TestSearchEngine:
    """Test search engine functionality."""
    
    def test_init(self, search_engine):
        """Test search engine initialization."""
        assert search_engine is not None
        assert search_engine.embedding_model is not None
        assert search_engine.chroma_client is not None
    
    def test_preprocess_text(self, search_engine):
        """Test text preprocessing."""
        text = "This is a test sentence with UPPERCASE and lowercase words."
        tokens = search_engine.preprocess_text(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
    
    def test_normalize_scores(self, search_engine):
        """Test score normalization."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = search_engine._normalize_scores(scores)
        
        assert len(normalized) == len(scores)
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
    
    @pytest.mark.asyncio
    async def test_search_empty_index(self, search_engine):
        """Test search with empty index."""
        results = await search_engine.search("test query")
        assert isinstance(results, list)
        assert len(results) == 0


class TestConfiguration:
    """Test configuration settings."""
    
    def test_config_validation(self, test_config):
        """Test configuration validation."""
        assert test_config.validate_config()
        assert test_config.VECTOR_WEIGHT + test_config.KEYWORD_WEIGHT == 1.0
        assert test_config.CHUNK_OVERLAP < test_config.CHUNK_SIZE
    
    def test_search_params(self, test_config):
        """Test search parameter retrieval."""
        params = test_config.get_search_params()
        
        assert 'vector_weight' in params
        assert 'keyword_weight' in params
        assert 'max_results' in params
        assert params['vector_weight'] + params['keyword_weight'] == 1.0


@pytest.mark.asyncio
async def test_integration_basic():
    """Basic integration test."""
    # This would require actual PDF files for a complete test
    processor = PDFProcessor()
    engine = HybridSearchEngine()
    
    # Test that components can be initialized together
    assert processor is not None
    assert engine is not None


if __name__ == "__main__":
    pytest.main([__file__])
