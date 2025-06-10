"""
Configuration settings for the PDF search engine.
"""
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Application configuration settings."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = BASE_DIR / "cache"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128  # 25% of chunk size
    
    # Search weights
    VECTOR_WEIGHT = 0.3
    KEYWORD_WEIGHT = 0.7
    
    # ChromaDB settings
    CHROMA_PERSIST_DIR = str(CACHE_DIR / "chromadb")
    COLLECTION_NAME = "pdf_documents"
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_WORKERS = int(os.getenv("API_WORKERS", 1))
    
    # Search settings
    MAX_RESULTS = 20
    RESULTS_PER_PAGE = 5
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 500
    SEARCH_TIMEOUT = 30.0
    
    # Cache settings
    CACHE_SIZE = 1000
    CACHE_TTL = 3600  # 1 hour
    
    # Processing settings
    MAX_CONCURRENT_PDFS = 5
    BATCH_SIZE = 50
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / "pdf_search.log"
    
    # Redis settings (optional)
    REDIS_URL = os.getenv("REDIS_URL")
    REDIS_ENABLED = bool(REDIS_URL)
    
    # Health check
    HEALTH_CHECK_INTERVAL = 30
    
    @classmethod
    def get_search_params(cls) -> Dict[str, Any]:
        """Get search configuration parameters."""
        return {
            "vector_weight": cls.VECTOR_WEIGHT,
            "keyword_weight": cls.KEYWORD_WEIGHT,
            "max_results": cls.MAX_RESULTS,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        if cls.VECTOR_WEIGHT + cls.KEYWORD_WEIGHT != 1.0:
            raise ValueError("Vector and keyword weights must sum to 1.0")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        if cls.MIN_QUERY_LENGTH >= cls.MAX_QUERY_LENGTH:
            raise ValueError("Min query length must be less than max query length")
        
        return True


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    API_WORKERS = int(os.getenv("API_WORKERS", 4))
    LOG_LEVEL = "INFO"


class TestConfig(Config):
    """Test environment configuration."""
    DEBUG = True
    CACHE_SIZE = 10
    DATA_DIR = Config.BASE_DIR / "test_data"


# Get configuration based on environment
def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()


# Global configuration instance
config = get_config()
config.validate_config()
