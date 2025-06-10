"""
Main application entry point for the PDF search engine.
"""
import asyncio
import sys
import signal
import threading
import time
from pathlib import Path
from typing import Optional
import subprocess

from loguru import logger
import uvicorn

from config import config
from src.pdf_processor import PDFProcessor
from src.search_engine import HybridSearchEngine


class PDFSearchApplication:
    """Main application orchestrator."""
    
    def __init__(self):
        self.pdf_processor: Optional[PDFProcessor] = None
        self.search_engine: Optional[HybridSearchEngine] = None
        self.api_process: Optional[subprocess.Popen] = None
        self.ui_process: Optional[subprocess.Popen] = None
        self.shutdown_requested = False
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Configure application logging."""
        # Remove default logger
        logger.remove()
        
        # Add console logger
        logger.add(
            sys.stderr,
            level=config.LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Add file logger
        logger.add(
            config.LOG_FILE,
            level=config.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="1 day",
            retention="7 days",
            compression="zip"
        )
        
        logger.info("Logging configured successfully")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def initialize_components(self):
        """Initialize PDF processor and search engine."""
        try:
            logger.info("Initializing application components...")
            
            # Initialize PDF processor
            self.pdf_processor = PDFProcessor()
            logger.success("PDF processor initialized")
            
            # Initialize search engine
            self.search_engine = HybridSearchEngine()
            logger.success("Search engine initialized")
            
            # Try to load existing index
            if self.search_engine.load_existing_index():
                logger.info("Loaded existing search index")
            else:
                logger.info("No existing index found")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    async def auto_index_pdfs(self):
        """Automatically index PDFs if needed."""
        try:
            # Check if components are initialized
            if not self.pdf_processor or not self.search_engine:
                logger.error("Components not initialized")
                return
            
            # Check if data directory exists and has PDFs
            if not config.DATA_DIR.exists():
                logger.warning(f"Data directory does not exist: {config.DATA_DIR}")
                config.DATA_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created data directory: {config.DATA_DIR}")
                return
            
            # Discover PDF files
            pdf_files = self.pdf_processor.discover_pdfs(config.DATA_DIR)
            
            if not pdf_files:
                logger.warning("No PDF files found in data directory")
                logger.info(f"Place PDF files in: {config.DATA_DIR}")
                return
            
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            # Check if we need to reindex
            index_stats = self.search_engine.get_index_stats()
            if index_stats.get('total_chunks', 0) > 0:
                logger.info("Search index already exists, skipping auto-indexing")
                logger.info("Use the API endpoint or UI to reindex if needed")
                return
            
            logger.info("Starting automatic PDF indexing...")
            
            # Process and index PDFs
            pdf_documents = await self.pdf_processor.process_pdfs_batch(pdf_files)
            
            if pdf_documents:
                success = await self.search_engine.index_documents(pdf_documents)
                
                if success:
                    total_chunks = sum(len(doc.chunks) for doc in pdf_documents)
                    logger.success(f"Successfully indexed {len(pdf_documents)} PDFs with {total_chunks} chunks")
                else:
                    logger.error("Failed to index PDF documents")
            else:
                logger.warning("No PDF documents were successfully processed")
                
        except Exception as e:
            logger.error(f"Error during auto-indexing: {e}")
    
    def start_api_server(self):
        """Start the FastAPI server in a separate process."""
        try:
            logger.info(f"Starting API server on {config.API_HOST}:{config.API_PORT}")
            
            # Check if we're in a uv environment
            import shutil
            if shutil.which("uv"):
                cmd = [
                    "uv", "run", "uvicorn",
                    "src.api:app",
                    "--host", config.API_HOST,
                    "--port", str(config.API_PORT),
                    "--workers", str(config.API_WORKERS),
                    "--log-level", config.LOG_LEVEL.lower()
                ]
            else:
                cmd = [
                    sys.executable, "-m", "uvicorn",
                    "src.api:app",
                    "--host", config.API_HOST,
                    "--port", str(config.API_PORT),
                    "--workers", str(config.API_WORKERS),
                    "--log-level", config.LOG_LEVEL.lower()
                ]
            
            self.api_process = subprocess.Popen(cmd, cwd=Path.cwd())
            logger.success("API server started")
            
            # Wait a moment for the server to start
            time.sleep(5)  # Increased wait time
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise
    
    def start_ui_server(self):
        """Start the Streamlit UI in a separate process."""
        try:
            ui_port = 8501  # Streamlit default port
            logger.info(f"Starting UI server on port {ui_port}")
            
            # Check if we're in a uv environment
            import shutil
            if shutil.which("uv"):
                cmd = [
                    "uv", "run", "streamlit", "run",
                    "src/ui.py",
                    "--server.port", str(ui_port),
                    "--server.address", "0.0.0.0",
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false"
                ]
            else:
                cmd = [
                    sys.executable, "-m", "streamlit", "run",
                    "src/ui.py",
                    "--server.port", str(ui_port),
                    "--server.address", "0.0.0.0",
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false"
                ]
            
            self.ui_process = subprocess.Popen(cmd, cwd=Path.cwd())
            logger.success(f"UI server started on http://localhost:{ui_port}")
            
        except Exception as e:
            logger.error(f"Failed to start UI server: {e}")
            # UI failure is not critical, continue without it
    
    def stop_servers(self):
        """Stop all server processes."""
        logger.info("Stopping servers...")
        
        if self.api_process:
            try:
                self.api_process.terminate()
                self.api_process.wait(timeout=10)
                logger.info("API server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("API server did not stop gracefully, killing...")
                self.api_process.kill()
            except Exception as e:
                logger.error(f"Error stopping API server: {e}")
        
        if self.ui_process:
            try:
                self.ui_process.terminate()
                self.ui_process.wait(timeout=10)
                logger.info("UI server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("UI server did not stop gracefully, killing...")
                self.ui_process.kill()
            except Exception as e:
                logger.error(f"Error stopping UI server: {e}")
    
    def health_check_loop(self):
        """Continuous health check of services."""
        while not self.shutdown_requested:
            try:
                # Check API process
                if self.api_process and self.api_process.poll() is not None:
                    logger.error("API server process died unexpectedly")
                    break
                
                # Check UI process (optional)
                if self.ui_process and self.ui_process.poll() is not None:
                    logger.warning("UI server process died")
                    self.ui_process = None
                
                time.sleep(config.HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                break
    
    async def run(self):
        """Run the complete PDF search application."""
        try:
            logger.info("Starting PDF Search Engine Application")
            logger.info(f"Configuration: {config.__class__.__name__}")
            logger.info(f"Data directory: {config.DATA_DIR}")
            logger.info(f"Cache directory: {config.CACHE_DIR}")
            
            # Initialize components
            if not await self.initialize_components():
                logger.error("Failed to initialize application components")
                return False
            
            # Auto-index PDFs
            await self.auto_index_pdfs()
            
            # Start API server
            self.start_api_server()
            
            # Start UI server
            self.start_ui_server()
            
            # Start health check in a separate thread
            health_thread = threading.Thread(target=self.health_check_loop, daemon=True)
            health_thread.start()
            
            # Print startup information
            print("\n" + "="*50)
            print("🔍 PDF Search Engine Started Successfully!")
            print("="*50)
            print(f"📡 API Server: http://localhost:{config.API_PORT}")
            print(f"📱 Web Interface: http://localhost:8501")
            print(f"📚 API Documentation: http://localhost:{config.API_PORT}/docs")
            print(f"📁 Data Directory: {config.DATA_DIR}")
            print(f"🧠 Embedding Model: {config.EMBEDDING_MODEL}")
            print("="*50)
            print("Press Ctrl+C to stop the application")
            print("="*50 + "\n")
            
            # Wait for shutdown signal
            while not self.shutdown_requested:
                await asyncio.sleep(1)
            
            logger.info("Shutdown requested, stopping application...")
            return True
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            return True
        except Exception as e:
            logger.error(f"Application error: {e}")
            return False
        finally:
            self.stop_servers()
            logger.info("PDF Search Engine Application stopped")


def main():
    """Main entry point."""
    try:
        app = PDFSearchApplication()
        success = asyncio.run(app.run())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
