"""
Direct API server runner for Docker deployment.
"""
import uvicorn
from src.api import app
from config import config

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        log_level=config.LOG_LEVEL.lower()
    )
