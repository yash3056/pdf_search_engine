"""
Direct API server runner for Docker deployment.
"""
import uvicorn
from config import config

if __name__ == "__main__":
    # Use import string for multiple workers support
    uvicorn.run(
        "src.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        log_level=config.LOG_LEVEL.lower()
    )
