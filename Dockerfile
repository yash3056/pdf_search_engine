FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the recommended copy method (faster and more reliable)
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

# Copy dependency files first to leverage Docker cache
COPY . .

# Install Python dependencies using uv
RUN uv sync --frozen

# Download NLTK data using uv run
RUN uv run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data cache logs

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uv", "run", "python", "main.py"]
