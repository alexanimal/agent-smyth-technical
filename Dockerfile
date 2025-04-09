# Use Python 3.10 as base image for compatibility with all dependencies
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies including jq which is needed for JSONLoader
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    jq \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -sSf https://install.ultraviolet.dev | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy project configuration
COPY pyproject.toml README.md ./

# Copy source code
COPY app/ ./app/
COPY data/ ./data/
COPY .env ./

# Create directory for the FAISS index
RUN mkdir -p /app/faiss_index && chmod 777 /app/faiss_index

# Install dependencies with UV
RUN uv pip install .

# Create entry script
RUN echo '#!/bin/bash\n\
set -e\n\
# Load environment variables from .env file\n\
export $(grep -v "^#" /app/.env | xargs)\n\
# Start the application\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8002\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8002

# Run app
ENTRYPOINT ["/app/entrypoint.sh"]