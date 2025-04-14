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

# Install Poetry
RUN pip install poetry

# Configure Poetry
RUN poetry config virtualenvs.create false

# Copy project configuration
COPY pyproject.toml poetry.lock ./

# Copy source code
COPY app/ ./app/
COPY data/ ./data/

# Create directory for the FAISS index
RUN mkdir -p /app/faiss_index && chmod 777 /app/faiss_index

# Install dependencies with Poetry
RUN poetry install --without dev --no-root

# Create entry script
RUN echo '#!/bin/bash\n\
set -e\n\
# Start the application on the correct port
\
exec uvicorn app.main:app --host 0.0.0.0 --port 8003\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose the correct port
EXPOSE 8003

# Run app
ENTRYPOINT ["/app/entrypoint.sh"]
