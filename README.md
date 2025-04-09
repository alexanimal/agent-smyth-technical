# Tweet RAG Agent API

## Overview

This project provides a FastAPI-based API for a Retrieval-Augmented Generation (RAG) agent. The agent uses LangChain and OpenAI's models to answer user queries based on a knowledge base built from tweet data (expected in JSON format). It features automatic query classification, source attribution, background knowledge base loading, and detailed API documentation.

## Key Features

*   **RAG Implementation:** Answers queries using context retrieved from a tweet knowledge base.
*   **FastAPI Backend:** Asynchronous web framework for high performance.
*   **LangChain Integration:** Leverages LangChain for document loading, splitting, embedding, vector storage (FAISS), and QA chains.
*   **OpenAI Embeddings & Models:** Uses OpenAI for generating embeddings and processing chat queries.
*   **Tweet Data Source:** Builds knowledge base from JSON files containing tweet data (expected in `__mocks__` directory).
*   **Background KB Loading:** Initializes the knowledge base in the background upon startup to avoid blocking API availability.
*   **Persistent Index:** Saves and loads the FAISS vector store index and processed documents to/from disk (`faiss_index` directory) to avoid costly regeneration on restarts.
*   **Automatic API Documentation:** Provides interactive Swagger UI (`/docs`) and ReDoc (`/redoc`) documentation generated automatically from FastAPI and Pydantic models.
*   **Enhanced API Models:** Uses detailed Pydantic models for requests and responses, including examples, descriptions, and validation.
*   **Custom Headers:** Includes informative headers like `X-Request-ID`, `X-Processing-Time`, etc.
*   **Logging & Middleware:** Implements request logging via middleware.

## Prerequisites

*   Python 3.10+
*   `pip` (Python package installer)
*   Git
*   An OpenAI API Key

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt`, create one using `pip freeze > requirements.txt` after installing necessary packages like `fastapi`, `uvicorn`, `langchain`, `langchain-openai`, `faiss-cpu` or `faiss-gpu`, `pydantic`, `python-dotenv`, `pytest`, etc.)*

## Configuration

1.  **Create a `.env` file** in the project root directory.
2.  **Add your OpenAI API Key:**
    ```dotenv
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
3.  **(Optional) Add Production API Key:** If you plan to use API key authentication in production (as hinted in `main.py`), add:
    ```dotenv
    API_KEY="your_secure_api_key_for_production"
    ENVIRONMENT="development" # Set to "production" for deployment
    ```
4.  **Prepare Tweet Data:**
    *   Ensure you have your tweet data (in JSON format, with each file containing a list of tweet objects) placed inside a directory named `__mocks__` in the project root. The `kb.py` script will load data from here.

## Running the Application (Development)

To run the FastAPI application locally for development:

```bash
uvicorn app.main:app --host localhost --port 8002 --reload
```

*   `app.main:app`: Points to the FastAPI `app` instance in `app/main.py`.
*   `--host localhost`: Makes the server available only on your local machine.
*   `--port 8002`: Specifies the port to run on.
*   `--reload`: Enables auto-reloading when code changes are detected.

The API will be available at `http://localhost:8002`.

*   **Interactive Docs (Swagger):** `http://localhost:8002/docs`
*   **Alternative Docs (ReDoc):** `http://localhost:8002/redoc`

The first time you run the application, it will:
1.  Attempt to load JSON files from the `__mocks__` directory.
2.  Process the documents, generate embeddings (this might take time and cost OpenAI credits).
3.  Create a FAISS index and save it along with processed documents to the `./faiss_index/` directory.
Subsequent runs will load the index directly from `./faiss_index/`, skipping the embedding step, provided the index files are found and valid.

## API Usage

The primary endpoint is `/chat`.

**Example using `curl`:**

```bash
curl -X 'POST' \
  'http://localhost:8002/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your_secure_api_key_for_production' \
  -d '{
  "message": "What are the latest developments for AAPL?",
  "num_results": 5,
  "context": {"user_id": "user123"}
}'
```

Refer to the `/docs` endpoint for detailed information on all endpoints (`/`, `/health`, `/chat`, `/chat/analyze`), request/response models, and headers.

## Testing

To run the test suite:

```bash
# Ensure your project root is in the python path for imports to work
export PYTHONPATH=$(pwd):$PYTHONPATH # Linux/macOS
# set PYTHONPATH=%cd%;%PYTHONPATH% # Windows CMD
# $env:PYTHONPATH = "$pwd;$env:PYTHONPATH" # Windows PowerShell

# Run tests
pytest tests/

# Run tests with coverage report
pytest --cov=app tests/
```

## Building and Packaging

There are two main ways to package this application:

**1. Using Docker (Recommended for Web Services/Deployment)**

This is the standard way to package web applications for deployment.

*   **Create a `Dockerfile`:** Add a file named `Dockerfile` (no extension) to your project root. Here's a common example for FastAPI:

    ```dockerfile
    # Use an official Python runtime as a parent image
    FROM python:3.10-slim

    # Set the working directory in the container
    WORKDIR /app

    # Prevent Python from writing pyc files to disc
    ENV PYTHONDONTWRITEBYTECODE 1
    # Ensure Python output is sent straight to terminal without buffering
    ENV PYTHONUNBUFFERED 1

    # Install system dependencies if needed (e.g., for FAISS or other libraries)
    # RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

    # Install Poetry or just use pip
    # RUN pip install --upgrade pip
    # If using requirements.txt:
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy the application code into the container
    COPY ./app /app/app
    # Copy mocks and potentially the pre-built index if desired
    # COPY ./__mocks__ /app/__mocks__
    # COPY ./faiss_index /app/faiss_index

    # Make port 8002 available to the world outside this container
    EXPOSE 8002

    # Define environment variable defaults (can be overridden)
    ENV ENVIRONMENT=production
    ENV PORT=8002
    # Note: OPENAI_API_KEY should be passed securely at runtime, not hardcoded

    # Run uvicorn server when the container launches
    # Use 0.0.0.0 to listen on all available interfaces inside the container
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
    ```

*   **Build the Docker Image:**
    ```bash
    docker build -t tweet-rag-api .
    ```
    (Replace `tweet-rag-api` with your desired image name).

*   **Run the Docker Container:**
    ```bash
    docker run -d --name my-rag-app \
      -p 8002:8002 \
      -e OPENAI_API_KEY="your_openai_api_key_here" \
      -e API_KEY="your_secure_api_key_for_production" \
      -e ENVIRONMENT="production" \
      --mount type=bind,source="$(pwd)"/faiss_index,target=/app/faiss_index \
      --mount type=bind,source="$(pwd)"/__mocks__,target=/app/__mocks__ \
      tweet-rag-api
    ```
    *   `-d`: Run in detached mode.
    *   `--name my-rag-app`: Assign a name to the container.
    *   `-p 8002:8002`: Map port 8002 on your host to port 8002 in the container.
    *   `-e`: Pass environment variables securely.
    *   `--mount`: (Optional but recommended) Mount local directories for the index and mocks into the container. This allows the container to use the index you built locally or build it persistently outside the container's ephemeral filesystem. If you copied these into the image during the build, you might not need the mounts unless the data changes frequently.

**2. Using Standard Python Packaging (Wheels/Sdist)**

This creates standard Python package files (`.whl`, `.tar.gz`) which can be installed using `pip`. This is less common for *deploying* a web service directly but useful for distributing libraries.

*   **Install the build tool:**
    ```bash
    pip install build
    ```
*   **Ensure you have a `pyproject.toml`:** This file defines build system requirements and project metadata. You likely have one already if you use `pytest`. Ensure it has a `[project]` section with metadata like `name`, `version`, etc., and a `[build-system]` section. Example:

    ```toml
    [build-system]
    requires = ["setuptools>=61.0"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "tweet_rag_api"
    version = "1.0.0"
    authors = [
      { name="Your Name", email="your@email.com" },
    ]
    description = "A RAG agent API for tweets."
    readme = "README.md"
    requires-python = ">=3.10"
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
    ]
    dependencies = [
        # List all your dependencies from requirements.txt here
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.20.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.1.0",
        "faiss-cpu", # or faiss-gpu
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        # ... other dependencies
    ]

    [project.optional-dependencies]
    dev = [
        "pytest",
        "pytest-cov",
        # ... other dev dependencies
    ]

    [project.urls]
    "Homepage" = "https://github.com/yourusername/yourproject" # Example
    "Bug Tracker" = "https://github.com/yourusername/yourproject/issues" # Example
    ```

*   **Run the build command:**
    ```bash
    python -m build
    ```
    This will create a `dist` directory containing a `.whl` (wheel) file and a `.tar.gz` (source distribution) file.

*   **Usage:** While you *can* install this package (`pip install dist/tweet_rag_api-1.0.0-py3-none-any.whl`), you still need a separate process (like `uvicorn app.main:app ...`) to run the actual web server. This is why Docker is generally preferred for deploying web applications.

## Deployment Notes

*   For production, use a production-grade ASGI server like Uvicorn managed by a process manager (e.g., Gunicorn with Uvicorn workers, Supervisor, systemd).
*   Set the `ENVIRONMENT` environment variable to `production`.
*   Ensure proper API key management and security.
*   Configure CORS (`allow_origins`) restrictively for your frontend domain(s).
*   Consider placing the application behind a reverse proxy like Nginx or Traefik for handling SSL, load balancing, and serving static files if needed.

## Contributing

[Optional: Add guidelines for contributing if this is an open project.]

## License

[Optional: Specify the project's license, e.g., MIT, Apache 2.0.]
