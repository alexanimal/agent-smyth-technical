# Tweet RAG Agent API

[![codecov](https://codecov.io/github/alexanimal/agent-smyth-technical/graph/badge.svg?token=B2BI5398GF)](https://codecov.io/github/alexanimal/agent-smyth-technical)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
  - [Module Structure](#module-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Development Workflow](#development-workflow)
- [API Usage](#api-usage)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Building and Packaging](#building-and-packaging)
- [Deployment Notes](#deployment-notes)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides a FastAPI-based API for a Retrieval-Augmented Generation (RAG) agent. The agent uses LangGraph, LangChain, and OpenAI's models to answer user queries based on a knowledge base built from tweet data (expected in JSON format). It features automatic query classification, source attribution, background knowledge base loading with persistence, automated code quality checks, CI/CD pipelines, and detailed API documentation.

## Key Features

*   **Advanced RAG Implementation:** Answers queries using context retrieved from a tweet knowledge base with query classification, document ranking, and alternative viewpoint generation.
*   **FastAPI Backend:** Asynchronous web framework for high performance.
*   **LangGraph Workflow:** Directed graph-based workflow for structured RAG processing.
*   **LangChain Integration:** Leverages LangChain for document loading, splitting, embedding, vector storage (FAISS), and QA chains.
*   **OpenAI Embeddings & Models:** Uses OpenAI for generating embeddings and processing chat queries.
*   **Tweet Data Source:** Builds knowledge base from JSON files containing tweet data (expected in `data` directory).
*   **Robust Knowledge Base Management:**
    *   Initializes the KB in the background upon startup.
    *   Uses parallel processing for faster document loading.
    *   Persists the FAISS vector store index and processed documents to disk (`faiss_index` directory) using pickle for efficient restarts.
    *   Includes integrity checks for the saved index.
*   **Poetry Dependency Management:** Uses Poetry for clear dependency declaration and environment management (`pyproject.toml`, `poetry.lock`).
*   **Automated Code Quality:** Integrates `pre-commit` hooks with `black` and `isort` for consistent code formatting and import sorting before commits.
*   **CI/CD Pipelines:** Includes GitHub Actions workflows (`ci.yaml`, `cd.yml`) for automated testing on push/pull requests and deployment on merges to main.
*   **Automatic API Documentation:** Provides interactive Swagger UI (`/docs`) and ReDoc (`/redoc`).
*   **Enhanced API Models:** Uses detailed Pydantic models for requests and responses.
*   **Custom Headers & Middleware:** Includes informative headers and request logging.

## Architecture

The application follows a modular architecture with several key components:

### Module Structure

The application is organized into several modules, each with specific responsibilities:

1. **RAG Module** (`/app/rag`): Core workflow implementation using LangGraph:
   - Query classification
   - Document retrieval and ranking
   - Response generation with alternative viewpoints
   - [View RAG Documentation](/app/rag/README.md)

2. **Routers Module** (`/app/routers`): FastAPI endpoint definitions:
   - Chat request processing
   - Status and health monitoring
   - [View Routers Documentation](/app/routers/README.md)

3. **Schemas Module** (`/app/schemas`): Pydantic models for data validation:
   - Request/response models with detailed validation
   - Enumerated types and configuration schemas
   - [View Schemas Documentation](/app/schemas/README.md)

4. **Utils Module** (`/app/utils`): Utility functions for processing:
   - Document handling and sentiment analysis
   - Technical indicator extraction
   - [View Utils Documentation](/app/utils/README.md)

5. **Services Module** (`/app/services`): Application state and initialization:
   - Knowledge base and chat handler management
   - Dependency injection for FastAPI
   - [View Services Documentation](/app/services/README.md)

6. **Core Module** (`/app/core`): Core business logic components
7. **Config Module** (`/app/config`): Application configuration and settings

## Prerequisites

*   Python >=3.10.12, <4.0 (as defined in `pyproject.toml`)
*   Poetry (Python packaging and dependency management tool)
*   Git
*   An OpenAI API Key
*   (Optional) AWS Credentials and CDK for deployment (`cd.yml`).

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Install Dependencies using Poetry:**
    This command reads the `pyproject.toml` file, resolves dependencies, installs them into a virtual environment managed by Poetry, and installs development dependencies (`pytest`, `black`, `isort`, `pre-commit`, etc.).
    ```bash
    poetry install --with dev
    ```

3.  **Activate the Virtual Environment (Optional but Recommended):**
    To work directly within the environment Poetry created:
    ```bash
    poetry shell
    ```
    Alternatively, prefix commands with `poetry run`.

4.  **Install Pre-commit Hooks:**
    This sets up the git hooks defined in `.pre-commit-config.yaml`. They will run automatically before each commit.
    ```bash
    poetry run pre-commit install
    ```

## Configuration

1.  **Create a `.env` file** in the project root directory.
2.  **Add your OpenAI API Key:**
    ```dotenv
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
3.  **(Optional) Add Production API Key & Environment:**
    ```dotenv
    API_KEY="your_secure_api_key_for_production"
    ENVIRONMENT="development" # Set to "production" for deployment
    ```
4.  **Prepare Tweet Data:**
    *   Ensure you have your tweet data (in JSON format, with each file containing a list of tweet objects) placed inside a directory named `data` in the project root. The `app/kb.py` script will load data from here.
    *   This data is currently in this repository only for demonstration purposes. This would never happen in a production repository unless the data size was very small.

## Running the Application (Development)

To run the FastAPI application locally for development (ensure you have activated the Poetry shell or use `poetry run`):

```bash
poetry run uvicorn app.main:app --host localhost --port 8003 --reload
```

*   `app.main:app`: Points to the FastAPI `app` instance in `app/main.py`.
*   `--host localhost`: Makes the server available only on your local machine.
*   `--port 8003`: Specifies the port to run on (adjust if needed, check `app/main.py`).
*   `--reload`: Enables auto-reloading when code changes are detected.

The API will be available at `http://localhost:8003`.

*   **Interactive Docs (Swagger):** `http://localhost:8003/docs`
*   **Alternative Docs (ReDoc):** `http://localhost:8003/redoc`

The first time you run the application, it will:
1.  Attempt to load JSON files from the `data` directory using parallel processing.
2.  Process the documents, generate embeddings (this might take time and cost OpenAI credits).
3.  Create a FAISS index and save it along with processed documents to the `./faiss_index/` directory.
Subsequent runs will load the index directly from `./faiss_index/`, skipping the embedding step, provided the index files are found and valid.

## Development Workflow

*   **Code Formatting & Imports:** Before committing changes, `pre-commit` hooks will automatically run `black` to format your code and `isort` to sort imports according to the configurations in `pyproject.toml`. If files are modified by the hooks, you'll need to `git add` them again before committing.
*   **Manual Checks:** You can manually run the pre-commit checks on all files:
    ```bash
    poetry run pre-commit run --all-files
    ```
*   **Adding Dependencies:** Use Poetry to add new dependencies:
    ```bash
    poetry add <package_name>
    poetry add --group dev <dev_package_name> # For development dependencies
    ```

## API Usage

The primary endpoint is `/chat`.

**Example using `curl`:**

```bash
curl -X 'POST' \
  'http://localhost:8003/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your_secure_api_key_for_production' \ # Only needed if API Key Auth is enabled
  -d '{
  "message": "What are the latest developments for AAPL?",
  "num_results": 5,
  "context": {"user_id": "user123"}
}'
```

Refer to the `/docs` endpoint for detailed information on all endpoints (`/`, `/health`, `/chat`), request/response models, and headers.

## Testing

To run the test suite (using `pytest`):

```bash
# Ensure you are in the poetry shell or prefix with poetry run
poetry run pytest tests/

# Run tests with coverage report
poetry run pytest --cov=app tests/
```
Poetry automatically manages the `PYTHONPATH`, so manual exporting is not needed.

## CI/CD

This project uses GitHub Actions for Continuous Integration and Continuous Deployment:

*   **CI (`.github/workflows/ci.yaml`):**
    *   Triggered on push/pull requests to `main` affecting relevant paths.
    *   Checks out code, sets up Python and Poetry.
    *   Installs dependencies (`poetry install --with dev`).
    *   Runs linters (`black --check`, `isort --check`), type checking (`mypy`), and unit tests (`pytest`).
    *   Uploads coverage reports to Codecov.
*   **CD (`.github/workflows/cd.yml`):**
    *   Triggered on successful completion of the CI workflow on the `main` branch.
    *   Checks out the specific commit that passed CI.
    *   Sets up Python, Poetry, Node.js (for CDK), and AWS credentials.
    *   Installs main and infrastructure dependencies (`poetry install --only main,infra`).
    *   Deploys the infrastructure using AWS CDK (`poetry run cdk deploy`). Requires AWS secrets configured in GitHub repository settings.

## Building and Packaging

**1. Using Docker (Recommended for Web Services/Deployment)**

*   **Create/Update a `Dockerfile`:** Ensure your Dockerfile uses Poetry for installing dependencies. Example snippet:

    ```dockerfile
    # Use an official Python runtime as a parent image
    FROM python:3.10-slim

    WORKDIR /app

    ENV PYTHONDONTWRITEBYTECODE 1
    ENV PYTHONUNBUFFERED 1

    # Install Poetry
    RUN pip install --upgrade pip poetry

    # Copy only files necessary for dependency installation
    COPY pyproject.toml poetry.lock ./

    # Install dependencies --no-root installs only dependencies, not the project itself
    # --no-interaction prevents prompts, --no-ansi prevents color codes in logs
    # Use --without dev to exclude development dependencies in production image
    RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi --without dev

    # Copy the application code into the container
    COPY ./app /app/app
    # Optionally copy data/index if not mounting volumes
    # COPY ./data /app/data
    # COPY ./faiss_index /app/faiss_index

    EXPOSE 8003 # Match the port used by uvicorn

    ENV ENVIRONMENT=production
    ENV PORT=8003
    # OPENAI_API_KEY should be passed securely at runtime

    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
    ```

*   **Build the Docker Image:**
    ```bash
    docker build -t tweet-rag-api .
    ```

*   **Run the Docker Container:**
    ```bash
    docker run -d --name my-rag-app \
      -p 8003:8003 \ # Match exposed port
      -e OPENAI_API_KEY="your_openai_api_key_here" \
      -e API_KEY="your_secure_api_key_for_production" \
      -e ENVIRONMENT="production" \
      # Mount volumes for persistent index/data if needed
      --mount type=bind,source="$(pwd)"/faiss_index,target=/app/faiss_index \
      --mount type=bind,source="$(pwd)"/data,target=/app/data \
      tweet-rag-api
    ```

**2. Using Standard Python Packaging (Wheels/Sdist via Poetry)**

Poetry handles building standard Python packages.

*   **Run the build command:**
    ```bash
    poetry build
    ```
    This will create wheel (`.whl`) and source distribution (`.tar.gz`) files in the `dist/` directory based on the configuration in `pyproject.toml`. These can be installed using `pip`, but running the web service still requires a separate process like `uvicorn`.

## Deployment Notes

*   For production, use a production-grade ASGI server like Uvicorn managed by a process manager (e.g., Gunicorn with Uvicorn workers, Supervisor, systemd). The Docker approach handles this.
*   Set the `ENVIRONMENT` environment variable to `production`.
*   Ensure proper API key management (pass via environment variables or secrets management).
*   Configure CORS (`allow_origins` in `app/config.py`) restrictively.
*   Consider a reverse proxy (Nginx, Traefik) for SSL, load balancing, etc.
*   The included `cd.yml` workflow provides an example of deploying via AWS CDK.

## Project Status

This project is actively maintained. Current development focuses on:
- Expanding the knowledge base capabilities
- Adding support for additional query types
- Improving document ranking algorithms
- Enhancing alternative viewpoint generation

## Contributing

Contributions are welcome! To contribute to this project:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests to ensure they pass
5. Submit a pull request with a detailed description of changes

For more details on the codebase organization, see the [Application Structure Documentation](/app/README.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
