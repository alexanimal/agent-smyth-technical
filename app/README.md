# RAG Agent Application

This application implements a Retrieval Augmented Generation (RAG) system for answering queries with up-to-date information from a knowledge base. It provides a FastAPI web service that processes user queries, retrieves relevant information, and generates comprehensive responses with source attribution.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Modules](#modules)
  - [RAG Module](#rag-appraq)
  - [Routers Module](#routers-approuters)
  - [Schemas Module](#schemas-appschemas)
  - [Utils Module](#utils-apputils)
  - [Services Module](#services-appservices)
  - [Core Module](#core-appcore)
  - [Config Module](#config-appconfig)
  - [Prompts Module](#prompts-appprompts)
  - [Middleware](#middleware-appmiddleware)
  - [Knowledge Base](#kb-appkb)
- [Data Flow](#data-flow)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Extending the Application](#extending-the-application)
- [Project Status](#project-status)
- [Contributing](#contributing)

## Architecture Overview

The application consists of several modules:

1. **RAG Module**: Core retrieval and generation logic using LangGraph
2. **Routers**: FastAPI endpoints for handling HTTP requests
3. **Schemas**: Pydantic models for request/response validation
4. **Utils**: Utility functions for document processing and analysis
5. **Services**: Business logic services and dependency injection
6. **Config**: Application configuration management
7. **Core**: Core components for handling requests

## Modules

### RAG (`/app/rag`)

The core RAG workflow implementation using LangGraph. It processes queries through a directed graph of nodes:
- Query classification
- Document retrieval
- Document ranking
- Response generation
- Alternative viewpoint generation

See the [RAG module README](/app/rag/README.md) for details.

### Routers (`/app/routers`)

FastAPI router definitions that handle HTTP requests:
- `chat_router.py`: Chat endpoints for processing queries
- `status_router.py`: Health and status endpoints

See the [Routers README](/app/routers/README.md) for details.

### Schemas (`/app/schemas`)

Pydantic models for request and response validation:
- API request/response models
- Enumerated types
- Status models
- Configuration schemas

See the [Schemas README](/app/schemas/README.md) for details.

### Utils (`/app/utils`)

Utility functions for document processing, technical analysis, and more:
- Document extraction and ranking
- Technical indicator identification
- Sentiment analysis for diversity

See the [Utils README](/app/utils/README.md) for details.

### Services (`/app/services`)

Service initialization, dependency injection, and application state management:
- Knowledge base initialization and loading
- Chat handler management
- Application state tracking
- Service availability checking for API endpoints

See the [Services README](/app/services/README.md) for details.

### Core (`/app/core`)

Core components for the application:
- Chat handler implementation
- Business logic processing
- Request handling middleware

### Config (`/app/config`)

Configuration management:
- Environment-specific settings
- Constants and defaults
- Configuration loading and validation

### Prompts Module (`/app/prompts`)

The Prompts module manages all prompting templates used throughout the RAG system:

- `PromptManager`: Central class for managing and accessing prompt templates
- Template creation for different query types (technical, investment, etc.)
- Dynamic prompt construction based on context and query classification
- Chain-of-Thought prompting techniques for complex reasoning

The system uses several advanced prompting techniques:
- Self-Evaluation (SE) for high-confidence responses
- Chain-of-Verification (CoVe) to reduce hallucinations
- Tree-of-Thought (ToT) for complex financial analysis queries

### Middleware (`/app/middleware`)

FastAPI middleware implementations for request processing:

- `RequestLoggingMiddleware`: Logs all incoming requests with timing information
- `ResponseHeaderMiddleware`: Adds custom headers to all responses
- `APIKeyMiddleware`: Validates API keys for protected endpoints
- `CORSMiddleware`: Configures Cross-Origin Resource Sharing
- `ErrorHandlingMiddleware`: Provides consistent error responses

### Knowledge Base (`/app/kb`)

The Knowledge Base module manages document storage and retrieval:

- `KnowledgeBaseManager`: Core class for KB operations
- Tweet data ingestion and processing pipeline
- Document chunking and embedding generation
- FAISS vector store management for similarity search
- Persistence layer for saving/loading the KB
- Background initialization with progress tracking
- Integrity verification of loaded KB instances

The KB provides methods for:
- `initialize()`: Create or load the knowledge base
- `search(query, k)`: Retrieve relevant documents
- `add_documents(documents)`: Extend the KB with new documents
- `health_check()`: Verify KB integrity
- `get_stats()`: Return statistics about the knowledge base

The system loads tweet data from the `data/` directory, processes it using parallel workers, and creates a FAISS index for efficient retrieval. The index is persisted to disk in the `faiss_index/` directory for faster startup on subsequent runs.

## Data Flow

1. **User query** is received via the Chat API endpoint
2. **RAG workflow** processes the query:
   - Query is classified (technical, investment, etc.)
   - Relevant documents are retrieved from the knowledge base
   - Documents are ranked and diversified
   - A response is generated using the ranked documents
   - Alternative viewpoints are generated if applicable
3. **Response** with source attribution is returned to the user

## Configuration

The application uses environment variables for configuration:
- `API_KEY`: API key for authentication (required in production)
- `ENVIRONMENT`: "development", "staging", or "production"
- `DEFAULT_MODEL`: Default OpenAI model to use for generation
- `ALLOWED_MODELS`: Comma-separated list of allowed models

## Running the Application

To run the application locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload
```

Access the API documentation at `http://localhost:8000/docs`.

## Testing

Run unit tests:

```bash
pytest
```

Run integration tests:

```bash
pytest tests/integration
```

## Extending the Application

To add new features:
1. Add new nodes to the RAG workflow in `app/rag/nodes.py`
2. Update the graph configuration in `app/rag/graph.py` if needed
3. Add new API endpoints in `app/routers/`
4. Add new schema models in `app/schemas/`
5. Add new utility functions in `app/utils/`
6. Add new services in `app/services/`

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

For more details, see the [Contributing Guide](CONTRIBUTING.md).
