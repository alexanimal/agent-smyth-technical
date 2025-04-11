# Routers Module

This module contains the FastAPI routers that define the API endpoints for the RAG Agent application. The routers handle request validation, route incoming requests to the appropriate business logic, and format responses.

## Overview

The module contains two main routers:
1. `chat_router.py`: Handles chat request processing and response generation
2. `status_router.py`: Provides system health and status information endpoints

## Chat Router (`chat_router.py`)

The Chat Router implements endpoints for processing user queries and generating responses using the RAG system.

### Endpoints

#### POST `/chat`

Processes a chat message and returns a complete response with sources.

**Request Body:**
- `message`: The user's query text (required)
- `num_results`: Number of sources to retrieve (default: 25, max: 250)
- `context`: Optional contextual information
- `model`: Optional override for the LLM model
- `generate_alternative_viewpoint`: Whether to generate alternative perspectives
- `ranking_weights`: Optional configuration for document ranking

**Response:**
- `request_id`: Unique identifier for the request
- `response`: The generated text response
- `sources`: List of source URLs used in the generation
- `processing_time`: Time taken to process the request
- `alternative_viewpoints`: Optional alternative perspective
- `metadata`: Additional metadata about the response and processing

#### POST `/chat/stream`

Streams a chat response using Server-Sent Events (SSE) for real-time updates.

**Request Body:**
- Same as the `/chat` endpoint

**Stream Events:**
- `start`: Indicates the start of processing
- `chunk`: Contains fragments of the response text
- `complete`: Contains the full response with sources and metadata
- `error`: Sent if an error occurs during processing

### Authentication

In production environments, both endpoints require a valid API key via the `X-API-Key` header.

## Status Router (`status_router.py`)

The Status Router provides endpoints for monitoring the system health and status.

### Endpoints

#### GET `/`

Root endpoint confirming API status.

**Response:**
- `message`: Informational message
- `status`: System status ("ready", "knowledge_base_loading", or "error")

#### GET `/health`

Health check endpoint for monitoring.

**Response:**
- `status`: "healthy" or error status
- `knowledge_base_loaded`: Whether the knowledge base is loaded
- `is_loading`: Whether the system is currently initializing components

## Usage

Import and include the routers in your FastAPI application:

```python
from fastapi import FastAPI
from app.routers.chat_router import router as chat_router
from app.routers.status_router import router as status_router

app = FastAPI()
app.include_router(chat_router)
app.include_router(status_router)
```

## Example Request

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What are investors saying about AAPL?",
        "num_results": 25,
        "generate_alternative_viewpoint": True,
        "model": "gpt-4-turbo"
    },
    headers={
        "X-API-Key": "your_api_key"
    }
)

print(response.json())
```

## Error Handling

The routers implement comprehensive error handling with appropriate HTTP status codes:
- 400/422: Validation errors for malformed requests
- 401: Authentication errors for invalid API keys
- 500: Internal server errors with descriptive messages
- 503: Service unavailable if the knowledge base is still loading
