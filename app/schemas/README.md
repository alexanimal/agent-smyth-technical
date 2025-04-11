# Schemas Module

This module defines the Pydantic data models used throughout the RAG Agent application for input validation, data serialization, and API documentation.

## Overview

The schemas module provides type-safe data structures with validation logic for all data that flows through the application, including:

- API request and response models
- Enumerated types for categorization
- Configuration models
- Shared type definitions

## Components

### API Schemas (`api.py`)

Defines the core request and response models for the API:

#### Request Models
- `ChatRequest`: Validates and structures incoming chat requests
  - Enforces constraints on message length and content
  - Provides configuration options for retrieval and ranking
  - Validates ranking weights

#### Response Models
- `ChatResponse`: Structures the data returned from chat requests
  - Includes the generated response and sources
  - Provides metadata about processing
  - Contains alternative viewpoints when applicable

#### Status Models
- `HealthStatus`: Reports system health information
- `RootResponse`: Basic response for the root endpoint

#### Enumerations
- `QueryType`: Defines the supported types of queries (GENERAL, INVESTMENT, TRADING, TECHNICAL)

### Model Schemas (`models.py`)

Defines models related to LLM selection and configuration:

- `OpenAIModel`: Enumeration of supported OpenAI models with validation
- `ModelSelectionRequest`: Schema for model selection in API requests

## Usage

### Validating Incoming Requests

The schemas automatically validate incoming API requests:

```python
from fastapi import APIRouter
from app.schemas import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    # The request is already validated according to the ChatRequest model
    # Fields have appropriate types and meet validation constraints
    message = request.message
    num_results = request.num_results

    # Process the request...

    return {
        "request_id": "123",
        "response": "Generated response",
        "sources": ["https://source1.com", "https://source2.com"],
        "processing_time": 1.5,
        "alternative_viewpoints": "Alternative perspective...",
        "metadata": {"model_used": "gpt-4-turbo"}
    }
```

### Creating Models in Code

```python
from app.schemas import ChatRequest, OpenAIModel

# Create a new chat request with validated data
request = ChatRequest(
    message="What are investors saying about AAPL?",
    num_results=25,
    ranking_weights={
        "recency_weight": 0.3,
        "view_weight": 0.2,
        "like_weight": 0.2,
        "retweet_weight": 0.3
    },
    model=OpenAIModel.GPT_4_TURBO
)

# Access validated fields
print(request.message)
print(request.model.value)  # "gpt-4-turbo"
```

## Schema Documentation

The schemas include detailed field descriptions that are automatically used to generate API documentation in tools like Swagger UI and ReDoc.

## Extending the Schemas

To add new schemas or extend existing ones:

1. Define new models in the appropriate file (`api.py` or `models.py`)
2. Export them in `__init__.py` for easy importing
3. Add appropriate validation logic using Pydantic validators

For example, to extend `ChatRequest` with a new field:

```python
from pydantic import Field
from app.schemas import ChatRequest

class ExtendedChatRequest(ChatRequest):
    language: str = Field("en", description="Language for the response")
```
