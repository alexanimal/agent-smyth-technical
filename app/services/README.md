# Services Module

This module provides service initialization, dependency injection, and application state management for the RAG Agent application.

## Overview

The services module handles:
- Knowledge base initialization and loading
- Chat handler management and dependency injection
- Application state tracking
- Service availability checking for API endpoints

This module is critical for the application's startup process and for ensuring resources are properly initialized before handling requests.

## Components

### Initialization (`initialization.py`)

Manages the core application services:

- `initialize_services()`: Asynchronous function that loads the knowledge base and initializes the chat handler
- `get_current_chat_handler()`: FastAPI dependency that provides the chat handler to route handlers
- `get_app_state()`: FastAPI dependency that provides access to the current application state

### Application State

The module maintains a global application state dictionary with the following keys:
- `knowledge_base`: The loaded knowledge base instance
- `chat_handler`: The initialized chat handler instance
- `is_kb_loading`: Flag indicating whether the knowledge base is currently loading

## Usage

### Service Initialization

The services should be initialized during application startup:

```python
from fastapi import FastAPI
from app.services import initialize_services

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await initialize_services()
```

### Dependency Injection

The module provides dependencies that can be used in FastAPI route handlers:

```python
from fastapi import APIRouter, Depends
from app.services import get_current_chat_handler, get_app_state
from app.core.handler import ChatHandler

router = APIRouter()

@router.post("/chat")
async def handle_chat(
    request_body: dict,
    chat_service: ChatHandler = Depends(get_current_chat_handler)
):
    # The chat handler is guaranteed to be initialized here
    # or a 503 Service Unavailable exception is raised
    result = await chat_service.process_query(
        message=request_body["message"]
    )
    return result

@router.get("/health")
async def health_check(app_state: dict = Depends(get_app_state)):
    return {
        "status": "healthy",
        "knowledge_base_loaded": app_state["chat_handler"] is not None,
        "is_loading": app_state["is_kb_loading"],
    }
```

## Error Handling

The service dependencies implement robust error handling:

- `get_current_chat_handler()` raises a `503 Service Unavailable` if the chat handler isn't initialized
- Different error messages are provided based on whether the knowledge base is still loading or failed to initialize
- Initialization failures are properly logged and the application state is updated accordingly

## Extending the Module

To add new services:

1. Define new service classes in separate files
2. Add initialization logic to `initialize_services()`
3. Add the service instances to the `app_state` dictionary
4. Create new dependency functions to provide access to the services

For example, to add a metrics service:

```python
# In a new file: app/services/metrics.py
class MetricsService:
    def __init__(self):
        self.requests_count = 0

    def increment_requests(self):
        self.requests_count += 1

# In initialization.py
app_state["metrics_service"] = None

async def initialize_services():
    # Existing initialization code...
    app_state["metrics_service"] = MetricsService()

def get_metrics_service():
    if not app_state["metrics_service"]:
        raise HTTPException(503, "Metrics service not initialized")
    return app_state["metrics_service"]
```
