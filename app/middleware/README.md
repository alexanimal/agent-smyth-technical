# Middleware Module

This module contains FastAPI middleware components that process HTTP requests and responses for the RAG Agent application.

## Overview

Middleware functions as a layer between the client and the application's endpoint handlers, enabling:
- Request preprocessing
- Response postprocessing
- Cross-cutting concerns implementation (logging, authentication, error handling)
- Performance monitoring

## Components

### RequestLoggingMiddleware

Logs details about incoming requests and their processing time:

```python
from app.middleware.logging import RequestLoggingMiddleware

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)
```

Key features:
- Logs request method, path, and client IP
- Records processing time
- Includes status code in the log output
- Configurable log level and format

### ResponseHeaderMiddleware

Adds custom headers to all responses:

```python
from app.middleware.headers import ResponseHeaderMiddleware

app = FastAPI()
app.add_middleware(
    ResponseHeaderMiddleware,
    headers={"X-API-Version": "1.0.0"}
)
```

Key features:
- Adds consistent headers to all responses
- Supports dynamic header values via callback functions
- Can be configured to exclude specific endpoints

### APIKeyMiddleware

Validates API keys for protected endpoints:

```python
from app.middleware.auth import APIKeyMiddleware

app = FastAPI()
app.add_middleware(
    APIKeyMiddleware,
    api_key_name="X-API-Key",
    excluded_paths=["/health", "/docs"]
)
```

Key features:
- Enforces API key validation
- Configurable header name for the API key
- Support for excluding specific paths
- Different validation rules based on environment

### CORSMiddleware

Configures Cross-Origin Resource Sharing for the application:

```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Key features:
- Controls which domains can access the API
- Configurable allowed methods and headers
- Support for credentials

### ErrorHandlingMiddleware

Provides consistent error responses across the application:

```python
from app.middleware.errors import ErrorHandlingMiddleware

app = FastAPI()
app.add_middleware(ErrorHandlingMiddleware)
```

Key features:
- Catches unhandled exceptions
- Formats error responses consistently
- Prevents sensitive error details in production
- Logs errors with appropriate severity levels

## Middleware Order

The order in which middleware is applied is important, as it determines the processing sequence:

1. **ErrorHandlingMiddleware** - Outermost to catch all errors
2. **RequestLoggingMiddleware** - To log the request before other processing
3. **APIKeyMiddleware** - To authenticate before processing
4. **CORSMiddleware** - To handle CORS headers
5. **ResponseHeaderMiddleware** - Innermost to add headers to the final response

## Configuration

Middleware components can be configured via environment variables or directly in the application startup:

```python
# In app/main.py
from app.config import settings
from app.middleware import (
    APIKeyMiddleware,
    RequestLoggingMiddleware,
    ResponseHeaderMiddleware,
)

app = FastAPI()

# Add middleware in the appropriate order
app.add_middleware(
    APIKeyMiddleware,
    api_key=settings.api_key,
    excluded_paths=["/health", "/docs", "/redoc"]
)
app.add_middleware(
    RequestLoggingMiddleware,
    log_level=settings.log_level
)
app.add_middleware(
    ResponseHeaderMiddleware,
    headers={
        "X-API-Version": settings.version,
        "X-Environment": settings.environment
    }
)
```

## Creating Custom Middleware

To create a new middleware component:

1. Create a new class inheriting from `BaseHTTPMiddleware`
2. Implement the `dispatch` method to process requests and responses
3. Add the middleware to the application

Example:

```python
from fastapi.middleware.base import BaseHTTPMiddleware

class CustomMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, some_parameter: str):
        super().__init__(app)
        self.some_parameter = some_parameter

    async def dispatch(self, request, call_next):
        # Process the request (before endpoint)
        # ...

        # Call the next middleware or endpoint
        response = await call_next(request)

        # Process the response (after endpoint)
        # ...

        return response
```

## Testing Middleware

To test middleware components:

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_key_middleware():
    response = client.get(
        "/protected-endpoint",
        headers={"X-API-Key": "invalid-key"}
    )
    assert response.status_code == 401

    response = client.get(
        "/protected-endpoint",
        headers={"X-API-Key": "valid-key"}
    )
    assert response.status_code == 200
```
