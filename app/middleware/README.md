# Middleware Package

This package contains FastAPI middleware components for cross-cutting concerns in the RAG Agent application.

## Contents

- `request_logging.py`: Implements middleware for logging incoming requests, adding timing information, and request ID tracking.
- `__init__.py`: Re-exports middleware functions for backward compatibility.

## Usage

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from app.middleware import log_requests

app = FastAPI()
app.add_middleware(BaseHTTPMiddleware, dispatch=log_requests)
```

## Key Features

- Request logging with unique request IDs
- Timing measurements for performance monitoring
- HTTP header management (X-Request-ID, X-Processing-Time)
- Exception handling during request processing
