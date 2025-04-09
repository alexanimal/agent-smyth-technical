"""
FastAPI application entry point for the RAG Agent.

This module serves as the entry point for the FastAPI application that powers the RAG Agent.
It handles the assembly of routers, middleware, configuration, and lifespan events.

The application exposes a REST API for querying the knowledge base with natural language
questions and receiving detailed responses with source attribution.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

# Use relative imports for modules within src
from .config import settings, setup_logging_and_sentry
from .middleware import log_requests
from .routers import chat_router, status_router
from .services import initialize_services

# Setup Logging and Sentry based on config
# Do this before other imports that might log
setup_logging_and_sentry(settings)


# Define lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle events.

    This async context manager ensures proper initialization of services during startup
    and cleanup during shutdown. It's responsible for initializing the knowledge base
    and other services asynchronously.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is yielded back to the application while it's running.
    """
    # Startup: Trigger background initialization of services (KB, etc.)
    startup_task = asyncio.create_task(initialize_services())
    yield
    # Shutdown: Add cleanup if needed (e.g., wait for tasks, close connections)
    # await startup_task # Optionally wait if needed, depends on desired shutdown behavior


# Create FastAPI app instance
app = FastAPI(
    title="Tweet RAG Agent API",
    description="""
    An API that uses Retrieval Augmented Generation (RAG) to answer user queries based on tweet data.

    ## Features
    * Real-time chat processing using LLM
    * Source attribution from Twitter data
    * Query classification for specialized responses

    ## Authentication
    Production environments require API key authentication via the `X-API-Key` header.
    """,
    version="1.0.0",  # Consider getting version from pyproject.toml later
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[  # Tags defined here or in routers
        {"name": "Chat", "description": "Endpoints for question answering."},
        {"name": "Health", "description": "Health check and status endpoints."},
    ],
    lifespan=lifespan,
)

# Add Middleware
# Note: Order matters. CORS should generally be early.
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Request-ID",
        "User-Agent",
        "Sentry-Trace",
        "Baggage",
    ],  # Add Sentry headers if using performance
    expose_headers=[
        "X-Request-ID",
        "X-Processing-Time",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
    ],
)

# Add custom logging middleware
# Use add_middleware for cleaner separation than decorator
app.add_middleware(BaseHTTPMiddleware, dispatch=log_requests)

# Include Routers
app.include_router(status_router.router)
app.include_router(chat_router.router)

# Root endpoint is now in status_router, included above.

# Development server execution (if running this file directly)
if __name__ == "__main__":
    import uvicorn

    """
    Development server configuration.

    This code block only executes when the module is run directly.
    It configures and launches the Uvicorn ASGI server with development
    settings (hot reload enabled).
    """
    # Run referring to the app instance in *this* file
    # Use the correct path relative to project root if running from there
    # Example: python -m src.main
    # Or use uvicorn src.main:app from the project root
    uvicorn.run("src.app:app", host="localhost", port=8003, reload=True)
