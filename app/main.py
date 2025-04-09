"""
FastAPI application entry point for the RAG Agent.
Assembles routers, middleware, configuration, and lifespan events.
"""
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

# Use relative imports for modules within src
from .config import setup_logging_and_sentry, settings
from .services import initialize_services
from .routers import chat_router, status_router
from .middleware import log_requests

# Setup Logging and Sentry based on config
# Do this before other imports that might log
setup_logging_and_sentry(settings)

# Define lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
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
    version="1.0.0", # Consider getting version from pyproject.toml later
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[ # Tags defined here or in routers
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
    allow_origins=settings.cors_origins if hasattr(settings, 'cors_origins') else ["http://localhost:3000", "http://localhost:5173"], # Example: Make origins configurable
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID", "User-Agent", "Sentry-Trace", "Baggage"], # Add Sentry headers if using performance
    expose_headers=["X-Request-ID", "X-Processing-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
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
    # Run referring to the app instance in *this* file
    # Use the correct path relative to project root if running from there
    # Example: python -m src.main
    # Or use uvicorn src.main:app from the project root
    uvicorn.run("src.app:app", host="localhost", port=8003, reload=True)
