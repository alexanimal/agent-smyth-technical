"""
Exception handling middleware for the RAG Agent application.

This module provides middleware for handling application-specific exceptions
and converting them to appropriate HTTP responses with detailed error messages.
"""

import logging
from typing import Any, Callable, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_422_UNPROCESSABLE_ENTITY

from app.config import settings

logger = logging.getLogger(__name__)


class ModelValidationError(ValueError):
    """Exception raised when an invalid model is specified."""

    pass


async def model_validation_exception_handler(request: Request, exc: ModelValidationError):
    """
    Handle invalid model exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse: A well-formatted error response
    """
    logger.warning(f"Invalid model requested: {str(exc)}")
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "type": "model_validation_error",
            "allowed_models": settings.allowed_models,
            "default_model": settings.default_model,
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors.

    Args:
        request: The request that caused the exception
        exc: The validation exception

    Returns:
        JSONResponse: A well-formatted validation error response
    """
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "type": "validation_error",
            "body": exc.body if hasattr(exc, "body") else None,
        },
    )


def add_exception_handlers(app: FastAPI):
    """
    Register all custom exception handlers with the FastAPI application.

    Args:
        app: The FastAPI application instance
    """
    app.add_exception_handler(ModelValidationError, model_validation_exception_handler)  # type: ignore
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore
