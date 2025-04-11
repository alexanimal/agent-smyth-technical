"""
Middleware package for the RAG Agent application.

This package re-exports middleware functions from the request_logging module
to maintain backward compatibility.
"""

from app.middleware.request_logging import log_requests

# For backward compatibility
__all__ = ["log_requests"]
