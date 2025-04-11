"""
Services package for the RAG Agent application.

This package re-exports service functions from the initialization module
to maintain backward compatibility.
"""

from app.services.initialization import get_app_state, get_current_chat_handler, initialize_services

# For backward compatibility
__all__ = ["initialize_services", "get_current_chat_handler", "get_app_state"]
