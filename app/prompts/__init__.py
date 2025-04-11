"""
Prompts package for the RAG Agent application.

This package re-exports the PromptManager class from the templates module
to maintain backward compatibility.
"""

from app.prompts.templates import PromptManager

# For backward compatibility
__all__ = ["PromptManager"]
