"""
Knowledge base package for the RAG Agent application.

This package re-exports the KnowledgeBaseManager class and other related
functions from the manager module to maintain backward compatibility.
"""

from app.kb.manager import (
    KnowledgeBaseManager,
    extract_content,
    get_knowledge_base,
    load_documents,
    metadata_extractor,
    process_file,
)

# For backward compatibility
__all__ = [
    "KnowledgeBaseManager",
    "extract_content",
    "metadata_extractor",
    "process_file",
    "load_documents",
    "get_knowledge_base",
]
