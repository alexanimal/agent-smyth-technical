"""
RAG state definition for LangGraph workflow.

This module defines the state schema used by the LangGraph workflow for Retrieval
Augmented Generation (RAG). It specifies the structure of the state object that
gets passed between nodes in the graph.
"""

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    """
    State schema for the RAG workflow.

    This typed dictionary defines the structure of the state that flows through
    the RAG workflow graph, including query information, retrieved documents,
    and generated responses.

    Attributes:
        query: The user's original query string
        classification: Dictionary containing query classification results
        retrieved_docs: List of initially retrieved documents
        ranked_docs: List of documents after ranking and diversification
        response: The generated response text
        sources: List of source URLs used to generate the response
        alternative_viewpoints: Optional alternative perspective or counterargument
    """

    query: str
    classification: Dict[str, Any]
    retrieved_docs: List[Document]
    ranked_docs: List[Document]
    response: Optional[str]
    sources: List[str]
    alternative_viewpoints: Optional[str]
