"""
LangGraph workflow definition for the RAG system.

This module defines and configures the LangGraph workflow for the Retrieval
Augmented Generation (RAG) system, creating a directed graph of processing
nodes with conditional routing based on query types.
"""

from langgraph.graph import END, StateGraph

from app.rag.nodes import (
    classify_query_node,
    generate_alternative_node,
    generate_response_node,
    rank_documents_node,
    retrieve_documents_node,
)
from app.rag.state import RAGState


def create_rag_workflow() -> StateGraph:
    """
    Create and configure the RAG workflow graph.

    This function assembles the LangGraph workflow by defining nodes,
    edges, and conditional routing logic based on query classification.

    Returns:
        StateGraph: The configured workflow graph
    """
    # Create state graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("rank_documents", rank_documents_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("generate_alternative", generate_alternative_node)

    # Add edges with conditional routing
    workflow.add_edge("classify_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "rank_documents")
    workflow.add_edge("rank_documents", "generate_response")

    # Conditional edge: only generate alternatives for certain query types
    workflow.add_conditional_edges(
        "generate_response",
        lambda state: (
            "generate_alternative"
            if state["classification"]["query_type"]
            in ["investment", "trading_thesis", "technical"]
            and not state["classification"]["is_mixed"]
            else END
        ),
    )
    workflow.add_edge("generate_alternative", END)

    # Set entry point
    workflow.set_entry_point("classify_query")

    return workflow


# Compile the graph
app_workflow = create_rag_workflow().compile()
