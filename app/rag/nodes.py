"""
Node functions for the RAG workflow graph.

This module defines the node functions used in the LangGraph workflow for
Retrieval Augmented Generation (RAG). Each function represents a step in the
RAG pipeline, from query classification to response generation.
"""

import json
import logging
from operator import itemgetter
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.prompts import PromptManager
from app.rag.scoring import diversify_ranked_documents, score_documents_with_social_metrics
from app.rag.state import RAGState

# Configure logging
logger = logging.getLogger(__name__)


async def classify_query_node(state: RAGState) -> RAGState:
    """
    Node to classify the query type.

    This node analyzes the input query and determines its type (technical,
    investment, trading thesis, or general) using an LLM-based classifier.

    Args:
        state: The current workflow state

    Returns:
        Updated workflow state with classification information
    """
    query = state["query"]

    # Create classifier model - could be moved to a singleton/factory
    classifier_model = ChatOpenAI(model="gpt-4o", temperature=0.0)

    # Get classification prompt
    classification_prompt = PromptManager.get_classification_prompt()

    # Create classification chain
    chain = classification_prompt | classifier_model | StrOutputParser()

    try:
        # Get classification result
        classification_result = await chain.ainvoke({"query": query})

        # Parse JSON result with confidence scores
        confidence_scores = json.loads(classification_result.strip())

        # Validate expected keys
        expected_keys = ["technical", "trading_thesis", "investment", "general"]
        if not all(key in confidence_scores for key in expected_keys):
            logger.warning(
                f"Invalid classification result: {classification_result}, defaulting to general"
            )
            confidence_scores = {
                "technical": 0,
                "trading_thesis": 0,
                "investment": 0,
                "general": 100,
            }

        # Determine highest confidence category
        max_category = max(confidence_scores.items(), key=itemgetter(1))[0]

        # Determine if mixed query
        scores = sorted(confidence_scores.values(), reverse=True)
        is_mixed = len(scores) > 1 and scores[1] > 30

        # Update state with classification info
        state["classification"] = {
            "query_type": max_category,
            "confidence_scores": confidence_scores,
            "is_mixed": is_mixed,
        }

    except Exception as e:
        logger.error(f"Error in classification node: {str(e)}", exc_info=True)
        # Default to general with low confidence
        state["classification"] = {
            "query_type": "general",
            "confidence_scores": {
                "technical": 0,
                "trading_thesis": 0,
                "investment": 0,
                "general": 100,
            },
            "is_mixed": False,
        }

    return state


async def retrieve_documents_node(state: RAGState) -> RAGState:
    """
    Node to retrieve documents based on the query.

    This node retrieves relevant documents from the knowledge base based on
    the user's query, adjusting retrieval parameters based on the query type.

    Args:
        state: The current workflow state

    Returns:
        Updated workflow state with retrieved documents
    """
    query = state["query"]
    classification = state["classification"]

    # Use user-provided num_results with reasonable constraints
    user_k = state["num_results"]

    # Apply query-type-based adjustments but respect user preferences
    base_k = user_k
    if (
        classification["query_type"] in ["trading_thesis", "technical"]
        or classification["is_mixed"]
    ):
        # For complex queries, increase the base k by 50% if user didn't specify a large number
        if user_k < 10:
            base_k = min(int(user_k * 1.5), user_k + 5)  # Increase but cap the increase

    # Oversample for better diversity
    oversample_factor = 3
    initial_k = base_k * oversample_factor

    # Create a knowledge base instance (typically this would be passed in or accessed via dependency)
    # This is a placeholder - you would replace with your actual KB implementation
    from app.kb import KnowledgeBaseManager

    kb_manager = KnowledgeBaseManager()
    kb = await kb_manager.load_or_create_kb()

    # Retrieve documents
    retriever = kb.as_retriever(search_kwargs={"k": initial_k})
    initial_docs = await retriever.ainvoke(query)

    # Update state with retrieved documents
    state["retrieved_docs"] = initial_docs

    return state


async def rank_documents_node(state: RAGState) -> RAGState:
    """
    Node to rank and diversify documents.

    This node re-ranks the retrieved documents based on recency, relevance,
    and social engagement metrics, optimizing for diversity of information and viewpoints.

    Args:
        state: The current workflow state

    Returns:
        Updated workflow state with ranked documents
    """
    docs = state["retrieved_docs"]
    query_type = state["classification"]["query_type"]
    is_mixed = state["classification"]["is_mixed"]
    user_k = state["num_results"]

    # Determine final k based on user request and query type
    final_k = user_k
    if query_type in ["trading_thesis", "technical"] or is_mixed:
        # For complex queries, we might want to provide slightly more results
        # But still respect user's request as a general guideline
        if user_k < 10:
            final_k = min(int(user_k * 1.5), user_k + 5)

    # Use ranking_config from state if provided, otherwise use defaults based on query type
    ranking_config = state.get("ranking_config", {})

    # Set default weights if not provided
    if not ranking_config:
        # Adjust weights based on query type
        if query_type == "investment":
            # For investment queries, prioritize engagement metrics more
            ranking_config = {
                "recency_weight": 0.3,
                "view_weight": 0.2,
                "like_weight": 0.2,
                "retweet_weight": 0.3,
            }
        elif query_type == "technical":
            # For technical analysis, recency matters more
            ranking_config = {
                "recency_weight": 0.5,
                "view_weight": 0.1,
                "like_weight": 0.2,
                "retweet_weight": 0.2,
            }
        else:
            # Default balanced weights
            ranking_config = {
                "recency_weight": 0.4,
                "view_weight": 0.2,
                "like_weight": 0.2,
                "retweet_weight": 0.2,
            }

    # Validate docs have required metadata
    valid_docs = []
    for doc in docs:
        # Check for timestamp which is required for ranking
        if "timestamp_unix" in doc.metadata:
            valid_docs.append(doc)
        else:
            logger.warning("Document missing timestamp_unix metadata. Skipping.")

    if not valid_docs:
        logger.warning("No valid documents with required metadata found for ranking.")
        state["ranked_docs"] = []
        return state

    try:
        # Score documents using social metrics
        scored_docs = score_documents_with_social_metrics(valid_docs, ranking_config)

        # Apply diversity to the ranked documents
        diversity_factor = 0.3  # Could be tuned or passed in state
        re_ranked_docs = diversify_ranked_documents(scored_docs, final_k, diversity_factor)

        # Update state with ranked documents
        state["ranked_docs"] = re_ranked_docs

    except Exception as e:
        logger.error(f"Error in document ranking: {e}", exc_info=True)
        # Fallback to simple timestamp-based ranking
        valid_docs_with_ts = []
        for doc in valid_docs:
            timestamp = doc.metadata.get("timestamp_unix")
            try:
                valid_docs_with_ts.append((doc, float(timestamp)))
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp_unix: {timestamp}. Skipping.")

        # Sort by timestamp (recency)
        valid_docs_with_ts.sort(key=lambda x: x[1], reverse=True)
        re_ranked_docs = [doc for doc, _ in valid_docs_with_ts[:final_k]]
        state["ranked_docs"] = re_ranked_docs

    return state


async def generate_response_node(state: RAGState) -> RAGState:
    """
    Node to generate a response based on ranked documents.

    This node generates a response to the user's query based on the ranked
    documents, using the appropriate prompt template and model based on the
    query type.

    Args:
        state: The current workflow state

    Returns:
        Updated workflow state with generated response and sources
    """
    query = state["query"]
    docs = state["ranked_docs"]
    classification = state["classification"]

    if not docs:
        state["response"] = "I couldn't find any relevant information."
        state["sources"] = []
        return state

    # Extract document content and build context string
    context_string = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # Add technical data for certain query types
    query_type = classification["query_type"]
    if query_type in ["trading_thesis", "technical"]:
        # This would call the full _get_technical_indicators method
        # Simplified placeholder
        technical_data = "Technical Indicator Data:\n(Technical indicators would be extracted here)"
        context_string += f"\n\n{technical_data}"

    # Choose prompt template based on query type
    prompt_templates = {
        "investment": PromptManager.get_investment_prompt(),
        "trading_thesis": PromptManager.get_trading_thesis_prompt(),
        "technical": PromptManager.get_technical_analysis_prompt(),
        "general": PromptManager.get_general_prompt(),
    }
    prompt_template = prompt_templates.get(query_type, PromptManager.get_general_prompt())

    # Choose the appropriate model
    model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    if query_type in ["trading_thesis", "technical"]:
        model = ChatOpenAI(model="gpt-4o", temperature=0.0)  # Lower temp for technical

    # Build and run the chain
    inputs = RunnableParallel(
        context=RunnableLambda(lambda x: context_string),
        question=RunnablePassthrough(),
    )
    chain = inputs | prompt_template | model | StrOutputParser()

    # Generate response
    response = await chain.ainvoke(query)

    # Extract sources
    sources = []
    for doc in docs:
        if doc.metadata and "url" in doc.metadata and doc.metadata["url"]:
            sources.append(doc.metadata["url"])

    # Update state
    state["response"] = response
    state["sources"] = list(set(sources))  # Deduplicate sources

    return state


async def generate_alternative_node(state: RAGState) -> RAGState:
    """
    Node to generate alternative viewpoints for certain query types.

    This node generates alternative perspectives or counterarguments for
    specific query types to provide a more balanced response to the user.

    Args:
        state: The current workflow state

    Returns:
        Updated workflow state with alternative viewpoints
    """
    query = state["query"]
    docs = state["ranked_docs"]
    classification = state["classification"]

    # Only process for certain query types that aren't mixed
    query_type = classification["query_type"]
    is_mixed = classification["is_mixed"]

    if query_type in ["investment", "trading_thesis", "technical"] and not is_mixed:
        try:
            # Build context string
            context_string = "\n\n---\n\n".join([doc.page_content for doc in docs])

            # Choose prompt template - fix the potential None issue
            prompt_template_getter = {
                "investment": PromptManager.get_investment_prompt,
                "trading_thesis": PromptManager.get_trading_thesis_prompt,
                "technical": PromptManager.get_technical_analysis_prompt,
            }.get(query_type)

            if prompt_template_getter:
                prompt_template = prompt_template_getter()

                # Use higher temperature model for alternatives
                explorer_model = ChatOpenAI(
                    model="gpt-4o", temperature=0.7
                )  # Higher temp for creativity

                # Build chain
                inputs = RunnableParallel(
                    context=RunnableLambda(lambda x: context_string),
                    question=RunnablePassthrough(),
                )
                counter_chain = inputs | prompt_template | explorer_model | StrOutputParser()

                # Generate alternative
                counter_prompt = (
                    f"Generate an alternative perspective or counter-argument to: {query}"
                )
                alternative_viewpoints = await counter_chain.ainvoke(counter_prompt)

                # Update state
                state["alternative_viewpoints"] = alternative_viewpoints
            else:
                logger.warning(f"No prompt template found for query type: {query_type}")

        except Exception as e:
            logger.warning(f"Failed to generate alternative viewpoints: {str(e)}")
            # No need to set alternative_viewpoints to None, it's either populated or not in the state

    return state
