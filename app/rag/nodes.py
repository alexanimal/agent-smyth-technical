"""
Node functions for the RAG workflow graph.

This module defines the node functions used in the LangGraph workflow for
Retrieval Augmented Generation (RAG). Each function represents a step in the
RAG pipeline, from query classification to response generation.
"""

import asyncio
import json
import logging
import time
from operator import itemgetter
from typing import Any, Dict, List, cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from app.config import settings
from app.prompts import PromptManager
from app.rag.scoring import diversify_ranked_documents, score_documents_with_social_metrics
from app.rag.state import RAGState
from app.rag.utils import generate_with_fallback, get_cached_llm

# Configure logging
logger = logging.getLogger(__name__)


def ensure_string_content(message_or_string: Any) -> Any:
    """
    Ensures that message objects are converted to their string content.

    This utility function handles conversion of AIMessage and other LangChain message objects
    to strings, while leaving other types unchanged.

    Args:
        message_or_string: An input that might be a message object or something else

    Returns:
        The string content if input is a message object, otherwise the original input
    """
    # Check if it's a message object with content attribute
    if hasattr(message_or_string, "content"):
        logger.debug(f"Converting message object to string: {type(message_or_string)}")
        return message_or_string.content

    # Return dictionaries and lists as is
    if isinstance(message_or_string, (dict, list)):
        return message_or_string

    # Handle bytes by decoding to string
    if isinstance(message_or_string, bytes):
        logger.debug(f"Converting bytes to string")
        return message_or_string.decode("utf-8")

    # For everything else, ensure it's a string
    if message_or_string is not None and not isinstance(message_or_string, str):
        logger.debug(f"Converting {type(message_or_string)} to string")
        return str(message_or_string)

    return message_or_string


def ensure_generation_metrics(state: RAGState) -> Dict[str, Any]:
    """
    Ensure the generation_metrics field exists in the state and return it.

    Args:
        state: The RAG state dictionary

    Returns:
        The generation_metrics dictionary
    """
    if "generation_metrics" not in state:
        state["generation_metrics"] = {}

    # Use cast to tell mypy this is not None
    return cast(Dict[str, Any], state["generation_metrics"])


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

    # Get model from state or use default for classification
    # For classification, we default to a fast model like gpt-4o even if a different model
    # is selected for content generation
    model_name = state.get("model", settings.default_model)
    logger.info(f"Classifying query using model: {model_name}")

    start_time = time.time()

    # Create classification prompt
    classification_prompt = PromptManager.get_classification_prompt()

    try:
        # Get classification result using cached model
        classifier_model = get_cached_llm(model_name, temperature=0.0)
        chain = classification_prompt | classifier_model | StrOutputParser()

        # Get classification result
        classification_result = await chain.ainvoke({"query": query})

        classification_time = time.time() - start_time
        logger.info(f"Query classified in {classification_time:.2f}s using {model_name}")

        # Parse JSON result with confidence scores
        try:
            # More robust JSON parsing with extensive error handling
            classification_result_stripped = classification_result.strip()

            # Log the raw result for debugging
            logger.debug(f"Raw classification result: '{classification_result_stripped}'")

            # Check if empty before parsing
            if not classification_result_stripped:
                logger.warning("Empty classification result received, using defaults")
                raise ValueError("Empty classification result")

            confidence_scores = json.loads(classification_result_stripped)

            # Validate expected keys
            expected_keys = ["technical", "trading_thesis", "investment", "general"]
            if not all(key in confidence_scores for key in expected_keys):
                logger.warning(
                    f"Invalid classification result missing required keys: {classification_result_stripped}, defaulting to general"
                )
                confidence_scores = {
                    "technical": 0,
                    "trading_thesis": 0,
                    "investment": 0,
                    "general": 100,
                }
        except (json.JSONDecodeError, ValueError) as json_error:
            logger.warning(
                f"Classification parsing failed: {str(json_error)}. Using default classification."
            )
            # Default to general with low confidence when JSON parsing fails
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
            "classification_time": classification_time,
            "model_used": model_name,
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
            "error": str(e),
            "model_used": model_name,
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

    # If user requests 0 results, return empty list immediately
    if user_k <= 0:
        state["retrieved_docs"] = []
        return state

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

    Robust error handling ensures:
    - Documents with missing timestamp metadata are skipped
    - Invalid timestamp formats (None, non-numeric strings, invalid types) are properly handled
    - When social metric scoring fails, fallback to simple timestamp-based sorting
    - If no valid documents remain after filtering, an empty list is returned

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
                # Convert timestamp to float, handle both None and non-numeric strings
                if timestamp is not None:
                    timestamp_float = float(timestamp)
                    valid_docs_with_ts.append((doc, timestamp_float))
                else:
                    logger.warning(f"Null timestamp_unix found. Skipping document.")
            except (ValueError, TypeError) as conversion_error:
                # Handle conversion errors (e.g., non-numeric strings)
                logger.warning(
                    f"Invalid timestamp_unix value: {timestamp}. Error: {conversion_error}. Skipping document."
                )

        # Sort by timestamp (recency)
        if valid_docs_with_ts:
            valid_docs_with_ts.sort(key=lambda x: x[1], reverse=True)
            re_ranked_docs = [doc for doc, _ in valid_docs_with_ts[:final_k]]
            state["ranked_docs"] = re_ranked_docs
        else:
            logger.warning(
                "No documents with valid timestamps after filtering. Returning empty list."
            )
            state["ranked_docs"] = []

    return state


async def generate_response_node(state: RAGState) -> RAGState:
    """
    Node to generate a response based on the retrieved documents.

    This node uses the query, classification, and ranked documents to generate a
    response using the appropriate prompt template for the query type.

    Args:
        state: The current workflow state

    Returns:
        Updated workflow state with response and sources
    """
    # Initialize time variables at the beginning to avoid UnboundLocalError in exception handling
    start_time = time.time()
    generation_time = None

    query = state["query"]
    docs = state["ranked_docs"]
    classification = state["classification"]

    # Get model name from state
    model_name = state.get("model", settings.default_model)

    # Add generation metrics to state if not already present
    metrics = ensure_generation_metrics(state)

    try:
        logger.info(f"Generating response using model: {model_name}")
        # start_time variable was moved to the beginning of the function

        # Build context string from ranked docs
        # This will be passed to the prompt template
        context_string = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Get query type from classification
        query_type = classification.get("query_type", "general")

        # Choose prompt template based on query type
        prompt_templates = {
            "investment": PromptManager.get_investment_prompt(),
            "trading_thesis": PromptManager.get_trading_thesis_prompt(),
            "technical": PromptManager.get_technical_analysis_prompt(),
            "general": PromptManager.get_general_prompt(),
        }
        prompt_template = prompt_templates.get(query_type, PromptManager.get_general_prompt())

        # Build inputs for the prompt
        inputs = RunnableParallel(
            context=RunnableLambda(lambda x: context_string),
            question=RunnablePassthrough(),
        )

        # Use generate_with_fallback for robust generation with retries
        prompt_inputs = await inputs.ainvoke(query)
        prompt_messages = await prompt_template.ainvoke(prompt_inputs)

        # Adjust temperature based on query type
        temperature = 0.0
        if query_type == "general":
            temperature = 0.2  # Slightly higher for general queries

        # Use our utility for generation with fallback
        response = await generate_with_fallback(
            prompt=prompt_messages,
            model_name=model_name if model_name is not None else settings.default_model,
            fallback_model="gpt-3.5-turbo",  # Fallback to a simpler model if needed
            temperature=temperature,
        )

        # Ensure the response is properly converted to a string
        parsed_response = StrOutputParser().parse(response)

        # Double-check that parsed_response is a string (handle AIMessage objects)
        # This is critical for calculating length and other string operations
        parsed_response = ensure_string_content(parsed_response)

        generation_time = time.time() - start_time
        logger.info(f"Response generated in {generation_time:.2f}s using {model_name}")

        # Extract sources
        sources = []
        for doc in docs:
            if doc.metadata and "url" in doc.metadata and doc.metadata["url"]:
                sources.append(doc.metadata["url"])

        # Safely get the content length
        try:
            response_length = len(parsed_response)
        except (TypeError, AttributeError):
            logger.warning(f"Could not determine length of response, using fallback length of 0")
            response_length = 0

        # Record metrics for monitoring and tracking
        metrics["model"] = model_name
        metrics["generation_time"] = generation_time
        metrics["document_count"] = len(docs)
        metrics["context_length"] = len(context_string)
        metrics["response_length"] = response_length

        # Update state
        state["response"] = parsed_response
        state["sources"] = list(set(sources))  # Deduplicate sources

    except Exception as e:
        # Calculate generation_time if not already set
        if generation_time is None:
            generation_time = time.time() - start_time

        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        state["response"] = (
            "I'm sorry, I encountered an error while generating a response. Please try again."
        )
        state["sources"] = []
        metrics["model"] = model_name
        metrics["error"] = str(e)
        metrics["document_count"] = len(docs)
        metrics["generation_time"] = generation_time

        if isinstance(e, asyncio.TimeoutError):
            logger.warning(f"Timeout occurred while generating response: {str(e)}")

    return state


async def generate_alternative_node(state: RAGState) -> RAGState:
    """
    Node to generate alternative viewpoints for certain query types.

    This node generates alternative perspectives or counterarguments for
    specific query types to provide a more balanced response to the user.
    The generation only happens if explicitly requested via the generate_alternative_viewpoint flag.

    Args:
        state: The current workflow state

    Returns:
        Updated workflow state with alternative viewpoints
    """
    # Initialize time variables at the beginning to avoid UnboundLocalError in exception handling
    start_time = time.time()
    generation_time = None

    # Check if alternative viewpoints are requested
    generate_alternative = state.get("generate_alternative_viewpoint", False)

    if not generate_alternative:
        logger.info("Alternative viewpoint generation skipped (not requested)")
        state["alternative_viewpoints"] = None
        return state

    query = state["query"]
    docs = state["ranked_docs"]
    classification = state["classification"]

    # Get model name from state
    model_name = state.get("model", settings.default_model)

    # Only process for certain query types that aren't mixed
    query_type = classification["query_type"]
    is_mixed = classification["is_mixed"]

    logger.info(
        f"Alternative viewpoint requested for query type: {query_type}, is_mixed: {is_mixed}"
    )

    # Initialize to None - ensure it's always set
    state["alternative_viewpoints"] = None

    # Relaxed condition to generate for all query types, not just specific ones
    # This helps when classification isn't working as expected
    try:
        logger.info(f"Generating alternative viewpoint using model: {model_name}")
        # start_time was moved to the beginning of the function

        # Build context string
        context_string = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Choose prompt template based on query type
        prompt_template = None

        # Try to get type-specific template first
        prompt_templates = {
            "investment": PromptManager.get_investment_prompt,
            "trading_thesis": PromptManager.get_trading_thesis_prompt,
            "technical": PromptManager.get_technical_analysis_prompt,
            "general": PromptManager.get_general_prompt,
        }

        prompt_template_getter = prompt_templates.get(query_type)

        if prompt_template_getter:
            logger.info(f"Using {query_type}-specific prompt template for alternative viewpoint")
            prompt_template = prompt_template_getter()
        else:
            # Fallback to general prompt if type-specific not available
            logger.info(
                f"No specific template for {query_type}, using general prompt template instead"
            )
            prompt_template = PromptManager.get_general_prompt()

        # Build inputs for the prompt
        inputs = RunnableParallel(
            context=RunnableLambda(lambda x: context_string),
            question=RunnablePassthrough(),
        )

        # Create a prompt specifically for alternative viewpoint
        counter_prompt = f"Generate an alternative perspective or counter-argument to: {query}"
        logger.debug(f"Using counter prompt: {counter_prompt}")

        prompt_inputs = await inputs.ainvoke(counter_prompt)
        prompt_messages = await prompt_template.ainvoke(prompt_inputs)

        # Use a higher temperature for exploring alternatives
        alternative_response = await generate_with_fallback(
            prompt=prompt_messages,
            model_name=model_name if model_name is not None else settings.default_model,
            fallback_model="gpt-3.5-turbo",
            temperature=0.7,  # Higher temperature for more creative alternatives
        )

        # Ensure the alternative viewpoint is properly converted to a string
        alternative_viewpoints = StrOutputParser().parse(alternative_response)
        alternative_viewpoints = ensure_string_content(alternative_viewpoints)

        generation_time = time.time() - start_time
        logger.info(f"Alternative viewpoint generated in {generation_time:.2f}s using {model_name}")

        # Update state
        state["alternative_viewpoints"] = alternative_viewpoints

        metrics = ensure_generation_metrics(state)
        metrics["alternative_time"] = generation_time

    except Exception as e:
        # Calculate generation_time if not already set
        if generation_time is None:
            generation_time = time.time() - start_time

        logger.error(f"Failed to generate alternative viewpoints: {str(e)}", exc_info=True)
        # Set to None on error as expected by tests
        state["alternative_viewpoints"] = None
        # Record the error but don't fail the whole process
        metrics = ensure_generation_metrics(state)
        metrics["alternative_error"] = str(e)
        if isinstance(e, asyncio.TimeoutError):
            logger.warning(f"Timeout occurred while generating alternative viewpoint: {str(e)}")

    return state
