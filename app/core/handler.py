"""
Chat handler for processing user queries with RAG capabilities.

This module contains the ChatHandler class which manages the RAG pipeline
for processing user queries, including document retrieval, re-ranking,
prompt selection, and response generation.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from app.config import settings
from app.core.router import ChatRouter
from app.prompts import PromptManager
from app.rag.graph import app_workflow
from app.utils.document import analyze_sentiment, diversify_documents, extract_sources
from app.utils.technical import get_technical_indicators

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

    # For everything else, ensure it's a string
    if message_or_string is not None and not isinstance(message_or_string, str):
        logger.debug(f"Converting {type(message_or_string)} to string")
        return str(message_or_string)

    return message_or_string


class ChatHandler:
    """
    Handles chat interactions using RAG with conditional response strategies.

    This class manages the entire RAG pipeline, including document retrieval,
    re-ranking, prompt selection, LLM invocation, and error handling. It
    supports different query types with specialized processing strategies.

    Attributes:
        knowledge_base: Vector store containing the indexed documents
        model_name: Name of the OpenAI model to use for responses
        temperature: Temperature setting for the LLM (0-1, higher = more creative)
        max_retries: Maximum number of retry attempts on failure
        retry_delay: Base delay in seconds between retry attempts
        _llm: Cached LLM instance (lazy-loaded)
        _router: Cached router instance (lazy-loaded)
        _technical_llm: Cached LLM instance for technical analysis (lazy-loaded)
    """

    def __init__(
        self,
        knowledge_base: VectorStore,
        model_name: str = "gpt-4o",
        temperature: float = 0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize ChatHandler with a knowledge base and configuration.

        Args:
            knowledge_base: Vector store containing the indexed documents
            model_name: Name of the OpenAI model to use for responses
            temperature: Temperature setting for the LLM (0-1)
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Base delay in seconds between retry attempts
        """
        self.knowledge_base = knowledge_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._llm: Optional[BaseChatModel] = None
        self._router: Optional[ChatRouter] = None
        self._technical_llm: Optional[BaseChatModel] = None
        self._explorer_llm: Optional[BaseChatModel] = None
        self._model_instances: Dict[str, BaseChatModel] = {}  # Cache for model instances

    def get_llm(self, model_name: Optional[str] = None) -> BaseChatModel:
        """
        Get an LLM instance for the specified model name.

        Args:
            model_name: Name of the model to use, or None to use the default model

        Returns:
            BaseChatModel: The initialized chat model

        Raises:
            ValueError: If the model name is not in the allowed models list
        """
        # Use the provided model name or fall back to the default
        model_to_use = model_name or self.model_name

        # Validate the model name
        if model_to_use not in settings.allowed_models:
            raise ValueError(
                f"Model '{model_to_use}' is not allowed. Allowed models: {settings.allowed_models}"
            )

        # Create or retrieve the model instance from cache
        if model_to_use not in self._model_instances:
            self._model_instances[model_to_use] = ChatOpenAI(
                model_name=model_to_use, temperature=self.temperature  # type: ignore
            )

        return self._model_instances[model_to_use]

    @property
    def llm(self) -> BaseChatModel:
        """
        Lazy-loaded LLM to avoid initialization overhead until needed.

        Returns:
            BaseChatModel: The initialized chat model
        """
        if self._llm is None:
            self._llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)  # type: ignore
        return self._llm

    @property
    def explorer_llm(self) -> BaseChatModel:
        """
        Specialized LLM with higher temperature for exploring alternative hypotheses.

        Uses a higher temperature to encourage more creative and diverse thinking
        when generating alternative scenarios or counter-arguments.

        Returns:
            BaseChatModel: The initialized chat model with higher temperature
        """
        if self._explorer_llm is None:
            # Use higher temperature for exploration (+0.3)
            explorer_temp = min(1.0, self.temperature + 0.3)
            self._explorer_llm = ChatOpenAI(
                model_name=self.model_name,  # type: ignore
                temperature=explorer_temp,  # type: ignore
            )
        return self._explorer_llm

    @property
    def technical_llm(self) -> BaseChatModel:
        """
        Specialized LLM optimized for technical analysis tasks.

        Uses a slightly lower temperature for more deterministic analysis
        of technical indicators and patterns.

        Returns:
            BaseChatModel: The initialized chat model for technical analysis
        """
        if self._technical_llm is None:
            # Use the same model but with lower temperature for technical analysis
            self._technical_llm = ChatOpenAI(
                model_name=self.model_name,  # type: ignore
                temperature=max(0, self.temperature - 0.2),  # Lower temperature
            )
        return self._technical_llm

    @property
    def router(self) -> ChatRouter:
        """
        Lazy-loaded router for query classification.

        Returns:
            ChatRouter: The initialized chat router
        """
        if self._router is None:
            # Use a smaller, faster model for classification
            classifier_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)  # type: ignore
            self._router = ChatRouter(classifier_model)
        return self._router

    def _get_investment_prompt(self) -> ChatPromptTemplate:
        """
        Get the investment-specific prompt template.

        Returns:
            ChatPromptTemplate: Prompt template for investment queries
        """
        return PromptManager.get_investment_prompt()

    def _get_general_prompt(self) -> ChatPromptTemplate:
        """
        Get the general-purpose prompt template.

        Returns:
            ChatPromptTemplate: Prompt template for general queries
        """
        return PromptManager.get_general_prompt()

    def _get_trading_thesis_prompt(self) -> ChatPromptTemplate:
        """
        Get a prompt template for transforming PM notes into comprehensive trading theses.

        Returns:
            ChatPromptTemplate: Prompt template for trading thesis queries
        """
        return PromptManager.get_trading_thesis_prompt()

    def _get_technical_analysis_prompt(self) -> ChatPromptTemplate:
        """
        Get a specialized prompt template for technical analysis.

        Returns:
            ChatPromptTemplate: Prompt template for technical analysis queries
        """
        return PromptManager.get_technical_analysis_prompt()

    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """
        Extract source URLs from a list of documents.

        Delegates to the utility function extract_sources to get unique URLs
        from the document metadata.

        Args:
            documents: List of Document objects with metadata

        Returns:
            List[str]: List of unique source URLs from the documents
        """
        return extract_sources(documents)

    async def process_query(
        self,
        message: str,
        k: int = 25,
        ranking_weights: Optional[Dict[str, float]] = None,
        model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        generate_alternative_viewpoint: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a user query using the LangGraph workflow.

        This method passes the user's query through the RAG workflow which handles
        query classification, document retrieval, response generation, and
        post-processing of the response.

        Args:
            message: The user's query text
            k: Number of documents to retrieve (default: 25)
            ranking_weights: Optional weights for document ranking
            model: Optional model name to override the default
            context: Optional context information for the query
            generate_alternative_viewpoint: Whether to generate an alternative perspective (default: False)

        Returns:
            Dict containing the response, sources, and metadata

        Raises:
            ValueError: If an invalid model is specified or if max retries are exceeded
        """
        start_time = time.time()
        logger.info(f"Processing query: '{message[:100]}...' with k={k}")

        # Validate model if provided
        if model and model not in settings.allowed_models:
            raise ValueError(
                f"Model '{model}' is not allowed. Allowed models: {settings.allowed_models}"
            )

        # Use the provided model or fall back to the default
        model_to_use = model or self.model_name
        logger.info(f"Using model: {model_to_use}")

        # Initialize or get the LLM for the specified model
        llm_instance = self.get_llm(model_to_use)

        # Set default ranking weights if not provided
        if ranking_weights is None:
            ranking_weights = {
                "recency_weight": 0.4,  # Default weight for recency (timestamp)
                "view_weight": 0.2,  # Default weight for view count
                "like_weight": 0.2,  # Default weight for like count
                "retweet_weight": 0.2,  # Default weight for retweet count
            }

        # Initialize the state
        initial_state = {
            "query": message,
            "retrieved_docs": [],
            "ranked_docs": [],
            "response": "",
            "sources": [],
            "classification": {},
            "alternative_viewpoints": None,
            "num_results": k,  # Include the user-requested document count
            "ranking_config": ranking_weights,  # Include ranking configuration
            "model": model_to_use,  # Include the model to use throughout the workflow
            "generate_alternative_viewpoint": generate_alternative_viewpoint,  # Include flag for alternative viewpoints
        }

        # Implement retry mechanism with exponential backoff
        attempts = 0
        last_error = None

        while attempts <= self.max_retries:
            try:
                # Run the workflow
                result_state = await app_workflow.ainvoke(initial_state)

                # Extract results from the state
                process_time = time.time() - start_time

                # Process and ensure all message objects are converted to strings
                response_text = ensure_string_content(
                    result_state.get("response", "No response generated")
                )

                # Keep sources as a list but ensure each item is a string
                sources = result_state.get("sources", [])

                # Get classification with query type
                classification = result_state.get("classification", {})
                query_type = ensure_string_content(classification.get("query_type", "general"))

                # Build response object with proper string values
                result = {
                    "response": response_text,
                    "sources": sources,
                    "processing_time": process_time,
                    "query_type": query_type,
                    "confidence_scores": classification.get("confidence_scores", {}),
                    "model_used": model_to_use,  # Include the model that was used
                }

                # Add alternative viewpoints if available, ensuring it's a string
                if "alternative_viewpoints" in result_state:
                    result["alternative_viewpoints"] = ensure_string_content(
                        result_state["alternative_viewpoints"]
                    )
                    logger.info(f"Alternative viewpoints successfully processed as string")

                return result

            except Exception as e:
                attempts += 1
                last_error = e
                logger.warning(
                    f"Error processing query (attempt {attempts}/{self.max_retries + 1}): {str(e)}"
                )

                if attempts <= self.max_retries:
                    # Calculate exponential backoff delay
                    backoff_delay = self.retry_delay * (2 ** (attempts - 1))
                    logger.info(f"Retrying in {backoff_delay:.2f} seconds...")
                    await asyncio.sleep(backoff_delay)
                else:
                    # Max retries exceeded
                    logger.error(f"Max retries exceeded. Last error: {str(last_error)}")
                    raise last_error

        # This code shouldn't be reached in normal operation due to the raise above,
        # but adding a return to satisfy mypy
        return {
            "response": "Error: Maximum retry attempts exceeded",
            "sources": [],
            "processing_time": time.time() - start_time,
            "query_type": "error",
            "confidence_scores": {},
            "model_used": model_to_use,
            "error": str(last_error) if last_error else "Unknown error occurred",
        }
