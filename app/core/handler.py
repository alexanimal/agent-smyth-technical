"""
Chat handler for processing user queries with RAG capabilities.

This module contains the ChatHandler class which manages the RAG pipeline
for processing user queries, including document retrieval, re-ranking,
prompt selection, and response generation.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from app.core.router import ChatRouter
from app.prompts import PromptManager
from app.rag.graph import app_workflow
from app.utils.document import analyze_sentiment, diversify_documents, extract_sources
from app.utils.technical import get_technical_indicators

# Configure logging
logger = logging.getLogger(__name__)


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

    @property
    def llm(self) -> BaseChatModel:
        """
        Lazy-loaded LLM to avoid initialization overhead until needed.

        Returns:
            BaseChatModel: The initialized chat model
        """
        if self._llm is None:
            self._llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
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
                model=self.model_name,
                temperature=explorer_temp,
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
                model=self.model_name,
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
            classifier_model = ChatOpenAI(model="gpt-4o", temperature=0.0)
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

    async def process_query(self, message: str, k: int = 25) -> Dict[str, Any]:
        """
        Process a user query using the LangGraph workflow.

        This method passes the user's query through the RAG workflow which handles
        query classification, document retrieval, response generation, and
        alternative viewpoint generation.

        Args:
            message: The user's query text
            k: Number of documents to retrieve (default: 25)

        Returns:
            Dictionary containing the response, sources, and metadata

        Raises:
            RuntimeError: If processing fails after maximum retry attempts
        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                start_time = time.time()

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
                }

                # Run the workflow
                result_state = await app_workflow.ainvoke(initial_state)

                # Extract results from the state
                process_time = time.time() - start_time

                # Build response object
                result = {
                    "response": result_state.get("response", "No response generated"),
                    "sources": result_state.get("sources", []),
                    "processing_time": process_time,
                    "query_type": result_state.get("classification", {}).get(
                        "query_type", "general"
                    ),
                    "confidence_scores": result_state.get("classification", {}).get(
                        "confidence_scores", {}
                    ),
                }

                # Add alternative viewpoints if available
                if "alternative_viewpoints" in result_state:
                    result["alternative_viewpoints"] = result_state["alternative_viewpoints"]

                return result

            except Exception as e:
                retries += 1
                last_error = e
                logger.warning(
                    f"Error in chat processing (attempt {retries}/{self.max_retries}): {str(e)}",
                    exc_info=True,
                )

                if retries <= self.max_retries:
                    await asyncio.sleep(self.retry_delay * retries)  # Exponential backoff

        # If all retries failed
        logger.error(f"Failed after {self.max_retries} attempts: {str(last_error)}")
        if isinstance(last_error, BaseException):
            raise last_error
        else:
            raise RuntimeError("Chat processing failed after multiple retries")
