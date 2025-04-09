"""
Module for handling chat interactions using RAG capabilities with conditional orchestration.

This module implements the core chat processing functionality of the RAG Agent, including:
- Query classification to determine the most appropriate response strategy
- Document retrieval and recency-based re-ranking
- Dynamic prompt selection based on query type
- LLM integration using LangChain
- Error handling with retry logic

The module consists of two main classes:
- ChatRouter: Handles query classification to route to appropriate handlers
- ChatHandler: Processes queries with RAG and manages the response generation flow
"""

import asyncio
import logging
import time
from operator import itemgetter
from typing import Any, Dict, List, Optional, Set

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from app.prompts import PromptManager

# Configure logging
logger = logging.getLogger(__name__)


class ChatRouter:
    """
    Routes chat queries to different handlers based on the content.

    This class is responsible for analyzing query content and determining
    the most appropriate query type and processing strategy to use.
    It leverages an LLM classifier to categorize queries.

    Attributes:
        classifier_model: The LLM used for query classification
    """

    def __init__(self, classifier_model: BaseChatModel):
        """
        Initialize the ChatRouter with a classifier model.

        Args:
            classifier_model: The LLM to use for query classification
        """
        self.classifier_model = classifier_model

    async def classify_query(self, query: str) -> str:
        """
        Classifies a query to determine which handler to use.

        Uses an LLM to analyze the query content and categorize it into
        predefined types that determine how it will be processed.

        Args:
            query: The user's query

        Returns:
            str: Classification result ("investment", "trading_thesis", "general", or "technical")

        Raises:
            None: Defaults to "general" if classification fails
        """
        # Get the classification prompt from PromptManager
        classification_prompt = PromptManager.get_classification_prompt()

        # Create a chain for classification
        chain = classification_prompt | self.classifier_model | StrOutputParser()

        # Get the classification
        classification = await chain.ainvoke({"query": query})

        # Ensure we get a valid classification
        classification = classification.strip().lower()
        if classification not in ["investment", "trading_thesis", "general", "technical"]:
            # Default to general if classification is unclear
            logger.warning(f"Invalid classification: {classification}, defaulting to 'general'")
            classification = "general"

        logger.info(f"Query classified as: {classification}")
        return classification


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

    async def _get_technical_indicators(self, message: str, docs: List[Document]) -> str:
        """
        Extract and analyze technical indicators from the query and documents.

        This method identifies potential ticker symbols in the query, extracts
        technical indicator information from the documents, and formats it
        for inclusion in the context.

        Args:
            message: The user's query message
            docs: List of retrieved documents

        Returns:
            str: Formatted technical indicator data
        """
        # Extract potential ticker symbols from the message
        # This is a simplified approach - in production, you would use a more robust method
        import re

        potential_tickers = set(re.findall(r"\$?[A-Z]{1,5}", message))

        # Extract technical indicators from documents
        technical_data: Dict[str, List[str]] = {}
        patterns: Dict[str, List[str]] = {}
        for doc in docs:
            content = doc.page_content.lower()

            # Extract moving averages
            ma_matches = re.findall(r"(sma|ema)[\s-]*(\d+)[^\d]", content)
            for ma_type, period in ma_matches:
                key = f"{ma_type.upper()}{period}"
                if key not in technical_data:
                    technical_data[key] = []
                technical_data[key].append(doc.metadata.get("url", "unknown"))

            # Extract other indicators
            indicators = {
                "rsi": r"rsi[\s-]*(\d+)[^\d]",
                "macd": r"macd",
                "stochastic": r"stoch(astic)?",
                "bollinger": r"bollinger",
                "volume": r"volume",
            }

            for ind_name, pattern in indicators.items():
                if re.search(pattern, content):
                    if ind_name not in technical_data:
                        technical_data[ind_name] = []
                    technical_data[ind_name].append(doc.metadata.get("url", "unknown"))

            # Extract chart patterns
            chart_patterns = [
                "head and shoulders",
                "double top",
                "double bottom",
                "triangle",
                "wedge",
                "flag",
                "pennant",
                "cup and handle",
            ]

            for pattern in chart_patterns:
                if pattern in content:
                    if pattern not in patterns:
                        patterns[pattern] = []
                    patterns[pattern].append(doc.metadata.get("url", "unknown"))

        # Format the technical data
        result = "Technical Indicator Data:\n"

        if technical_data:
            result += "Indicators mentioned in context:\n"
            for indicator, sources in technical_data.items():
                result += f"- {indicator}: mentioned in {len(sources)} sources\n"
        else:
            result += "No specific technical indicators found in the context.\n"

        if patterns:
            result += "\nChart Patterns mentioned in context:\n"
            for pattern, sources in patterns.items():
                result += f"- {pattern.title()}: mentioned in {len(sources)} sources\n"
        else:
            result += "\nNo specific chart patterns found in the context.\n"

        if potential_tickers:
            result += f"\nPotential tickers identified: {', '.join(potential_tickers)}\n"

        return result

    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """
        Extract source URLs from documents.

        Args:
            documents: List of Document objects with metadata

        Returns:
            List[str]: List of unique source URLs from the documents
        """
        sources: Set[str] = set()

        for doc in documents:
            if doc.metadata and "url" in doc.metadata and doc.metadata["url"]:
                sources.add(doc.metadata["url"])

        return list(sources)

    async def process_query(self, message: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a user query with dynamic orchestration and re-ranking by recency.

        This method:
        1. Classifies the query to determine the appropriate processing strategy
        2. Retrieves and re-ranks documents based on recency
        3. Selects an appropriate prompt template based on query type
        4. Invokes the LLM with the prepared context and prompt
        5. Handles any errors with retry logic

        Args:
            message: The user's query message
            k: The final number of documents to use after re-ranking

        Returns:
            Dict containing:
                - response: The generated text response
                - sources: List of source URLs
                - processing_time: Time taken to process the query
                - query_type: The classified query type

        Raises:
            RuntimeError: If all retry attempts fail
            Various exceptions: If an unhandled error occurs during processing
        """
        retries = 0
        last_error = None
        oversample_factor = 2  # Fetch more documents initially for better re-ranking

        while retries <= self.max_retries:
            try:
                start_time = time.time()

                # 1. Classify the query
                query_type = await self.router.classify_query(message)

                # Adjust k if needed for specific query types
                final_k = k
                if query_type in ["trading_thesis", "technical"]:
                    final_k = max(k, 10)  # Fetch more for technical analysis

                # 2. Initial Retrieval (Oversampled)
                initial_k = final_k * oversample_factor
                logger.debug(f"Retrieving initial {initial_k} documents for query: {message}")
                retriever = self.knowledge_base.as_retriever(search_kwargs={"k": initial_k})
                # Use retriever directly instead of RetrievalQA chain
                initial_docs = await retriever.ainvoke(message)
                logger.debug(f"Retrieved {len(initial_docs)} initial documents.")

                # 3. Re-ranking by Timestamp (Recency)
                # Assuming 'timestamp_unix' (float/int) exists in metadata
                valid_docs_with_ts = []
                for doc in initial_docs:
                    timestamp = doc.metadata.get("timestamp_unix")
                    if timestamp is not None:
                        try:
                            # Ensure it's a comparable number (float/int)
                            valid_docs_with_ts.append((doc, float(timestamp)))
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Document metadata has invalid timestamp_unix: {timestamp}. Skipping for sorting."
                            )
                    else:
                        logger.warning(
                            "Document missing 'timestamp_unix' metadata. Skipping for sorting."
                        )

                # Sort by timestamp descending (newest first)
                valid_docs_with_ts.sort(key=itemgetter(1), reverse=True)

                # Select top k documents after sorting
                re_ranked_docs = [doc for doc, ts in valid_docs_with_ts[:final_k]]
                logger.debug(f"Selected {len(re_ranked_docs)} documents after re-ranking.")

                if not re_ranked_docs:
                    logger.warning("No valid documents found after re-ranking. Cannot proceed.")
                    # Handle case with no docs: return empty response or raise error
                    return {
                        "response": "I couldn't find any relevant information.",
                        "sources": [],
                        "processing_time": time.time() - start_time,
                        "query_type": query_type,
                    }

                # 4. Construct Context String
                context_string = "\\n\\n---\\n\\n".join(
                    [doc.page_content for doc in re_ranked_docs]
                )

                # 4a. For trading_thesis or technical, add technical analysis information
                if query_type in ["trading_thesis", "technical"]:
                    logger.debug(f"Adding technical analysis data for {query_type} query")
                    technical_data = await self._get_technical_indicators(message, re_ranked_docs)
                    context_string += f"\n\n{technical_data}"

                # 5. Choose Prompt
                if query_type == "investment":
                    prompt_template = self._get_investment_prompt()
                elif query_type == "trading_thesis":
                    prompt_template = self._get_trading_thesis_prompt()
                elif query_type == "technical":
                    prompt_template = self._get_technical_analysis_prompt()
                else:  # general
                    prompt_template = self._get_general_prompt()

                # 6. Define and Invoke LLM Chain using LCEL
                # Setup runnable to format inputs for the prompt
                inputs = RunnableParallel(
                    context=RunnableLambda(lambda x: context_string),  # Pass re-ranked context
                    question=RunnablePassthrough(),  # Pass original question
                )

                # Choose appropriate LLM based on query type
                model_to_use = (
                    self.technical_llm
                    if query_type in ["trading_thesis", "technical"]
                    else self.llm
                )

                # Chain: Prepare inputs -> Format prompt -> Call LLM -> Parse output
                chain = inputs | prompt_template | model_to_use | StrOutputParser()

                logger.debug(f"Invoking LLM chain for query type: {query_type}")
                response = await chain.ainvoke(message)

                # 7. Extract sources from the re-ranked documents used
                sources = self._extract_sources(re_ranked_docs)

                process_time = time.time() - start_time
                logger.info(f"Query processed in {process_time:.2f} seconds (type: {query_type})")

                return {
                    "response": response,
                    "sources": sources,
                    "processing_time": process_time,
                    "query_type": query_type,
                }

            except Exception as e:
                retries += 1
                last_error = e
                logger.warning(
                    f"Error in chat processing (attempt {retries}/{self.max_retries}): {str(e)}",
                    exc_info=True,  # Log stack trace for debugging
                )

                if retries <= self.max_retries:
                    # Wait before retrying
                    await asyncio.sleep(self.retry_delay * retries)  # Exponential backoff

        # If we get here, all retries failed
        logger.error(
            f"Failed to process chat query after {self.max_retries} attempts: {str(last_error)}"
        )
        if isinstance(last_error, BaseException):
            raise last_error
        else:
            raise RuntimeError(
                f"Chat processing failed after multiple retries, but no specific error captured."
            )
