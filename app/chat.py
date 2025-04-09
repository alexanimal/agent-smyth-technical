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
import json
import logging
import random
import time
from operator import itemgetter
from typing import Any, Dict, List, Optional, Set, Tuple

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
    Routes chat queries to different handlers based on the content with confidence scoring.

    This class is responsible for analyzing query content and determining
    the most appropriate query type and processing strategy to use.
    It provides confidence scores for multiple categories to enable
    more nuanced handling of mixed-type queries.

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

    async def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classifies a query and returns confidence scores for multiple categories.

        Uses an LLM to analyze the query content and provide confidence scores for
        each possible category, allowing for nuanced, non-binary classification.

        Args:
            query: The user's query

        Returns:
            Dict containing:
                - query_type: The highest confidence category
                - confidence_scores: Dict mapping categories to confidence values (0-100)
                - is_mixed: Boolean indicating if query spans multiple categories significantly

        Raises:
            None: Defaults to "general" with low confidence if classification fails
        """
        # Get the classification prompt from PromptManager
        classification_prompt = PromptManager.get_classification_prompt()

        # Create a chain for classification
        chain = classification_prompt | self.classifier_model | StrOutputParser()

        # Get the classification
        try:
            classification_result = await chain.ainvoke({"query": query})

            # Parse the JSON result with confidence scores
            confidence_scores = json.loads(classification_result.strip())

            # Validate the format
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

            # Determine the highest confidence category
            max_category = max(confidence_scores.items(), key=itemgetter(1))[0]

            # Determine if this is a mixed query (second highest score > threshold)
            scores = sorted(confidence_scores.values(), reverse=True)
            is_mixed = len(scores) > 1 and scores[1] > 30  # Second score > 30% indicates mixed type

            result = {
                "query_type": max_category,
                "confidence_scores": confidence_scores,
                "is_mixed": is_mixed,
            }

            logger.info(
                f"Query classified as: {max_category} with confidence scores: {confidence_scores}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in classification: {str(e)}", exc_info=True)
            # Default to general with low confidence
            return {
                "query_type": "general",
                "confidence_scores": {
                    "technical": 0,
                    "trading_thesis": 0,
                    "investment": 0,
                    "general": 100,
                },
                "is_mixed": False,
            }


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

    async def _analyze_sentiment(self, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Analyze sentiment of documents to enable diversity-aware re-ranking.

        This simplified sentiment analysis assigns sentiment scores to documents
        to help ensure representation of diverse viewpoints in the final context.

        Args:
            documents: List of Document objects to analyze

        Returns:
            List of tuples containing (document, sentiment_score)
            where sentiment_score ranges from -1.0 (very negative) to 1.0 (very positive)
        """
        scored_docs = []

        for doc in documents:
            # Simple keyword-based sentiment analysis
            # In a production system, this would use a proper sentiment analysis model
            content = doc.page_content.lower()

            # Positive sentiment keywords
            pos_words = [
                "bullish",
                "positive",
                "uptrend",
                "buy",
                "growth",
                "oversold",
                "support",
                "opportunity",
                "undervalued",
                "breakout",
                "upside",
            ]

            # Negative sentiment keywords
            neg_words = [
                "bearish",
                "negative",
                "downtrend",
                "sell",
                "decline",
                "overbought",
                "resistance",
                "risk",
                "overvalued",
                "breakdown",
                "downside",
            ]

            # Count keyword occurrences
            pos_count = sum(content.count(word) for word in pos_words)
            neg_count = sum(content.count(word) for word in neg_words)

            # Calculate sentiment score (-1 to 1)
            total = pos_count + neg_count
            if total > 0:
                sentiment = (pos_count - neg_count) / total
            else:
                sentiment = 0.0  # Neutral if no sentiment words found

            scored_docs.append((doc, sentiment))

        return scored_docs

    async def _diversify_documents(
        self, docs_with_timestamps: List[Tuple[Document, float]], k: int
    ) -> List[Document]:
        """
        Re-rank documents to balance recency and viewpoint diversity.

        This method ensures the final selection of documents presents diverse
        perspectives by balancing recency with sentiment diversity.

        Args:
            docs_with_timestamps: Documents with timestamp information
            k: Number of documents to select

        Returns:
            List of selected documents with diverse viewpoints
        """
        if not docs_with_timestamps or k <= 0:
            return []

        # Extract sentiment scores
        docs_with_sentiment = await self._analyze_sentiment(
            [doc for doc, _ in docs_with_timestamps]
        )

        # Combine timestamp and sentiment information
        docs_with_metadata = []
        for (doc1, timestamp), (doc2, sentiment) in zip(docs_with_timestamps, docs_with_sentiment):
            # Verify they're the same document
            if doc1.page_content == doc2.page_content:
                docs_with_metadata.append((doc1, timestamp, sentiment))

        # If k is small, ensure diversity directly
        if k <= 5:
            # First, sort by recency (highest timestamp first)
            docs_with_metadata.sort(key=lambda x: x[1], reverse=True)

            # Take top k/2 most recent documents (rounded up)
            recent_half = docs_with_metadata[: max(1, k // 2)]

            # For the rest, ensure diverse sentiment
            remaining = docs_with_metadata[max(1, k // 2) :]

            # Sort remaining by absolute sentiment (to get strongest opinions)
            remaining.sort(key=lambda x: abs(x[2]), reverse=True)

            # Group by positive and negative sentiment
            positive = [item for item in remaining if item[2] > 0]
            negative = [item for item in remaining if item[2] < 0]
            neutral = [item for item in remaining if item[2] == 0]

            # Alternate selection from positive and negative groups
            diverse_selection: List[Tuple[Document, float, float]] = []
            while len(diverse_selection) < (k - len(recent_half)) and (
                positive or negative or neutral
            ):
                if positive and (not diverse_selection or diverse_selection[-1][2] <= 0):
                    diverse_selection.append(positive.pop(0))
                elif negative and (not diverse_selection or diverse_selection[-1][2] >= 0):
                    diverse_selection.append(negative.pop(0))
                elif neutral:
                    diverse_selection.append(neutral.pop(0))
                elif positive:
                    diverse_selection.append(positive.pop(0))
                elif negative:
                    diverse_selection.append(negative.pop(0))

            # Combine recent and diverse selections
            combined = recent_half + diverse_selection
            return [item[0] for item in combined[:k]]

        # For larger k, use a more sophisticated approach
        else:
            # Get timestamp range for normalization
            if docs_with_metadata:
                max_ts = max(item[1] for item in docs_with_metadata)
                min_ts = min(item[1] for item in docs_with_metadata)
                ts_range = max(0.001, max_ts - min_ts)  # Avoid division by zero

                # Compute combined score = 0.7*recency + 0.3*diversity
                scored_docs = []

                # Track sentiments we've seen so far
                selected_sentiments: List[float] = []

                for doc, ts, sentiment in docs_with_metadata:
                    # Normalize timestamp to 0-1 (1 = most recent)
                    recency_score = (ts - min_ts) / ts_range

                    # Compute diversity score based on how different this document's
                    # sentiment is from already selected documents
                    diversity_score = 1.0
                    if selected_sentiments:
                        # Lower score if similar sentiments already selected
                        similarity = min(abs(sentiment - s) for s in selected_sentiments)
                        diversity_score = min(1.0, similarity * 5)  # Scale up for effect

                    # Combined score with weightings
                    combined_score = (0.7 * recency_score) + (0.3 * diversity_score)
                    scored_docs.append((doc, combined_score, sentiment))

                # Sort by combined score (highest first)
                scored_docs.sort(key=lambda x: x[1], reverse=True)

                # Select top documents
                result = []
                for i in range(min(k, len(scored_docs))):
                    result.append(scored_docs[i][0])
                    selected_sentiments.append(scored_docs[i][2])

                return result

            return [doc for doc, _, _ in docs_with_metadata[:k]]

    async def process_query(self, message: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a user query with dynamic orchestration and diverse viewpoint re-ranking.

        This method:
        1. Classifies the query to determine the appropriate processing strategy with confidence scores
        2. Retrieves, analyzes, and diversifies documents to ensure balanced perspectives
        3. Selects an appropriate prompt template based on query type
        4. Invokes the LLM with the prepared context and prompt
        5. Handles any errors with retry logic
        6. For certain query types, generates counter-narratives with a higher-temperature model

        Args:
            message: The user's query message
            k: The final number of documents to use after re-ranking

        Returns:
            Dict containing:
                - response: The generated text response
                - sources: List of source URLs
                - processing_time: Time taken to process the query
                - query_type: The classified query type
                - confidence_scores: Classification confidence by category
                - alternative_viewpoints: Counter-arguments for balanced perspective (if applicable)

        Raises:
            RuntimeError: If all retry attempts fail
            Various exceptions: If an unhandled error occurs during processing
        """
        retries = 0
        last_error = None
        oversample_factor = 3  # Fetch more documents initially for better diversity

        while retries <= self.max_retries:
            try:
                start_time = time.time()

                # 1. Classify the query with confidence scores
                classification = await self.router.classify_query(message)
                query_type = classification["query_type"]
                confidence_scores = classification["confidence_scores"]
                is_mixed_query = classification["is_mixed"]

                # Adjust k if needed for specific query types or mixed queries
                final_k = k
                if query_type in ["trading_thesis", "technical"] or is_mixed_query:
                    final_k = max(k, 10)  # Fetch more for complex analysis

                # 2. Initial Retrieval (Oversampled)
                initial_k = final_k * oversample_factor
                logger.debug(f"Retrieving initial {initial_k} documents for query: {message}")
                retriever = self.knowledge_base.as_retriever(search_kwargs={"k": initial_k})
                # Use retriever directly instead of RetrievalQA chain
                initial_docs = await retriever.ainvoke(message)
                logger.debug(f"Retrieved {len(initial_docs)} initial documents.")

                # 3. Process Document Metadata
                # Extract timestamp metadata for each document
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

                # 4. Diversity-aware re-ranking
                re_ranked_docs = await self._diversify_documents(valid_docs_with_ts, final_k)

                logger.debug(
                    f"Selected {len(re_ranked_docs)} documents after diversity-aware re-ranking."
                )

                if not re_ranked_docs:
                    logger.warning("No valid documents found after re-ranking. Cannot proceed.")
                    # Handle case with no docs: return empty response or raise error
                    return {
                        "response": "I couldn't find any relevant information.",
                        "sources": [],
                        "processing_time": time.time() - start_time,
                        "query_type": query_type,
                        "confidence_scores": confidence_scores,
                    }

                # 5. Construct Context String
                context_string = "\\n\\n---\\n\\n".join(
                    [doc.page_content for doc in re_ranked_docs]
                )

                # 5a. For trading_thesis or technical, add technical analysis information
                if query_type in ["trading_thesis", "technical"]:
                    logger.debug(f"Adding technical analysis data for {query_type} query")
                    technical_data = await self._get_technical_indicators(message, re_ranked_docs)
                    context_string += f"\n\n{technical_data}"

                # 6. Choose Prompt
                if query_type == "investment":
                    prompt_template = self._get_investment_prompt()
                elif query_type == "trading_thesis":
                    prompt_template = self._get_trading_thesis_prompt()
                elif query_type == "technical":
                    prompt_template = self._get_technical_analysis_prompt()
                else:  # general
                    prompt_template = self._get_general_prompt()

                # 7. Define and Invoke LLM Chain using LCEL
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

                # 8. Generate counter-narrative for certain query types
                alternative_viewpoints = None
                if (
                    query_type in ["investment", "trading_thesis", "technical"]
                    and not is_mixed_query
                ):
                    try:
                        # Use a higher temperature model to generate alternative perspectives
                        counter_chain = (
                            inputs | prompt_template | self.explorer_llm | StrOutputParser()
                        )
                        alternative_viewpoints = await counter_chain.ainvoke(
                            "Generate an alternative perspective or counter-argument to: " + message
                        )
                        logger.debug("Generated alternative viewpoints for balanced perspective")
                    except Exception as e:
                        logger.warning(f"Failed to generate alternative viewpoints: {str(e)}")

                # 9. Extract sources from the re-ranked documents used
                sources = self._extract_sources(re_ranked_docs)

                process_time = time.time() - start_time
                logger.info(f"Query processed in {process_time:.2f} seconds (type: {query_type})")

                result = {
                    "response": response,
                    "sources": sources,
                    "processing_time": process_time,
                    "query_type": query_type,
                    "confidence_scores": confidence_scores,
                }

                # Add alternative viewpoints if available
                if alternative_viewpoints:
                    result["alternative_viewpoints"] = alternative_viewpoints

                return result

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
