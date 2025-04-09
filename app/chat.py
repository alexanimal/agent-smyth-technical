"""
Module for handling chat interactions using RAG capabilities with conditional orchestration.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Set, Optional

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from app.prompts import PromptManager

# Configure logging
logger = logging.getLogger(__name__)


class ChatRouter:
    """
    Routes chat queries to different handlers based on the content.
    """

    def __init__(self, classifier_model: BaseChatModel):
        self.classifier_model = classifier_model

    async def classify_query(self, query: str) -> str:
        """
        Classifies a query to determine which handler to use.

        Args:
            query: The user's query

        Returns:
            Classification: "investment", "trading_thesis", or "general"
        """
        # Get the classification prompt from PromptManager
        classification_prompt = PromptManager.get_classification_prompt()

        # Create a chain for classification
        chain = classification_prompt | self.classifier_model | StrOutputParser()

        # Get the classification
        classification = await chain.ainvoke({"query": query})

        # Ensure we get a valid classification
        classification = classification.strip().lower()
        if classification not in ["investment", "trading_thesis", "general"]:
            # Default to general if classification is unclear
            logger.warning(f"Invalid classification: {classification}, defaulting to 'general'")
            classification = "general"

        logger.info(f"Query classified as: {classification}")
        return classification


class ChatHandler:
    """
    Handles chat interactions using RAG with conditional response strategies.
    """

    def __init__(
        self,
        knowledge_base: VectorStore,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize ChatHandler with a knowledge base."""
        self.knowledge_base = knowledge_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._llm: Optional[BaseChatModel] = None
        self._router: Optional[ChatRouter] = None

    @property
    def llm(self) -> BaseChatModel:
        """Lazy-loaded LLM to avoid initialization overhead until needed."""
        if self._llm is None:
            self._llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        return self._llm

    @property
    def router(self) -> ChatRouter:
        """Lazy-loaded router for query classification."""
        if self._router is None:
            # Use a smaller, faster model for classification
            classifier_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
            self._router = ChatRouter(classifier_model)
        return self._router

    def _get_investment_prompt(self) -> ChatPromptTemplate:
        """Get the investment-specific prompt template."""
        return PromptManager.get_investment_prompt()

    def _get_general_prompt(self) -> ChatPromptTemplate:
        """Get the general-purpose prompt template."""
        return PromptManager.get_general_prompt()

    def _get_trading_thesis_prompt(self) -> ChatPromptTemplate:
        """Get a prompt template for transforming PM notes into comprehensive trading theses."""
        return PromptManager.get_trading_thesis_prompt()

    def _create_qa_chain(self, prompt_template: ChatPromptTemplate, k: int = 5) -> RetrievalQA:
        """
        Create a retrieval QA chain with the specified prompt template.

        Args:
            prompt_template: The prompt template to use
            k: Number of documents to retrieve

        Returns:
            A configured RetrievalQA chain
        """
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.knowledge_base.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )

    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """Extract source URLs from documents."""
        sources: Set[str] = set()

        for doc in documents:
            if doc.metadata and "url" in doc.metadata and doc.metadata["url"]:
                sources.add(doc.metadata["url"])

        return list(sources)

    async def process_query(self, message: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a user query with dynamic orchestration.

        Args:
            message: The user's query message
            k: Number of documents to retrieve

        Returns:
            Dict containing response and sources
        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                start_time = time.time()

                # Classify the query
                query_type = await self.router.classify_query(message)

                # Choose the appropriate prompt based on classification
                if query_type == "investment":
                    prompt_template = self._get_investment_prompt()
                elif query_type == "trading_thesis":
                    prompt_template = self._get_trading_thesis_prompt()
                    # For trading thesis, we want more relevant documents
                    k = max(k, 10)
                else:
                    prompt_template = self._get_general_prompt()

                # Create QA chain with the selected prompt
                qa_chain = self._create_qa_chain(prompt_template, k)

                # Invoke the chain
                result = qa_chain.invoke(message)

                # Extract sources
                sources = self._extract_sources(result.get("source_documents", []))

                process_time = time.time() - start_time
                logger.info(f"Query processed in {process_time:.2f} seconds (type: {query_type})")

                return {
                    "response": result["result"],
                    "sources": sources,
                    "processing_time": process_time,
                    "query_type": query_type,
                }

            except Exception as e:
                retries += 1
                last_error = e
                logger.warning(
                    f"Error in chat processing (attempt {retries}/{self.max_retries}): {str(e)}"
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
            raise RuntimeError(f"Chat processing failed after multiple retries, but no specific error captured.")
