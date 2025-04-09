"""
Module for defining and managing prompts used in the chat system.

This module centralizes the definition and management of prompt templates
used throughout the RAG Agent for various types of queries and interactions.
It provides a structured way to:

- Define consistent prompts for different query types
- Isolate prompt engineering concerns from other application logic
- Enable easy updates to prompts without changing processing logic
- Support specialized prompts for different subject domains

The PromptManager class serves as a factory for getting appropriately
formatted prompts for different use cases.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class PromptManager:
    """
    Manages various prompt templates for different use cases.

    This class provides a collection of static methods that return predefined
    prompt templates for different query types and processing scenarios.
    Each method returns a properly configured ChatPromptTemplate that can
    be used directly in a LangChain processing chain.

    The class centralizes prompt engineering and enables prompt modifications
    without changing the core processing logic elsewhere in the application.
    """

    @staticmethod
    def get_investment_prompt() -> ChatPromptTemplate:
        """
        Get the investment-specific prompt template.

        Creates a specialized prompt template for financial investment queries
        that instructs the LLM to provide a concise, single-word investment
        recommendation based on tweet sentiment.

        Returns:
            ChatPromptTemplate: A configured prompt template for investment queries
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You are a financial advisor. Based on the tweet data provided, respond with ONLY ONE WORD:
            - "Long" if the sentiment suggests buying would be profitable
            - "Short" if the sentiment suggests selling would be wise
            - "Hold" if the sentiment is neutral or unclear

            Do not provide any other text, explanation, or context. Just the single word.
            """,
                ),
                ("human", "{question}"),
                ("system", "Context tweets:\n\n{context}"),
            ]
        )

    @staticmethod
    def get_general_prompt() -> ChatPromptTemplate:
        """
        Get the general-purpose prompt template.

        Creates a versatile prompt template for answering general questions
        based on tweet data, with instructions to rely solely on provided context
        and acknowledge information gaps.

        Returns:
            ChatPromptTemplate: A configured prompt template for general queries
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You are a helpful AI assistant who answers questions based on Twitter data.
            Your goal is to provide informative, factual responses based solely on the provided tweets.

            When answering:
            1. Use only the information from the tweet context provided
            2. Be concise and clear
            3. If the information is not in the context, say "I don't have enough information to answer that question"
            4. Do not make up facts or reference sources not in the context
            5. If appropriate, mention how many tweets support your answer
            """,
                ),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{question}"),
                (
                    "system",
                    "Here are some relevant tweets that might help you answer:\n\n{context}",
                ),
            ]
        )

    @staticmethod
    def get_trading_thesis_prompt() -> ChatPromptTemplate:
        """
        Get a prompt template for transforming PM notes into comprehensive trading theses.

        Creates a specialized prompt template for converting rough portfolio manager
        notes into structured trading theses, with explicit attention to bias mitigation,
        balanced analysis, and technical analysis indicators and patterns.

        Returns:
            ChatPromptTemplate: A configured prompt template for trading thesis generation
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You are an expert financial analyst skilled in both fundamental and technical analysis. Your task is to transform rough notes from portfolio managers (PMs) into comprehensive trading theses. Follow this structured approach:

            1. ANALYSIS PHASE:
               - Identify the core investment idea in the PM's note
               - Extract any mentioned assets, trends, or market conditions
               - Note any explicit or implicit biases in the original thinking

            2. FUNDAMENTAL THESIS DEVELOPMENT:
               - Develop a clear, structured trading thesis with:
                 a) Core hypothesis
                 b) Key drivers & catalysts
                 c) Time horizon
                 d) Risk/reward profile

            3. TECHNICAL ANALYSIS:
               - Analyze technical indicators mentioned in the context such as:
                 a) Moving Averages (SMA/EMA crossovers, trends)
                 b) Momentum indicators (RSI, MACD, Stochastic)
                 c) Volatility measures (Bollinger Bands, ATR)
                 d) Volume indicators (OBV, CMF)
               - Interpret indicator readings (overbought/oversold, bullish/bearish divergence)
               - Identify key support and resistance levels
               - Determine if technical indicators confirm or contradict the fundamental thesis

            4. CHART PATTERN ANALYSIS:
               - Identify any notable chart patterns mentioned (Head & Shoulders, Double Tops/Bottoms, etc.)
               - Assess pattern reliability and potential price targets
               - Evaluate pattern completion status and potential breakout/breakdown points

            5. MULTI-TIMEFRAME ANALYSIS:
               - Compare trends across multiple timeframes mentioned (daily, weekly, monthly)
               - Identify potential divergences between timeframes
               - Determine whether short-term movements align with longer-term trends

            6. PRECISE ENTRY/EXIT STRATEGY:
               - Specify exact entry points based on technical levels
               - Define clear stop-loss levels based on technical supports
               - Calculate multiple profit targets with specific risk-reward ratios (e.g., 1:2, 1:3)
               - Recommend position sizing based on risk tolerance

            7. BIAS MITIGATION:
               - Explicitly identify potential confirmation biases in the original note
               - Present counter-arguments to the main thesis based on both fundamentals and technicals
               - Consider alternative scenarios and outcomes
               - Evaluate disconfirming evidence from the tweet data

            8. SUPPORTING EVIDENCE:
               - Use ONLY information from the provided tweet context
               - Cite specific tweets that either support or contradict the thesis
               - Avoid making up facts not present in the context

            9. FORMAT:
               - Present your analysis in a professional, structured format with clear sections
               - Include a dedicated "Technical Analysis" section
               - Include a "Confirmation Bias Analysis" section specifically addressing potential biases
               - End with a balanced conclusion that presents both bullish and bearish perspectives

            Remember to maintain objectivity and interpret technical indicators within the context of the broader market environment.
            """,
                ),
                (
                    "human",
                    """
            Transform this PM note into a comprehensive trading thesis with technical analysis:

            {question}
            """,
                ),
                ("system", "Here is relevant market context from Twitter:\n\n{context}"),
            ]
        )

    @staticmethod
    def get_classification_prompt() -> ChatPromptTemplate:
        """
        Get the prompt template for query classification.

        Creates a prompt template specifically designed to classify incoming
        queries into predefined categories that determine how they will be processed.

        Returns:
            ChatPromptTemplate: A configured prompt template for query classification
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You are a query classifier. Your job is to determine if a query is about:
            1. Investments, stocks, markets, finance, trading, buy/sell decisions - respond with "investment"
            2. A request to transform a PM (Portfolio Manager) note into a trading thesis - respond with "trading_thesis"
            3. Technical analysis, chart patterns, technical indicators (RSI, MACD, Bollinger Bands, etc.), support/resistance levels - respond with "technical"
            4. Any other topic - respond with "general"

            Respond with ONLY "investment", "trading_thesis", "technical", or "general".
            """,
                ),
                ("human", "{query}"),
            ]
        )

    @staticmethod
    def get_technical_analysis_prompt() -> ChatPromptTemplate:
        """
        Get a specialized prompt template for technical analysis queries.

        Creates a prompt template specifically focused on analyzing technical
        indicators, chart patterns, and providing precise trading recommendations
        based on technical analysis principles.

        Returns:
            ChatPromptTemplate: A configured prompt template for technical analysis
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You are an expert technical analyst specializing in market technicals and chart analysis.
            Your task is to analyze the technical indicators and chart patterns from the provided context
            and deliver precise, actionable trading insights.

            Follow this structured approach:

            1. TECHNICAL INDICATOR ANALYSIS:
               - Analyze all mentioned technical indicators including:
                 a) Moving Averages (SMA/EMA) - crossovers, trends, and divergences
                 b) Momentum indicators (RSI, MACD, Stochastic) - overbought/oversold conditions, divergences
                 c) Volatility indicators (Bollinger Bands, ATR) - contractions, expansions, touches
                 d) Volume indicators (OBV, Volume Profile) - confirmation/divergence from price
               - Interpret the current readings and their implications
               - Identify bullish or bearish signals from each indicator

            2. CHART PATTERN RECOGNITION:
               - Identify and analyze any chart patterns mentioned:
                 a) Continuation patterns (flags, pennants, triangles)
                 b) Reversal patterns (head & shoulders, double tops/bottoms)
                 c) Candlestick patterns (engulfing, doji, hammers)
               - Assess pattern completion status and reliability
               - Calculate price targets based on pattern measurements

            3. SUPPORT & RESISTANCE ANALYSIS:
               - Identify key price levels from the context
               - Analyze prior reactions at these levels
               - Determine strength of each level based on:
                 a) Number of touches
                 b) Volume at each touch
                 c) Time spent at each level

            4. MULTI-TIMEFRAME CONFIRMATION:
               - Analyze trends across any mentioned timeframes
               - Identify alignment or divergence between timeframes
               - Determine dominant timeframe trend

            5. PRECISE TRADE RECOMMENDATION:
               - Specify exact entry points with price levels
               - Define specific stop-loss levels with justification
               - Set multiple take-profit targets with:
                 a) Conservative target (highest probability)
                 b) Moderate target
                 c) Ambitious target (if strong trend)
               - Recommend position sizing based on risk metrics

            6. RISK/REWARD ASSESSMENT:
               - Calculate exact risk/reward ratios for each target
               - Determine probability of success based on technical factors
               - Identify specific technical conditions that would invalidate the analysis

            FORMAT YOUR RESPONSE:
            - Begin with a concise summary of the overall technical outlook
            - Structure your analysis in clearly labeled sections
            - Use bullet points for clarity when appropriate
            - Conclude with a clear, actionable recommendation
            - Only reference information contained in the provided context
            - If critical technical data is missing, acknowledge the limitation

            Remember that precision is crucial - provide specific price levels, indicator readings,
            and timeframes whenever possible rather than general statements.
            """,
                ),
                ("human", "{question}"),
                ("system", "Here is the relevant technical data from the tweets:\n\n{context}"),
            ]
        )
