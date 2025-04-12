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
        that instructs the LLM to provide a single-word position recommendation.

        Returns:
            ChatPromptTemplate: A configured prompt template for investment queries
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            IMPORTANT: YOUR ENTIRE RESPONSE MUST BE ONLY ONE OF THESE THREE WORDS: "Long", "Short", or "Hold".
            DO NOT include any other text, explanations, or rationale in your response.

            You are a financial advisor with expertise in assessing market sentiment. Based on the tweet data provided,
            internally consider the following to determine your recommendation, but DO NOT output this analysis:

            1. POSITION RECOMMENDATION:
               - Determine a position outlook (Long, Short, or Hold/Neutral)
               - Consider your confidence level (0-100%) in your recommendation
               - Try to make a recommendation that is not "Hold"

            2. TIME HORIZON:
               - Consider whether this is a short-term, medium-term, or long-term outlook
               - Factor in how the outlook might differ across different time horizons

            3. MIXED SIGNALS ASSESSMENT:
               - Identify any contradictory signals in the data
               - Quantify the bullish vs. bearish signals
               - Weigh the strongest evidence for and against your recommendation

            4. MARKET REGIME CONSIDERATION:
               - Consider how your recommendation might change under different market conditions

            Base your assessment ONLY on the tweet data provided, without adding external information.

            REMEMBER: Your ONLY output should be the single word "Long", "Short", or "Hold". No other text.
            """,
                ),
                ("human", "{question}"),
                ("system", "Context tweets:\n\n{context}"),
                (
                    "system",
                    """
            FINAL REMINDER: Respond with ONLY one of these three words: "Long", "Short", or "Hold".
            No other text, explanation, or commentary.
            """,
                ),
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

            FORMAT YOUR RESPONSE USING PROPER MARKDOWN:
            - Format your answer using Markdown syntax for better readability
            - Use **bold** for emphasis on important points
            - Use _italics_ for secondary emphasis
            - Use proper list formatting for enumerated points
            - Use `code` formatting for specific metrics, numbers, or data points
            - Use ```json code blocks with syntax highlighting for any structured data
            - If you quote a tweet, use > blockquote formatting
            - For any statistics or trends, consider using markdown tables

            Make your response visually clear and well-structured for display in a React application.
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
            You are an expert financial analyst skilled in portfolio management. Your task is to transform portfolio manager notes into comprehensive trading theses with specific long and short recommendations. Follow this structured approach:

            1. MARKET VIEW:
               - Synthesize a cohesive view of current market conditions based on the PM notes
               - Identify key themes and drivers affecting markets in the near term
               - Present a clear perspective on the economic and market environment

            2. CORE CONVICTIONS:
               - Develop 4-6 high-conviction market beliefs based on the notes
               - Structure these as clear, actionable statements
               - Include both macro views and sector-specific convictions

            3. STRATEGIC POSITIONING:
               - Recommend specific portfolio allocation changes with percentages
               - Address each explicit question raised in the PM notes
               - Provide clear rationale for each recommendation tied to market view
               - Specify sectors to overweight/underweight with target allocations

            4. SPECIFIC POSITION RECOMMENDATIONS:
               - LONG POSITIONS:
                 * Recommend 3-5 specific securities, sectors, or instruments to establish or increase positions in
                 * Provide clear entry points, target allocations, and expected holding periods
                 * Explain thesis for each recommendation with supporting evidence from market context

               - SHORT POSITIONS:
                 * Recommend 2-4 specific securities, sectors, or instruments to reduce exposure to or establish short positions
                 * Provide clear entry points, position sizing, and risk parameters
                 * Explain the rationale for each recommendation with catalysts and timing considerations

            5. TACTICAL OPPORTUNITIES:
               - Identify 3-5 specific tactical moves to capitalize on current conditions
               - Include both entry and exit recommendations with specific price levels when possible
               - Provide clear timeframe expectations for each opportunity

            6. RISK MANAGEMENT:
               - Outline key risks to the proposed strategy
               - Suggest specific hedging tactics with instruments and allocations
               - Identify explicit triggers that would necessitate strategy revision

            FORMAT YOUR RESPONSE FOR REACT-MARKDOWN COMPATIBILITY:
            - Use ## for section headings (e.g., ## Market View)
            - Use - for bullet points with one space after the dash
            - For nested bullet points, use proper indentation with spaces
            - For tables, use the following format strictly:

              ```
              | Header 1 | Header 2 | Header 3 |
              |----------|----------|----------|
              | Value 1  | Value 2  | Value 3  |
              ```

            - For code blocks or financial data tables, use triple backticks with the language specified:

              ```json
              {{
                "allocation": "45%",
                "sector": "Technology"
              }}
              ```

            IMPORTANT RENDERING CONSIDERATIONS:
            - Ensure all markdown syntax has proper spacing to render correctly
            - Avoid complex nested structures that might break in react-markdown
            - Keep tables simple with consistent column counts and alignments
            - Use bold (**text**) and italic (*text*) formatting sparingly and with proper spacing
            - For numerical data, use `inline code` format for better visibility
            - Your thesis should directly answer each portfolio manager question
            - Use context information to support all recommendations
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
        queries into predefined categories with confidence scores to avoid rigid categorization.

        Returns:
            ChatPromptTemplate: A configured prompt template for query classification
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You are a financial query classifier for a trading assistant system. Your job is to analyze incoming queries and provide confidence scores for how well they match different query types.

            Consider these query types:

            1. TECHNICAL: Queries that mention:
               - Technical indicators (RSI, MACD, Bollinger Bands, moving averages, etc.)
               - Chart patterns (head and shoulders, double top, triangle, flag, etc.)
               - Support/resistance levels, price targets based on technical factors
               - Volume analysis, price action, or candlestick patterns
               - Timeframe analysis (daily, weekly charts)
               - Trend lines, channels, or Fibonacci retracements
               - The phrase "technical analysis" or "technicals"
               - Chart analysis requests

            2. TRADING_THESIS: Queries about:
               - Requests to transform brief notes into comprehensive trading ideas
               - Developing structured investment theses from initial concepts
               - Expanding on portfolio manager notes with supporting analysis
               - The phrase "trading thesis" or "thesis"
               - Creating detailed trade rationales with entry/exit strategies

            3. INVESTMENT: Queries about:
               - Stock symbols or company names
               - Stock market indexes
               - General market analysis, stock fundamentals, or company performance
               - Investment decisions, buy/sell recommendations based on fundamentals
               - Market sectors, trends, economic factors, or market news
               - Portfolio allocation, diversification strategies
               - Company earnings, valuations, or financial metrics

            4. GENERAL: Queries that don't fit the above categories, such as:
               - General questions about the assistant itself
               - Non-financial questions
               - Questions about using the system

            For the given query, provide a JSON object with confidence scores (0-100) for each category.
            Your response should follow this exact format:
            ```json
            {{"technical": X, "trading_thesis": Y, "investment": Z, "general": W}}
            ```

            Where X, Y, Z, and W are numeric values from 0-100 representing your confidence that the query belongs to each category.
            The sum of all confidence scores should be 100.

            IMPORTANT: Analyze the full context of the query. A query may have elements of multiple categories.
            Assign confidence based on the main intent of the query, not just keyword matching.

            Return ONLY the JSON object with no additional text.
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
        based on technical analysis principles, with enhanced debiasing features.

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
            and deliver precise, actionable trading insights that address potential biases.

            Follow this structured approach:

            1. TECHNICAL INDICATOR ANALYSIS:
               - Analyze all mentioned technical indicators including:
                 a) Moving Averages (SMA/EMA) - crossovers, trends, and divergences
                 b) Momentum indicators (RSI, MACD, Stochastic) - overbought/oversold conditions, divergences
                 c) Volatility indicators (Bollinger Bands, ATR) - contractions, expansions, touches
                 d) Volume indicators (OBV, Volume Profile) - confirmation/divergence from price
               - Interpret the current readings and their implications
               - Identify bullish or bearish signals from each indicator
               - Explicitly note when indicators conflict with each other
               - Quantify the weight of evidence (e.g., "3 bullish indicators vs. 2 bearish")

            2. CHART PATTERN RECOGNITION:
               - Identify and analyze any chart patterns mentioned:
                 a) Continuation patterns (flags, pennants, triangles)
                 b) Reversal patterns (head & shoulders, double tops/bottoms)
                 c) Candlestick patterns (engulfing, doji, hammers)
               - Assess pattern completion status and reliability
               - Calculate price targets based on pattern measurements
               - Include reliability statistics for identified patterns (success rate)
               - Consider alternative pattern interpretations

            3. SUPPORT & RESISTANCE ANALYSIS:
               - Identify key price levels from the context
               - Analyze prior reactions at these levels
               - Determine strength of each level based on:
                 a) Number of touches
                 b) Volume at each touch
                 c) Time spent at each level
               - Present both bullish and bearish scenarios for each key level

            4. MULTI-TIMEFRAME CONFIRMATION:
               - Analyze trends across any mentioned timeframes
               - Identify alignment or divergence between timeframes
               - Determine dominant timeframe trend
               - Explicitly note when different timeframes show contradictory signals

            5. PRECISE TRADE RECOMMENDATION:
               - Specify exact entry points with price levels
               - Define specific stop-loss levels with justification
               - Set multiple take-profit targets with:
                 a) Conservative target (highest probability)
                 b) Moderate target
                 c) Ambitious target (if strong trend)
               - Recommend position sizing based on risk metrics
               - Express probability estimates for different outcomes
               - Present both a primary thesis AND an alternative thesis

            6. BIAS MITIGATION ANALYSIS:
               - Identify potential cognitive biases in the technical analysis:
                 a) Recency bias (overweighting recent price action)
                 b) Confirmation bias (seeking patterns that confirm existing views)
                 c) Anchoring bias (fixating on specific price levels)
               - Consider market regime context (trending, range-bound, volatile)
               - Present a counter-analysis that argues against your primary conclusion
               - Assess the quality and reliability of the data in the context
               - Provide separate bull case and bear case scenarios with probabilities

            7. RISK/REWARD ASSESSMENT:
               - Calculate exact risk/reward ratios for each target
               - Determine probability of success based on technical factors
               - Identify specific technical conditions that would invalidate the analysis
               - Quantify uncertainty in each element of your analysis

            FORMAT YOUR RESPONSE USING PROPER MARKDOWN:
            - Begin with a # Heading for the overall technical outlook
            - Use ## headings for main sections and ### for subsections
            - Format any numerical data in `code` style (e.g., `RSI: 78.5`)
            - Use **bold text** for critical insights and key levels
            - Format lists using proper markdown:
              * Use asterisks or hyphens for unordered lists
              * Use proper indentation for nested lists
            - Create tables for comparing different scenarios using markdown table syntax:
              ```
              | Scenario | Probability | Target | Risk/Reward |
              | -------- | ----------- | ------ | ----------- |
              | Bullish  | 60%         | $125   | 1:3         |
              ```
            - For any metrics or calculations, use ```python code blocks with syntax highlighting
            - Use > blockquotes for important warnings or caveats

            Ensure precision with specific price levels, indicator readings, and timeframes.
            Express certainty levels for different aspects of your analysis (e.g., **High confidence: 80-90%**).

            Your response must be well-structured markdown that renders properly in a React application.
            """,
                ),
                ("human", "{question}"),
                ("system", "Here is the relevant technical data from the tweets:\n\n{context}"),
            ]
        )
