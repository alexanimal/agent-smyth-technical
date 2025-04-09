"""
Module for defining and managing prompts used in the chat system.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class PromptManager:
    """Manages various prompt templates for different use cases."""
    
    @staticmethod
    def get_investment_prompt() -> ChatPromptTemplate:
        """Get the investment-specific prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are a financial advisor. Based on the tweet data provided, respond with ONLY ONE WORD:
            - "Long" if the sentiment suggests buying would be profitable
            - "Short" if the sentiment suggests selling would be wise
            - "Hold" if the sentiment is neutral or unclear
            
            Do not provide any other text, explanation, or context. Just the single word.
            """),
            ("human", "{question}"),
            ("system", "Context tweets:\n\n{context}")
        ])
    
    @staticmethod
    def get_general_prompt() -> ChatPromptTemplate:
        """Get the general-purpose prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful AI assistant who answers questions based on Twitter data. 
            Your goal is to provide informative, factual responses based solely on the provided tweets.
            
            When answering:
            1. Use only the information from the tweet context provided
            2. Be concise and clear
            3. If the information is not in the context, say "I don't have enough information to answer that question"
            4. Do not make up facts or reference sources not in the context
            5. If appropriate, mention how many tweets support your answer
            """),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{question}"),
            ("system", "Here are some relevant tweets that might help you answer:\n\n{context}")
        ])
    
    @staticmethod
    def get_trading_thesis_prompt() -> ChatPromptTemplate:
        """Get a prompt template for transforming PM notes into comprehensive trading theses."""
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert financial analyst tasked with transforming rough notes from portfolio managers (PMs) 
            into comprehensive trading theses. Follow this structured approach:

            1. ANALYSIS PHASE:
               - Identify the core investment idea in the PM's note
               - Extract any mentioned assets, trends, or market conditions
               - Note any explicit or implicit biases in the original thinking

            2. THESIS DEVELOPMENT:
               - Develop a clear, structured trading thesis with:
                 a) Core hypothesis
                 b) Key drivers & catalysts
                 c) Time horizon
                 d) Entry/exit points
                 e) Risk/reward profile

            3. BIAS MITIGATION:
               - Explicitly identify potential confirmation biases in the original note
               - Present counter-arguments to the main thesis
               - Consider alternative scenarios and outcomes
               - Evaluate disconfirming evidence from the tweet data

            4. SUPPORTING EVIDENCE:
               - Use ONLY information from the provided tweet context
               - Cite specific tweets that either support or contradict the thesis
               - Avoid making up facts not present in the context

            5. FORMAT:
               - Present your analysis in a professional, structured format with clear sections
               - Include a "Confirmation Bias Analysis" section specifically addressing potential biases
               - End with a balanced conclusion that presents both bullish and bearish perspectives

            Remember to maintain objectivity and avoid favoring the PM's initial perspective.
            """),
            ("human", """
            Transform this PM note into a comprehensive trading thesis, considering potential confirmation biases:
            
            {question}
            """),
            ("system", "Here is relevant market context from Twitter:\n\n{context}")
        ])

    @staticmethod
    def get_classification_prompt() -> ChatPromptTemplate:
        """Get the prompt template for query classification."""
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are a query classifier. Your job is to determine if a query is about:
            1. Investments, stocks, markets, finance, trading, buy/sell decisions - respond with "investment"
            2. A request to transform a PM (Portfolio Manager) note into a trading thesis - respond with "trading_thesis"
            3. Any other topic - respond with "general"
            
            Respond with ONLY "investment", "trading_thesis", or "general".
            """),
            ("human", "{query}")
        ])
