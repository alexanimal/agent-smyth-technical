"""
FastAPI application for a RAG Agent using LangChain and OpenAI.
"""
import os
import logging
import asyncio
import sys
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request, Response, Body
from pydantic import BaseModel, Field, validator, field_validator, ValidationInfo
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from uuid import uuid4
from datetime import datetime
import time
from enum import Enum
import sentry_sdk

sentry_sdk.init(
    dsn="https://7cd15bfe524de72b0d00fc47641c4e59@o4509118886248448.ingest.us.sentry.io/4509118890311680",
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for tracing.
    traces_sample_rate=1.0,
    # Set profile_session_sample_rate to 1.0 to profile 100%
    # of profile sessions.
    profile_session_sample_rate=1.0,
    # Set profile_lifecycle to "trace" to automatically
    # run the profiler on when there is an active transaction
    profile_lifecycle="trace",
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path if running as a module
if __name__ != "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

# Local imports - use relative imports
try:
    from src.kb import KnowledgeBaseManager
    from src.chat import ChatHandler
except ImportError:
    # Fallback for direct execution
    from kb import KnowledgeBaseManager
    from chat import ChatHandler

# Load environment variables from .env file
load_dotenv()

# Global variables
knowledge_base = None
chat_handler = None
is_kb_loading = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load knowledge base in background
    asyncio.create_task(load_kb_in_background())
    yield
    # Shutdown: Cleanup if needed
    # No cleanup needed currently

# Initialize FastAPI app with improved documentation
app = FastAPI(
    title="Tweet RAG Agent API",
    description="""
    An API that uses Retrieval Augmented Generation (RAG) to answer user queries based on tweet data.
    
    ## Features
    
    * Real-time chat processing using LLM
    * Source attribution from Twitter data
    * Query classification for specialized responses
    
    ## Authentication
    
    Production environments require API key authentication via the `X-API-Key` header.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "Chat",
            "description": "Chat endpoints for question answering",
        },
        {
            "name": "Health",
            "description": "Health check and monitoring endpoints",
        }
    ],
    lifespan=lifespan,
)

# Add environment attribute
app.environment = os.getenv("ENVIRONMENT", "development")

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourappdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Processing-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# Ensure the OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

# Create models
class QueryType(str, Enum):
    """Types of queries supported by the system."""
    GENERAL = "general"
    INVESTMENT = "investment"
    TRADING = "trading"

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's query text",
        example="What are investors saying about AAPL?"
    )
    num_results: int = Field(
        5,
        ge=1,
        le=100,
        description="Number of sources to retrieve"
    )
    query_type: Optional[QueryType] = Field(
        None,
        description="Override automatic query classification"
    )
    verbose: bool = Field(
        False,
        description="Whether to include detailed source information"
    )
    
    @field_validator('message')
    def message_must_be_meaningful(cls, v, info: ValidationInfo):
        if len(v.strip()) < 3:
            raise ValueError('Message must contain meaningful content')
        return v
    message: str
    num_results: Optional[int] = Field(5, 
        description="Number of sources to retrieve", 
        ge=1, le=20,
        example=5
    )
    context: Optional[Dict[str, Any]] = Field(
        default={},
        description="Optional context information for the query",
        example={"user_location": "US", "platform": "web"}
    )
    model: Optional[str] = Field(
        default="gpt-4o-mini",
        description="The model to use for the query",
        example="gpt-4o-mini"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are the recent trends for AAPL stock?",
                "num_results": 5,
                "context": {"session_id": "user123"}
            }
        }

class ChatResponse(BaseModel):
    request_id: str
    response: str
    sources: List[str] = []
    processing_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "response": "Apple stock has shown positive momentum in recent weeks...",
                "sources": ["https://twitter.com/example/status/123456", "https://twitter.com/finance/status/789012"],
                "processing_time": 1.24,
                "timestamp": "2023-04-28T14:23:15.123456",
                "metadata": {"model_used": "gpt-4", "token_count": 423}
            }
        }

# Initialize the knowledge base manager
kb_manager = KnowledgeBaseManager()

# Dependency for getting the chat handler
async def get_chat_handler():
    """Get the chat handler, ensuring knowledge base is loaded."""
    if not knowledge_base:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not yet initialized. Please try again later."
        )
    return chat_handler

async def load_kb_in_background():
    """Load knowledge base in background without blocking API startup."""
    global knowledge_base, chat_handler, is_kb_loading
    is_kb_loading = True
    
    try:
        logger.info("Starting knowledge base initialization...")
        
        # Verify that the mocks directory exists before proceeding
        mocks_dir = kb_manager.mocks_dir
        if not os.path.exists(mocks_dir):
            logger.warning(f"Mocks directory {mocks_dir} does not exist. Creating it.")
            os.makedirs(mocks_dir, exist_ok=True)
            
        # Check if there are any JSON files in the mocks directory
        import glob
        json_files = glob.glob(os.path.join(mocks_dir, "*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {mocks_dir}")
            
        knowledge_base = await kb_manager.load_or_create_kb()
        chat_handler = ChatHandler(knowledge_base=knowledge_base)
        logger.info("Knowledge base successfully initialized")
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {str(e)}")
        knowledge_base = None
        chat_handler = None
    finally:
        is_kb_loading = False

@app.post(
    "/chat", 
    response_model=ChatResponse,
    summary="Process a chat message using RAG",
    response_description="The AI generated response with sources",
    status_code=200,
    tags=["Chat"]
)
async def chat(
    request: ChatRequest,
    response: Response,
    request_obj: Request,
    x_request_id: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None),
    chat_service: ChatHandler = Depends(get_chat_handler)
) -> ChatResponse:
    """
    Process a chat message and return a response using Retrieval Augmented Generation (RAG).
    
    The endpoint retrieves relevant information from a knowledge base of tweets and
    generates a natural language response.
    
    - **message**: The user's query text (required)
    
    The API requires a valid API key passed in the `X-API-Key` header for production use.
    """
    # Generate request ID if not provided
    request_id = x_request_id or str(uuid4())
    
    # Add request headers to response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-RateLimit-Limit"] = "100"
    response.headers["X-RateLimit-Remaining"] = "99" # In a real app, track this
    response.headers["X-Processing-Time"] = "0"  # Will update later
    
    # Simple API key validation (implement proper auth in production)
    if app.environment == "production" and (not x_api_key or x_api_key != os.getenv("API_KEY")):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    start_time = time.time()
    
    try:
        # Log request info
        logger.info(f"Processing request {request_id} | User-Agent: {user_agent}")
        
        # Process the query
        result = await chat_service.process_query(
            message=request.message,
            k=request.num_results
        )
        
        processing_time = time.time() - start_time
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
        
        # Enhance with optional metadata
        metadata = {
            "model_used": os.getenv("MODEL_NAME", "gpt-4o-mini"),
            "client_info": {
                "user_agent": user_agent,
                "ip": request_obj.client.host if request_obj.client else None
            }
        }
        
        return ChatResponse(
            request_id=request_id,
            response=result["response"],
            sources=result["sources"],
            processing_time=result.get("processing_time"),
            metadata=metadata
        )
    
    except Exception as e:
        logger.exception(f"Error processing chat request {request_id}")
        response.headers["X-Error-Code"] = "PROCESSING_ERROR"
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint to confirm the API is running."""
    return {
        "message": "Tweet RAG API is running. Use /chat endpoint to interact with the assistant.",
        "status": "knowledge_base_loading" if is_kb_loading else 
                  "ready" if knowledge_base else "error"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "knowledge_base_loaded": knowledge_base is not None,
        "is_loading": is_kb_loading
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    
    # Log the request
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    
    # Process the request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add headers to all responses
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time"] = f"{process_time:.3f}"
    
    # Log the response
    logger.info(f"Request {request_id} completed: Status {response.status_code} in {process_time:.3f}s")
    
    return response

# If running this script directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8003, reload=True)
