import logging
import time
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response

from ..chat import ChatHandler
from ..config import settings  # Import settings

# Use relative imports
from ..schemas import ChatRequest, ChatResponse
from ..services import get_current_chat_handler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",  # Path relative to prefix
    response_model=ChatResponse,
    summary="Process a chat message using RAG",
    response_description="The AI generated response with sources",
    status_code=200,
)
async def handle_chat(
    request_body: ChatRequest,
    response: Response,
    fastapi_request: Request,
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    user_agent: Optional[str] = Header(None, alias="User-Agent"),
    chat_service: ChatHandler = Depends(get_current_chat_handler),
) -> ChatResponse:
    """
    Process a chat message and return a response using Retrieval Augmented Generation (RAG).

    - **message**: The user's query text (required). Length/content validation applies.
    - **num_results**: Number of sources to retrieve (default: 25, max: 250).
    - **context**: Optional contextual information.
    - **model**: Optional override for the LLM model.

    Requires a valid API key via `X-API-Key` header in production.

    Returns a response that includes:
    - Primary analysis based on retrieved sources
    - Alternative viewpoints for balanced perspective (when applicable)
    - Confidence scores for query classification
    - Source attribution for transparency
    """
    request_id = x_request_id or str(uuid4())
    start_time = time.time()

    # Add common response headers (Middleware adds X-Request-ID, X-Processing-Time)
    response.headers["X-RateLimit-Limit"] = "100"  # Example
    response.headers["X-RateLimit-Remaining"] = "99"  # Example

    # API Key Validation
    if settings.environment == "production" and (not x_api_key or x_api_key != settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    try:
        logger.info(
            f"Processing chat request {request_id} for message: '{request_body.message[:50]}...'"
        )

        # Process the query using the injected chat_service
        result = await chat_service.process_query(
            message=request_body.message,
            k=request_body.num_results,
            # Pass other relevant fields if needed, e.g., model override
            # model_override=request_body.model
        )

        processing_time = time.time() - start_time  # Calculate actual time if needed here

        # Prepare metadata
        response_metadata = {
            "model_used": result.get(
                "model_used", settings.model_name
            ),  # Get model used from result if available
            "query_type": result.get("query_type", "unknown"),
            "confidence_scores": result.get("confidence_scores", {}),
            "client_info": {
                "user_agent": user_agent,
                "ip": fastapi_request.client.host if fastapi_request.client else "unknown",
            },
            # Add token counts etc. if returned by process_query
            # "token_input": result.get("token_input"),
            # "token_output": result.get("token_output"),
        }

        return ChatResponse(
            request_id=request_id,
            response=result["response"],
            sources=result["sources"],
            processing_time=result.get("processing_time"),  # Use time from service if available
            alternative_viewpoints=result.get(
                "alternative_viewpoints"
            ),  # Include alternative perspectives
            metadata=response_metadata,
            # Timestamp added automatically by Pydantic
        )

    except ValueError as ve:  # Handle validation errors specifically if needed
        logger.warning(f"Validation error for request {request_id}: {ve}")
        raise HTTPException(status_code=422, detail=f"Invalid input: {ve}")
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like the 503 from get_current_chat_handler)
        raise http_exc
    except Exception as e:
        logger.exception(f"Error processing chat request {request_id}")  # Log full traceback
        response.headers["X-Error-Code"] = "CHAT_PROCESSING_ERROR"
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred while processing your request."
        )


# Potential future endpoint example - keeping it separate
# @router.post("/analyze", ...)
# async def handle_analyze(...): ...
