import time
import logging
from fastapi import Request, Response
from uuid import uuid4

logger = logging.getLogger(__name__)

async def log_requests(request: Request, call_next):
    """Middleware to log incoming requests and add timing/ID headers."""
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    start_time = time.time()

    # Log request start
    logger.info(
        f"rid={request_id} START Request: {request.method} {request.url.path} "
        f"Client={request.client.host if request.client else 'unknown'}"
    )

    try:
        response: Response = await call_next(request)
        process_time = time.time() - start_time

        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{process_time:.4f}" # More precision

        # Log request end
        logger.info(
            f"rid={request_id} END Request: Status={response.status_code} "
            f"Duration={process_time:.4f}s"
        )
        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.exception(
            f"rid={request_id} FAIL Request: Unhandled exception after {process_time:.4f}s"
        )
        # Allow FastAPI's exception handlers to deal with it, but ensure ID is available
        # We can't easily add headers here as the response object might not be formed yet
        raise e # Re-raise the exception 