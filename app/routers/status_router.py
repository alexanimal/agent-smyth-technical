from fastapi import APIRouter, Depends

# Use relative imports
from ..schemas import RootResponse, HealthStatus
from ..services import get_app_state

router = APIRouter(tags=["Health"])

@router.get("/", response_model=RootResponse)
async def root(app_state: dict = Depends(get_app_state)):
    """Root endpoint confirming API status."""
    status = "knowledge_base_loading" if app_state["is_kb_loading"] else \
             "ready" if app_state["chat_handler"] else "error"
    return {
        "message": "Tweet RAG API is running.",
        "status": status
    }

@router.get("/health", response_model=HealthStatus)
async def health_check(app_state: dict = Depends(get_app_state)):
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "knowledge_base_loaded": app_state["chat_handler"] is not None,
        "is_loading": app_state["is_kb_loading"]
    } 