import asyncio
import glob
import logging
import os
from typing import cast

from fastapi import HTTPException

from .chat import ChatHandler
from .config import MOCKS_DIR_PATH, settings  # Import path and settings

# Use relative imports within src
from .kb import KnowledgeBaseManager

logger = logging.getLogger(__name__)

# Encapsulated application state
app_state = {
    "knowledge_base": None,
    "chat_handler": None,
    "is_kb_loading": False,
}

# Instantiate the manager here, using the path from config
kb_manager = KnowledgeBaseManager(mocks_dir=MOCKS_DIR_PATH)


async def initialize_services():
    """Load knowledge base and initialize chat handler in background."""
    global app_state  # Modify global state dictionary
    if app_state["is_kb_loading"] or app_state["knowledge_base"]:
        logger.info("Initialization already in progress or completed.")
        return

    app_state["is_kb_loading"] = True
    try:
        logger.info("Starting knowledge base initialization...")

        # Verify mocks directory
        if not os.path.exists(MOCKS_DIR_PATH):
            logger.warning(f"Mocks directory {MOCKS_DIR_PATH} does not exist. Creating it.")
            os.makedirs(MOCKS_DIR_PATH, exist_ok=True)

        json_files = glob.glob(os.path.join(MOCKS_DIR_PATH, "*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {MOCKS_DIR_PATH}")

        # Load KB (uses configured mocks path within kb_manager)
        kb = await kb_manager.load_or_create_kb()
        handler = ChatHandler(knowledge_base=kb, model_name=settings.model_name)

        # Update state only on success
        app_state["knowledge_base"] = kb
        app_state["chat_handler"] = handler
        logger.info("Knowledge base and chat handler successfully initialized.")

    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}", exc_info=True)
        # Reset state on error
        app_state["knowledge_base"] = None
        app_state["chat_handler"] = None
    finally:
        app_state["is_kb_loading"] = False


# Dependency function to get the handler
async def get_current_chat_handler() -> ChatHandler:
    """Dependency to get the chat handler, raising 503 if not ready."""
    if not app_state["chat_handler"]:
        status_code = 503  # Service Unavailable
        detail = "Knowledge base is initializing. Please try again shortly."
        if not app_state["is_kb_loading"]:
            # If not loading and no handler, it failed
            detail = "Knowledge base failed to initialize. Check server logs."
        raise HTTPException(status_code=status_code, detail=detail)
    assert app_state["chat_handler"] is not None  # Assure mypy it's not None here
    # Use cast to be explicit about the type for mypy
    return cast(ChatHandler, app_state["chat_handler"])


# Dependency function to get the app state (for status endpoints)
def get_app_state() -> dict:
    """Dependency to get the current application state."""
    return app_state
