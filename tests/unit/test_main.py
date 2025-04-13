"""
Unit tests for main.py application entry point.

This module contains tests for the application entry point defined in app/main.py,
focusing on application startup, shutdown, and the health endpoint.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.main import app, health_check, lifespan


class TestLifespan:
    """Tests for the application lifespan."""

    @pytest.mark.asyncio
    async def test_lifespan_startup(self):
        """Test initialization during application startup."""
        # Create a mock FastAPI app
        mock_app = MagicMock(spec=FastAPI)

        # Mock initialize_services function
        with patch("app.main.initialize_services", new_callable=AsyncMock) as mock_init_services:
            # Create a lifespan manager context and enter it
            async with lifespan(mock_app):
                # Verify that initialize_services was called
                mock_init_services.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_with_error(self):
        """Test lifespan handling with initialization error."""
        # Create a mock FastAPI app
        mock_app = MagicMock(spec=FastAPI)

        # Mock initialize_services to raise an exception
        with patch(
            "app.main.initialize_services", AsyncMock(side_effect=Exception("Test error"))
        ) as mock_init_services:
            # Lifespan should not propagate the error
            async with lifespan(mock_app):
                # Verify initialize_services was called even though it raised an exception
                mock_init_services.assert_called_once()


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self):
        """Test health check endpoint."""
        # Use the test client to call the endpoint
        client = TestClient(app)
        response = client.get("/health")

        # Verify the response
        assert response.status_code == 200

        # Get the JSON response
        data = response.json()

        # Health endpoint response includes more fields than just status
        assert "status" in data
        assert data["status"] == "healthy"
        # Other fields may be present but can vary, so we don't test them specifically

    @pytest.mark.asyncio
    async def test_health_check_function(self):
        """Test the health_check function directly."""
        # Call the function
        result = await health_check()

        # Verify the result
        assert result == {"status": "healthy"}


def test_app_configuration():
    """Test that the FastAPI app is configured with expected settings."""
    # Verify app title and version
    assert app.title == "Tweet RAG Agent API"
    assert app.version == "1.0.0"

    # Verify API documentation endpoints
    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"
    assert app.openapi_url == "/openapi.json"

    # Verify health route is registered
    route_paths = []
    for route in app.routes:
        # Different route types have different attributes
        try:
            route_paths.append(route.path)  # type: ignore
        except AttributeError:
            # Some route types may not have a path attribute
            pass

    # Check that our health endpoint is registered
    assert "/health" in route_paths
