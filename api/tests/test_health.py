"""Tests for the health check endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest.fixture
async def client():
    """Provide an async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_status_ok(self, client):
        """GET /api/v1/health should return status ok."""
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_version(self, client):
        """GET /api/v1/health should include version."""
        response = await client.get("/api/v1/health")
        data = response.json()
        assert data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_health_model_loaded(self, client):
        """GET /api/v1/health should include boolean model_loaded."""
        response = await client.get("/api/v1/health")
        data = response.json()
        assert isinstance(data["model_loaded"], bool)

    @pytest.mark.asyncio
    async def test_health_uptime_positive(self, client):
        """GET /api/v1/health should include positive uptime_seconds."""
        response = await client.get("/api/v1/health")
        data = response.json()
        assert isinstance(data["uptime_seconds"], float)
        assert data["uptime_seconds"] > 0

    @pytest.mark.asyncio
    async def test_health_response_shape(self, client):
        """GET /api/v1/health body should contain all four expected keys."""
        response = await client.get("/api/v1/health")
        data = response.json()
        assert set(data.keys()) == {
            "status",
            "version",
            "model_loaded",
            "uptime_seconds",
        }


class TestPing:
    @pytest.mark.asyncio
    async def test_ping_status(self, client):
        """GET /api/v1/health/ping should return 200."""
        response = await client.get("/api/v1/health/ping")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_ping_body(self, client):
        """GET /api/v1/health/ping should return {"ping": "pong"}."""
        response = await client.get("/api/v1/health/ping")
        assert response.json() == {"ping": "pong"}
