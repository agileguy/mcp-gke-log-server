"""Tests for the health check server."""

import pytest
import subprocess
from unittest.mock import MagicMock, patch, AsyncMock

from starlette.testclient import TestClient


class TestHealthCheckEndpoints:
    """Tests for health check HTTP endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked subprocess."""
        with patch("healthcheck.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process

            from healthcheck import app
            with TestClient(app) as client:
                yield client, mock_process

    def test_health_endpoint(self, client):
        """Health endpoint should return 200."""
        test_client, _ = client
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_healthz_endpoint(self, client):
        """Healthz endpoint (K8s convention) should return 200."""
        test_client, _ = client
        response = test_client.get("/healthz")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_ready_endpoint_when_running(self, client):
        """Ready endpoint should return 200 when process is running."""
        test_client, mock_process = client
        mock_process.poll.return_value = None

        response = test_client.get("/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_readyz_endpoint_when_running(self, client):
        """Readyz endpoint (K8s convention) should return 200 when running."""
        test_client, mock_process = client
        mock_process.poll.return_value = None

        response = test_client.get("/readyz")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_ready_endpoint_when_crashed(self, client):
        """Ready endpoint should return 503 when process has crashed."""
        test_client, mock_process = client
        mock_process.poll.return_value = 1  # Process exited with code 1

        response = test_client.get("/ready")

        assert response.status_code == 503
        assert response.json()["status"] == "crashed"
        assert response.json()["returncode"] == 1


class TestHealthCheckStartupShutdown:
    """Tests for startup and shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_startup_spawns_mcp_process(self):
        """Startup should spawn the MCP server process."""
        with patch("healthcheck.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            import healthcheck
            healthcheck.mcp_process = None

            await healthcheck.startup()

            mock_popen.assert_called_once()
            assert healthcheck.mcp_process is not None
            assert healthcheck.mcp_process.pid == 12345

    @pytest.mark.asyncio
    async def test_shutdown_terminates_process(self):
        """Shutdown should terminate the MCP process gracefully."""
        mock_process = MagicMock()
        mock_process.wait.return_value = 0

        import healthcheck
        healthcheck.mcp_process = mock_process

        await healthcheck.shutdown()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_shutdown_kills_on_timeout(self):
        """Shutdown should kill process if terminate times out."""
        mock_process = MagicMock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=10)

        import healthcheck
        healthcheck.mcp_process = mock_process

        await healthcheck.shutdown()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_no_process(self):
        """Shutdown should handle case where process is None."""
        import healthcheck
        healthcheck.mcp_process = None

        # Should not raise
        await healthcheck.shutdown()


class TestHealthCheckNotStarted:
    """Tests for when MCP process hasn't started."""

    def test_ready_returns_503_when_not_started(self):
        """Ready endpoint should return 503 if process not started."""
        import healthcheck

        # Save original and set to None
        original = healthcheck.mcp_process
        healthcheck.mcp_process = None

        try:
            # Create app without startup
            from starlette.testclient import TestClient
            from starlette.applications import Starlette
            from starlette.routing import Route

            # Create a minimal app without startup hook
            test_app = Starlette(
                routes=[
                    Route("/ready", healthcheck.ready, methods=["GET"]),
                ]
            )

            with TestClient(test_app, raise_server_exceptions=False) as client:
                response = client.get("/ready")

                assert response.status_code == 503
                assert response.json()["status"] == "not_started"
        finally:
            healthcheck.mcp_process = original
