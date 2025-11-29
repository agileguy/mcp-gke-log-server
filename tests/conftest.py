"""Pytest configuration and shared fixtures."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

from gke_logs_mcp.server import (
    GKELogsConfig,
    GKELogsClient,
    create_server,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    return GKELogsConfig(
        project_id="test-project",
        default_max_entries=100,
        default_hours_back=1,
        timeout_seconds=30,
        cache_ttl_seconds=60,
    )


@pytest.fixture
def config_no_cache():
    """Create a test configuration with caching disabled."""
    return GKELogsConfig(
        project_id="test-project",
        default_max_entries=100,
        default_hours_back=1,
        timeout_seconds=30,
        cache_ttl_seconds=0,
    )


@pytest.fixture
def logs_client(config):
    """Create a GKELogsClient with test config."""
    return GKELogsClient(config)


@pytest.fixture
def mock_log_entry():
    """Create a mock log entry."""
    entry = MagicMock()
    entry.timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    entry.severity = "INFO"
    entry.payload = {"message": "Test log message"}
    entry.resource.labels = {
        "cluster_name": "test-cluster",
        "namespace_name": "default",
        "pod_name": "test-pod-abc123",
        "container_name": "main",
    }
    return entry


@pytest.fixture
def mock_log_entries(mock_log_entry):
    """Create a list of mock log entries."""
    entries = []
    for i in range(5):
        entry = MagicMock()
        entry.timestamp = datetime(2024, 1, 15, 10, 30 + i, 0, tzinfo=timezone.utc)
        entry.severity = "INFO" if i % 2 == 0 else "WARNING"
        entry.payload = {"message": f"Test log message {i}"}
        entry.resource.labels = {
            "cluster_name": "test-cluster",
            "namespace_name": "default" if i < 3 else "kube-system",
            "pod_name": f"test-pod-{i}",
            "container_name": "main",
        }
        entries.append(entry)
    return entries


@pytest.fixture
def mock_cloud_logging_client(mocker, mock_log_entries):
    """Mock the Cloud Logging client."""
    mock_client = MagicMock()
    mock_client.project = "test-project"
    mock_client.list_entries.return_value = iter(mock_log_entries)

    mocker.patch(
        "gke_logs_mcp.server.cloud_logging.Client",
        return_value=mock_client
    )
    return mock_client


@pytest.fixture
def server_and_client(config, mock_cloud_logging_client):
    """Create a server and client with mocked dependencies."""
    app, client = create_server(config)
    return app, client
