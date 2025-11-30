"""Tests for the GKE Logs MCP Server."""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

from gke_logs_mcp.server import (
    GKELogsConfig,
    GKELogsClient,
    LogSeverity,
    ValidationError,
    validate_resource_name,
    escape_filter_string,
    timed_lru_cache,
    create_server,
)


class TestValidateResourceName:
    """Tests for resource name validation."""

    def test_valid_simple_name(self):
        """Valid simple names should pass."""
        assert validate_resource_name("my-cluster", "cluster_name") == "my-cluster"

    def test_valid_name_with_dots(self):
        """Names with dots should be valid."""
        assert validate_resource_name("my.cluster.name", "cluster_name") == "my.cluster.name"

    def test_valid_name_with_underscores(self):
        """Names with underscores should be valid."""
        assert validate_resource_name("my_cluster_name", "cluster_name") == "my_cluster_name"

    def test_valid_alphanumeric(self):
        """Alphanumeric names should be valid."""
        assert validate_resource_name("cluster123", "cluster_name") == "cluster123"

    def test_empty_name_raises(self):
        """Empty names should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_resource_name("", "cluster_name")

    def test_name_too_long_raises(self):
        """Names exceeding max length should raise ValidationError."""
        long_name = "a" * 254
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            validate_resource_name(long_name, "cluster_name")

    def test_invalid_start_char_raises(self):
        """Names starting with non-alphanumeric should raise ValidationError."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_resource_name("-invalid", "cluster_name")

    def test_invalid_chars_raises(self):
        """Names with invalid characters should raise ValidationError."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_resource_name('cluster"injection', "cluster_name")

    def test_spaces_raises(self):
        """Names with spaces should raise ValidationError."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_resource_name("my cluster", "cluster_name")


class TestEscapeFilterString:
    """Tests for filter string escaping."""

    def test_escape_double_quotes(self):
        """Double quotes should be escaped."""
        assert escape_filter_string('test"value') == 'test\\"value'

    def test_escape_backslashes(self):
        """Backslashes should be escaped."""
        assert escape_filter_string("test\\value") == "test\\\\value"

    def test_escape_both(self):
        """Both quotes and backslashes should be escaped."""
        assert escape_filter_string('test\\"value') == 'test\\\\\\"value'

    def test_no_escaping_needed(self):
        """Strings without special chars should be unchanged."""
        assert escape_filter_string("simple-value") == "simple-value"

    def test_regex_pattern(self):
        """Regex patterns with backslashes should be properly escaped."""
        assert escape_filter_string("\\d+") == "\\\\d+"


class TestTimedLruCache:
    """Tests for the timed LRU cache decorator."""

    def test_cache_returns_cached_value(self):
        """Cache should return cached values within TTL."""
        call_count = 0

        @timed_lru_cache(seconds=60)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_func(5)
        result2 = expensive_func(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1

    def test_cache_different_args(self):
        """Cache should store different values for different args."""
        call_count = 0

        @timed_lru_cache(seconds=60)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_func(5)
        result2 = expensive_func(10)

        assert result1 == 10
        assert result2 == 20
        assert call_count == 2

    def test_cache_clear(self):
        """Cache should be clearable."""
        call_count = 0

        @timed_lru_cache(seconds=60)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive_func(5)
        expensive_func.cache_clear()
        expensive_func(5)

        assert call_count == 2


class TestGKELogsConfig:
    """Tests for configuration handling."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = GKELogsConfig()
        assert config.project_id is None
        assert config.default_max_entries == 500
        assert config.default_hours_back == 1

    def test_custom_values(self):
        """Config should accept custom values."""
        config = GKELogsConfig(
            project_id="my-project",
            default_max_entries=1000,
            default_hours_back=24,
            timeout_seconds=120,
            cache_ttl_seconds=600,
        )
        assert config.project_id == "my-project"
        assert config.default_max_entries == 1000
        assert config.default_hours_back == 24
        assert config.timeout_seconds == 120
        assert config.cache_ttl_seconds == 600

    def test_validation_max_entries_bounds(self):
        """Max entries should be within valid range."""
        with pytest.raises(ValueError):
            GKELogsConfig(default_max_entries=0)

        with pytest.raises(ValueError):
            GKELogsConfig(default_max_entries=10001)

    def test_validation_hours_back_bounds(self):
        """Hours back should be within valid range."""
        with pytest.raises(ValueError):
            GKELogsConfig(default_hours_back=0)

        with pytest.raises(ValueError):
            GKELogsConfig(default_hours_back=169)

    def test_from_env(self, monkeypatch):
        """Config should load from environment variables."""
        monkeypatch.setenv("GCP_PROJECT_ID", "env-project")
        monkeypatch.setenv("DEFAULT_MAX_ENTRIES", "200")
        monkeypatch.setenv("DEFAULT_HOURS_BACK", "12")
        monkeypatch.setenv("TIMEOUT_SECONDS", "90")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "120")

        config = GKELogsConfig.from_env()

        assert config.project_id == "env-project"
        assert config.default_max_entries == 200
        assert config.default_hours_back == 12
        assert config.timeout_seconds == 90
        assert config.cache_ttl_seconds == 120


class TestGKELogsClient:
    """Tests for the GKE Logs Client."""

    def test_build_filter_basic(self, logs_client):
        """Build filter with only cluster name."""
        filter_str = logs_client._build_filter(cluster_name="my-cluster")

        assert 'resource.type="k8s_container"' in filter_str
        assert 'resource.labels.cluster_name="my-cluster"' in filter_str

    def test_build_filter_with_namespace(self, logs_client):
        """Build filter with namespace."""
        filter_str = logs_client._build_filter(
            cluster_name="my-cluster",
            namespace="default"
        )

        assert 'resource.labels.namespace_name="default"' in filter_str

    def test_build_filter_with_pod_regex(self, logs_client):
        """Build filter with pod name regex."""
        filter_str = logs_client._build_filter(
            cluster_name="my-cluster",
            pod_name="api-.*"
        )

        assert 'resource.labels.pod_name=~"api-.*"' in filter_str

    def test_build_filter_with_severity(self, logs_client):
        """Build filter with severity level."""
        filter_str = logs_client._build_filter(
            cluster_name="my-cluster",
            severity=LogSeverity.WARNING
        )

        assert "severity>=WARNING" in filter_str

    def test_build_filter_with_search_text(self, logs_client):
        """Build filter with search text."""
        filter_str = logs_client._build_filter(
            cluster_name="my-cluster",
            search_text="error"
        )

        assert 'textPayload=~"error"' in filter_str
        assert 'jsonPayload.message=~"error"' in filter_str

    def test_build_filter_escapes_search_text(self, logs_client):
        """Search text should be escaped."""
        filter_str = logs_client._build_filter(
            cluster_name="my-cluster",
            search_text='test"value'
        )

        assert 'test\\"value' in filter_str

    def test_build_filter_with_time_range(self, logs_client):
        """Build filter with time range."""
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

        filter_str = logs_client._build_filter(
            cluster_name="my-cluster",
            start_time=start,
            end_time=end
        )

        assert 'timestamp>="2024-01-01T00:00:00+00:00"' in filter_str
        assert 'timestamp<="2024-01-01T01:00:00+00:00"' in filter_str

    def test_build_filter_validates_cluster_name(self, logs_client):
        """Invalid cluster name should raise ValidationError."""
        with pytest.raises(ValidationError):
            logs_client._build_filter(cluster_name='invalid"name')

    def test_build_filter_validates_namespace(self, logs_client):
        """Invalid namespace should raise ValidationError."""
        with pytest.raises(ValidationError):
            logs_client._build_filter(
                cluster_name="my-cluster",
                namespace='invalid"namespace'
            )

    def test_format_entry_dict_payload(self, logs_client, mock_log_entry):
        """Format entry with dict payload."""
        formatted = logs_client._format_entry(mock_log_entry)

        assert formatted["timestamp"] == "2024-01-15T10:30:00+00:00"
        assert formatted["severity"] == "INFO"
        assert formatted["namespace"] == "default"
        assert formatted["pod"] == "test-pod-abc123"
        assert formatted["container"] == "main"
        assert formatted["message"] == "Test log message"

    def test_format_entry_string_payload(self, logs_client):
        """Format entry with string payload."""
        entry = MagicMock()
        entry.timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        entry.severity = "ERROR"
        entry.payload = "Simple string message"
        entry.resource.labels = {
            "cluster_name": "test-cluster",
            "namespace_name": "default",
            "pod_name": "test-pod",
            "container_name": "main",
        }

        formatted = logs_client._format_entry(entry)

        assert formatted["message"] == "Simple string message"

    def test_format_entry_empty_payload(self, logs_client):
        """Format entry with empty payload."""
        entry = MagicMock()
        entry.timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        entry.severity = "INFO"
        entry.payload = None
        entry.resource.labels = {
            "cluster_name": "test-cluster",
            "namespace_name": "default",
            "pod_name": "test-pod",
            "container_name": "main",
        }

        formatted = logs_client._format_entry(entry)

        assert formatted["message"] == ""

    def test_clear_cache(self, logs_client):
        """Clear cache should reset all cached data."""
        logs_client._clusters_cache = (time.monotonic(), ["cluster1"])
        logs_client._namespaces_cache["cluster1"] = (time.monotonic(), ["ns1"])

        logs_client.clear_cache()

        assert logs_client._clusters_cache is None
        assert len(logs_client._namespaces_cache) == 0

    def test_cache_validity_check(self, logs_client):
        """Cache validity should be based on TTL."""
        current_time = time.monotonic()

        # Valid cache (just created)
        assert logs_client._is_cache_valid(current_time)

        # Invalid cache (too old)
        old_time = current_time - logs_client._cache_ttl - 1
        assert not logs_client._is_cache_valid(old_time)

    def test_cache_disabled_when_ttl_zero(self, config_no_cache):
        """Cache should be disabled when TTL is 0."""
        client = GKELogsClient(config_no_cache)
        assert not client._is_cache_valid(time.monotonic())


class TestGKELogsClientAsync:
    """Async tests for the GKE Logs Client."""

    @pytest.mark.asyncio
    async def test_get_logs(self, logs_client, mock_cloud_logging_client, mock_log_entries):
        """Get logs should return formatted entries."""
        result = await logs_client.get_logs(cluster_name="test-cluster")

        assert "metadata" in result
        assert "logs" in result
        assert result["metadata"]["cluster"] == "test-cluster"
        assert len(result["logs"]) == 5

    @pytest.mark.asyncio
    async def test_get_logs_with_filters(self, logs_client, mock_cloud_logging_client):
        """Get logs should apply filters."""
        result = await logs_client.get_logs(
            cluster_name="test-cluster",
            namespace="default",
            severity=LogSeverity.WARNING,
            hours_back=24,
            max_entries=50,
        )

        assert result["metadata"]["namespace"] == "default"
        assert result["metadata"]["severity_minimum"] == "WARNING"
        assert result["metadata"]["time_range"]["hours"] == 24
        assert result["metadata"]["max_entries"] == 50

    @pytest.mark.asyncio
    async def test_list_clusters(self, logs_client, mock_cloud_logging_client):
        """List clusters should return unique cluster names."""
        clusters = await logs_client.list_clusters()

        assert "test-cluster" in clusters
        # Should be cached now
        assert logs_client._clusters_cache is not None

    @pytest.mark.asyncio
    async def test_list_clusters_uses_cache(self, logs_client, mock_cloud_logging_client, mocker):
        """List clusters should use cache on subsequent calls."""
        # First call
        await logs_client.list_clusters()

        # Clear the mock call count on the container client
        logs_client._container_client.list_clusters.reset_mock()

        # Second call should use cache
        await logs_client.list_clusters()

        logs_client._container_client.list_clusters.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_namespaces(self, logs_client, mock_cloud_logging_client):
        """List namespaces should return unique namespace names."""
        namespaces = await logs_client.list_namespaces("test-cluster")

        assert "default" in namespaces
        assert "kube-system" in namespaces

    @pytest.mark.asyncio
    async def test_list_namespaces_validates_cluster(self, logs_client, mock_cloud_logging_client):
        """List namespaces should validate cluster name."""
        with pytest.raises(ValidationError):
            await logs_client.list_namespaces('invalid"cluster')


class TestCreateServer:
    """Tests for the server factory function."""

    def test_create_server_with_config(self, config, mock_cloud_logging_client):
        """Create server should accept custom config."""
        app, client = create_server(config)

        assert app is not None
        assert client is not None
        assert client.config.project_id == "test-project"

    def test_create_server_default_config(self, mock_cloud_logging_client, monkeypatch):
        """Create server should use default config from env."""
        monkeypatch.setenv("GCP_PROJECT_ID", "env-project")

        app, client = create_server()

        assert client.config.project_id == "env-project"


class TestMCPToolHandlers:
    """Tests for MCP tool handlers."""

    @pytest.mark.asyncio
    async def test_list_tools(self, server_and_client):
        """List tools should return all available tools."""
        app, _ = server_and_client

        # Get the list_tools handler
        tools = await app._tool_manager.list_tools()

        tool_names = [t.name for t in tools]
        assert "list_gke_clusters" in tool_names
        assert "list_gke_namespaces" in tool_names
        assert "get_gke_logs" in tool_names

    @pytest.mark.asyncio
    async def test_call_tool_list_clusters(self, server_and_client, mock_log_entries):
        """Call list_gke_clusters tool."""
        app, _ = server_and_client

        result = await app._tool_manager.call_tool("list_gke_clusters", {})

        assert len(result) == 1
        assert "cluster" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_list_namespaces(self, server_and_client, mock_log_entries):
        """Call list_gke_namespaces tool."""
        app, _ = server_and_client

        result = await app._tool_manager.call_tool(
            "list_gke_namespaces",
            {"cluster_name": "test-cluster"}
        )

        assert len(result) == 1
        assert "namespace" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_list_namespaces_missing_cluster(self, server_and_client):
        """Call list_gke_namespaces without cluster_name should error."""
        app, _ = server_and_client

        result = await app._tool_manager.call_tool("list_gke_namespaces", {})

        assert "error" in result[0].text.lower()
        assert "cluster_name is required" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_get_logs(self, server_and_client, mock_log_entries):
        """Call get_gke_logs tool."""
        app, _ = server_and_client

        result = await app._tool_manager.call_tool(
            "get_gke_logs",
            {"cluster_name": "test-cluster"}
        )

        assert len(result) == 1
        assert "Log Query Results" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_get_logs_with_filters(self, server_and_client, mock_log_entries):
        """Call get_gke_logs with filters."""
        app, _ = server_and_client

        result = await app._tool_manager.call_tool(
            "get_gke_logs",
            {
                "cluster_name": "test-cluster",
                "namespace": "default",
                "severity": "WARNING",
                "hours_back": 24,
                "max_entries": 50,
            }
        )

        assert len(result) == 1
        text = result[0].text
        assert "Namespace:** default" in text
        assert "Minimum Severity:** WARNING" in text

    @pytest.mark.asyncio
    async def test_call_tool_get_logs_missing_cluster(self, server_and_client):
        """Call get_gke_logs without cluster_name should error."""
        app, _ = server_and_client

        result = await app._tool_manager.call_tool("get_gke_logs", {})

        assert "error" in result[0].text.lower()
        assert "cluster_name is required" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_validation_error(self, server_and_client):
        """Call tool with invalid input should return validation error."""
        app, _ = server_and_client

        result = await app._tool_manager.call_tool(
            "get_gke_logs",
            {"cluster_name": 'invalid"cluster'}
        )

        assert "validation error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool(self, server_and_client):
        """Call unknown tool should return error."""
        app, _ = server_and_client

        result = await app._tool_manager.call_tool("unknown_tool", {})

        assert "unknown tool" in result[0].text.lower()


class TestLogSeverity:
    """Tests for LogSeverity enum."""

    def test_severity_values(self):
        """All severity values should be strings."""
        for severity in LogSeverity:
            assert isinstance(severity.value, str)

    def test_severity_from_string(self):
        """Severity should be creatable from string."""
        severity = LogSeverity("WARNING")
        assert severity == LogSeverity.WARNING

    def test_invalid_severity(self):
        """Invalid severity string should raise ValueError."""
        with pytest.raises(ValueError):
            LogSeverity("INVALID")
