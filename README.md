# GKE Logs MCP Server

A production-ready Model Context Protocol (MCP) server that exposes Google Kubernetes Engine logs to AI assistants like Claude.

## Features

- **Query GKE logs** by cluster, namespace, pod, container
- **Flexible filtering** by severity level, time range, and search text
- **List discovery** for clusters and namespaces
- **Production-ready** with health checks, security hardening, and Kubernetes manifests
- **Workload Identity** support for secure GCP authentication
- **Built-in caching** for cluster and namespace listings with configurable TTL
- **Timeout handling** with configurable request timeouts
- **Input validation** to prevent filter injection attacks

## Tools Exposed

| Tool | Description |
|------|-------------|
| `list_gke_clusters` | Discover available GKE clusters with logs |
| `list_gke_namespaces` | List namespaces within a cluster |
| `get_gke_logs` | Query logs with filters (namespace, pod, severity, text search) |

## Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud SDK (`gcloud`)
- Docker (for containerised deployment)
- A GCP project with GKE clusters

### Local Development

1. **Clone and install dependencies:**
   ```bash
   cd gke-logs-mcp
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Authenticate with GCP:**
   ```bash
   gcloud auth application-default login
   export GCP_PROJECT_ID="your-project-id"
   ```

3. **Run the server:**
   ```bash
   python -m gke_logs_mcp.server
   ```

### Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gke-logs": {
      "command": "python",
      "args": ["-m", "gke_logs_mcp.server"],
      "env": {
        "GCP_PROJECT_ID": "your-project-id"
      }
    }
  }
}
```

## Testing

The project includes a comprehensive test suite covering the server, client, and healthcheck components.

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=gke_logs_mcp --cov=healthcheck --cov-report=term-missing

# Run specific test file
pytest tests/test_server.py

# Run specific test class
pytest tests/test_server.py::TestGKELogsClient

# Run specific test
pytest tests/test_server.py::TestValidateResourceName::test_valid_simple_name
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures and mocks
├── test_server.py       # Server and client tests
└── test_healthcheck.py  # Health check endpoint tests
```

### Test Categories

- **Unit Tests**: Test individual functions like `validate_resource_name`, `escape_filter_string`
- **Client Tests**: Test `GKELogsClient` filter building, entry formatting, caching
- **Async Tests**: Test async methods like `get_logs`, `list_clusters`, `list_namespaces`
- **MCP Handler Tests**: Test tool registration and tool call handling
- **Healthcheck Tests**: Test HTTP endpoints and subprocess lifecycle

### Writing Tests

Tests use pytest with the following plugins:
- `pytest-asyncio`: For testing async code
- `pytest-mock`: For mocking dependencies
- `pytest-cov`: For coverage reporting

Example test:

```python
import pytest
from gke_logs_mcp.server import validate_resource_name, ValidationError

def test_valid_cluster_name():
    assert validate_resource_name("my-cluster", "cluster_name") == "my-cluster"

def test_invalid_cluster_name():
    with pytest.raises(ValidationError):
        validate_resource_name('invalid"name', "cluster_name")

@pytest.mark.asyncio
async def test_list_clusters(logs_client, mock_cloud_logging_client):
    clusters = await logs_client.list_clusters()
    assert "test-cluster" in clusters
```

## Code Quality

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black gke_logs_mcp/ tests/

# Lint code
ruff check gke_logs_mcp/ tests/

# Type checking
mypy gke_logs_mcp/
```

## Production Deployment

### 1. Set Up GCP Resources

```bash
export GCP_PROJECT_ID="your-project-id"
export GKE_CLUSTER_NAME="your-cluster"

chmod +x scripts/setup-gcp.sh
./scripts/setup-gcp.sh
```

### 2. Build and Push Container

```bash
docker build -t gcr.io/$GCP_PROJECT_ID/gke-logs-mcp:latest .
docker push gcr.io/$GCP_PROJECT_ID/gke-logs-mcp:latest
```

### 3. Deploy to Kubernetes

```bash
sed -i "s/YOUR_PROJECT_ID/$GCP_PROJECT_ID/g" k8s/deployment.yaml
kubectl apply -f k8s/deployment.yaml
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GCP_PROJECT_ID` | GCP project containing your GKE clusters | Auto-detect |
| `DEFAULT_MAX_ENTRIES` | Maximum log entries per query | `500` |
| `DEFAULT_HOURS_BACK` | Default time window (hours) | `1` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `TIMEOUT_SECONDS` | Request timeout in seconds | `60` |
| `CACHE_TTL_SECONDS` | Cache TTL for cluster/namespace lists | `300` |

## Usage Examples

Once connected to Claude:

> "What clusters do we have?"

> "Show me errors in the production cluster's default namespace from the last hour"

> "Search for 'connection refused' in the api-gateway pods"

## Architecture

### Server Components

- **`GKELogsConfig`**: Pydantic model for configuration with validation
- **`GKELogsClient`**: Wrapper around Cloud Logging API with caching and timeout handling
- **`create_server()`**: Factory function for dependency injection and testability
- **MCP Tool Handlers**: Registered via decorators for `list_tools` and `call_tool`

### Key Features

- **Input Validation**: All resource names validated against K8s naming rules
- **Filter Escaping**: Proper escaping of quotes and backslashes in filter strings
- **Caching**: TTL-based caching for cluster and namespace listings
- **Streaming**: Memory-efficient iteration over log entries
- **Timeouts**: Configurable timeouts with `asyncio.wait_for`
- **Retry**: Built-in retry for transient API failures

## Security

- Uses Workload Identity for GCP authentication (no keys in production)
- Requires only `roles/logging.viewer` (read-only)
- Runs as non-root user
- Includes Network Policies for egress/ingress control
- Input validation prevents filter injection attacks

## License

MIT
