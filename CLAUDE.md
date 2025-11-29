# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Model Context Protocol (MCP) server that exposes Google Kubernetes Engine logs to AI assistants. It queries GKE container logs from Cloud Logging and provides three tools: `list_gke_clusters`, `list_gke_namespaces`, and `get_gke_logs`.

## Development Commands

```bash
# Setup virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the server locally (requires GCP auth)
gcloud auth application-default login
export GCP_PROJECT_ID="your-project-id"
python -m gke_logs_mcp.server

# Run with Docker Compose
docker-compose up gke-logs-mcp

# Dev mode with live reload (mounts local code)
docker-compose --profile dev up gke-logs-mcp-dev

# Build container
docker build -t gcr.io/$GCP_PROJECT_ID/gke-logs-mcp:latest .

# Linting and formatting (dev dependencies)
pip install -e ".[dev]"
black gke_logs_mcp/
ruff check gke_logs_mcp/
mypy gke_logs_mcp/

# Run tests
pytest

# Run tests with coverage
pytest --cov=gke_logs_mcp --cov=healthcheck --cov-report=term-missing
```

## Architecture

**Single-file MCP server**: All server logic is in `gke_logs_mcp/server.py`:
- `GKELogsConfig`: Pydantic model for configuration from environment variables
- `GKELogsClient`: Wrapper around `google-cloud-logging` that builds filter queries and formats log entries
- MCP tool handlers registered via `@app.list_tools()` and `@app.call_tool()` decorators
- Server runs on stdio transport using `mcp.server.stdio.stdio_server`

**Production deployment**: Uses Kubernetes Workload Identity for GCP auth (no service account keys). The `healthcheck.py` runs a Starlette HTTP server on port 8080 that spawns the MCP server as a subprocess and provides `/healthz` and `/readyz` endpoints.

**Cloud Logging filters**: Logs are queried with `resource.type="k8s_container"` and support filtering by cluster, namespace, pod (regex), container, severity level, and text search (regex in `textPayload` or `jsonPayload.message`).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | auto-detect | GCP project ID |
| `DEFAULT_MAX_ENTRIES` | 500 | Max log entries per query |
| `DEFAULT_HOURS_BACK` | 1 | Default time window in hours |
| `LOG_LEVEL` | INFO | Python logging level |
| `TIMEOUT_SECONDS` | 60 | Request timeout in seconds |
| `CACHE_TTL_SECONDS` | 300 | Cache TTL for cluster/namespace listings |

## GCP Setup

Run `scripts/setup-gcp.sh` with `GCP_PROJECT_ID` set. It creates a service account with `roles/logging.viewer` and optionally configures Workload Identity if `GKE_CLUSTER_NAME` is provided.
