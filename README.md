# GKE Logs MCP Server

A production-ready Model Context Protocol (MCP) server that exposes Google Kubernetes Engine logs to AI assistants like Claude.

## Features

- **Query GKE logs** by cluster, namespace, pod, container
- **Flexible filtering** by severity level, time range, and search text
- **List discovery** for clusters and namespaces
- **Production-ready** with health checks, security hardening, and Kubernetes manifests
- **Workload Identity** support for secure GCP authentication

## Tools Exposed

| Tool | Description |
|------|-------------|
| `list_gke_clusters` | Discover available GKE clusters with logs |
| `list_gke_namespaces` | List namespaces within a cluster |
| `get_gke_logs` | Query logs with filters (namespace, pod, severity, text search) |

## Quick Start

### Prerequisites

- Python 3.12+
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

## Usage Examples

Once connected to Claude:

> "What clusters do we have?"

> "Show me errors in the production cluster's default namespace from the last hour"

> "Search for 'connection refused' in the api-gateway pods"

## Security

- Uses Workload Identity for GCP authentication (no keys in production)
- Requires only `roles/logging.viewer` (read-only)
- Runs as non-root user
- Includes Network Policies for egress/ingress control

## License

MIT
