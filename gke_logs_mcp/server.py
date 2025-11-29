#!/usr/bin/env python3
"""
GKE Logs MCP Server

An MCP server that exposes Google Kubernetes Engine logs via Cloud Logging.
Supports querying logs by cluster, namespace, pod, and container with
flexible filtering options.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from google.cloud import logging as cloud_logging
from google.cloud.logging_v2 import DESCENDING
from google.api_core import exceptions as google_exceptions
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gke-logs-mcp")


class LogSeverity(str, Enum):
    """Log severity levels matching Cloud Logging."""
    DEFAULT = "DEFAULT"
    DEBUG = "DEBUG"
    INFO = "INFO"
    NOTICE = "NOTICE"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    ALERT = "ALERT"
    EMERGENCY = "EMERGENCY"


class GKELogsConfig(BaseModel):
    """Configuration for the GKE Logs MCP server."""
    project_id: str | None = Field(default=None, description="GCP project ID")
    default_max_entries: int = Field(default=500, ge=1, le=10000)
    default_hours_back: int = Field(default=1, ge=1, le=168)  # Max 1 week
    
    @classmethod
    def from_env(cls) -> "GKELogsConfig":
        return cls(
            project_id=os.getenv("GCP_PROJECT_ID"),
            default_max_entries=int(os.getenv("DEFAULT_MAX_ENTRIES", "500")),
            default_hours_back=int(os.getenv("DEFAULT_HOURS_BACK", "1")),
        )


class GKELogsClient:
    """Client for querying GKE logs from Cloud Logging."""
    
    def __init__(self, config: GKELogsConfig):
        self.config = config
        self._client: cloud_logging.Client | None = None
    
    @property
    def client(self) -> cloud_logging.Client:
        if self._client is None:
            self._client = cloud_logging.Client(project=self.config.project_id)
            logger.info(f"Initialized Cloud Logging client for project: {self._client.project}")
        return self._client
    
    def _build_filter(
        self,
        cluster_name: str,
        namespace: str | None = None,
        pod_name: str | None = None,
        container_name: str | None = None,
        severity: LogSeverity | None = None,
        search_text: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> str:
        """Build a Cloud Logging filter string for GKE logs."""
        filters = [
            'resource.type="k8s_container"',
            f'resource.labels.cluster_name="{cluster_name}"',
        ]
        
        if namespace:
            filters.append(f'resource.labels.namespace_name="{namespace}"')
        if pod_name:
            filters.append(f'resource.labels.pod_name=~"{pod_name}"')  # Regex match
        if container_name:
            filters.append(f'resource.labels.container_name="{container_name}"')
        if severity:
            filters.append(f'severity>={severity.value}')
        if search_text:
            # Escape quotes in search text
            escaped = search_text.replace('"', '\\"')
            filters.append(f'textPayload=~"{escaped}" OR jsonPayload.message=~"{escaped}"')
        
        if start_time:
            filters.append(f'timestamp>="{start_time.isoformat()}"')
        if end_time:
            filters.append(f'timestamp<="{end_time.isoformat()}"')
        
        return " AND ".join(filters)
    
    def _format_entry(self, entry: Any) -> dict:
        """Format a log entry for output."""
        payload = entry.payload
        if isinstance(payload, dict):
            message = payload.get("message", str(payload))
        else:
            message = str(payload) if payload else ""
        
        labels = entry.resource.labels
        return {
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
            "severity": entry.severity or "DEFAULT",
            "namespace": labels.get("namespace_name", "unknown"),
            "pod": labels.get("pod_name", "unknown"),
            "container": labels.get("container_name", "unknown"),
            "message": message,
        }
    
    async def get_logs(
        self,
        cluster_name: str,
        namespace: str | None = None,
        pod_name: str | None = None,
        container_name: str | None = None,
        severity: LogSeverity | None = None,
        search_text: str | None = None,
        hours_back: int | None = None,
        max_entries: int | None = None,
    ) -> dict:
        """
        Retrieve GKE logs with the specified filters.
        
        Returns a dict with metadata and log entries.
        """
        hours = hours_back or self.config.default_hours_back
        limit = max_entries or self.config.default_max_entries
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        filter_str = self._build_filter(
            cluster_name=cluster_name,
            namespace=namespace,
            pod_name=pod_name,
            container_name=container_name,
            severity=severity,
            search_text=search_text,
            start_time=start_time,
            end_time=end_time,
        )
        
        logger.debug(f"Executing log query: {filter_str}")
        
        # Run the synchronous API call in a thread pool
        loop = asyncio.get_event_loop()
        entries = await loop.run_in_executor(
            None,
            lambda: list(self.client.list_entries(
                filter_=filter_str,
                order_by=DESCENDING,
                max_results=limit,
            ))
        )
        
        formatted = [self._format_entry(e) for e in entries]
        
        return {
            "metadata": {
                "cluster": cluster_name,
                "namespace": namespace,
                "pod_filter": pod_name,
                "container": container_name,
                "severity_minimum": severity.value if severity else None,
                "search_text": search_text,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours,
                },
                "entries_returned": len(formatted),
                "max_entries": limit,
            },
            "logs": formatted,
        }
    
    async def list_clusters(self) -> list[str]:
        """List available GKE clusters that have logs."""
        filter_str = 'resource.type="k8s_container"'
        
        loop = asyncio.get_event_loop()
        entries = await loop.run_in_executor(
            None,
            lambda: list(self.client.list_entries(
                filter_=filter_str,
                max_results=1000,
            ))
        )
        
        clusters = set()
        for entry in entries:
            cluster = entry.resource.labels.get("cluster_name")
            if cluster:
                clusters.add(cluster)
        
        return sorted(clusters)
    
    async def list_namespaces(self, cluster_name: str) -> list[str]:
        """List namespaces in a cluster that have logs."""
        filter_str = f'resource.type="k8s_container" AND resource.labels.cluster_name="{cluster_name}"'
        
        loop = asyncio.get_event_loop()
        entries = await loop.run_in_executor(
            None,
            lambda: list(self.client.list_entries(
                filter_=filter_str,
                max_results=1000,
            ))
        )
        
        namespaces = set()
        for entry in entries:
            ns = entry.resource.labels.get("namespace_name")
            if ns:
                namespaces.add(ns)
        
        return sorted(namespaces)


# Initialize server and client
app = Server("gke-logs-mcp")
config = GKELogsConfig.from_env()
logs_client = GKELogsClient(config)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="list_gke_clusters",
            description="List GKE clusters that have logs in Cloud Logging. Use this to discover available clusters before querying logs.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="list_gke_namespaces",
            description="List Kubernetes namespaces within a GKE cluster that have logs. Use this to discover namespaces before querying namespace-specific logs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the GKE cluster",
                    },
                },
                "required": ["cluster_name"],
            },
        ),
        Tool(
            name="get_gke_logs",
            description="Retrieve logs from a GKE cluster. Can filter by namespace, pod, container, severity, and search text. Returns logs in reverse chronological order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the GKE cluster (required)",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace to filter by (optional)",
                    },
                    "pod_name": {
                        "type": "string",
                        "description": "Pod name pattern to filter by - supports regex (optional)",
                    },
                    "container_name": {
                        "type": "string",
                        "description": "Container name to filter by (optional)",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["DEFAULT", "DEBUG", "INFO", "NOTICE", "WARNING", "ERROR", "CRITICAL", "ALERT", "EMERGENCY"],
                        "description": "Minimum severity level (optional, returns this level and above)",
                    },
                    "search_text": {
                        "type": "string",
                        "description": "Text to search for in log messages (optional, regex supported)",
                    },
                    "hours_back": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 168,
                        "description": "Number of hours to look back (default: 1, max: 168)",
                    },
                    "max_entries": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10000,
                        "description": "Maximum number of log entries to return (default: 500)",
                    },
                },
                "required": ["cluster_name"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "list_gke_clusters":
            clusters = await logs_client.list_clusters()
            if not clusters:
                content = "No GKE clusters found with logs in this project."
            else:
                content = f"Found {len(clusters)} cluster(s):\n" + "\n".join(f"  - {c}" for c in clusters)
            return [TextContent(type="text", text=content)]
        
        elif name == "list_gke_namespaces":
            cluster = arguments.get("cluster_name")
            if not cluster:
                return [TextContent(type="text", text="Error: cluster_name is required")]
            
            namespaces = await logs_client.list_namespaces(cluster)
            if not namespaces:
                content = f"No namespaces found with logs in cluster '{cluster}'."
            else:
                content = f"Found {len(namespaces)} namespace(s) in '{cluster}':\n" + "\n".join(f"  - {ns}" for ns in namespaces)
            return [TextContent(type="text", text=content)]
        
        elif name == "get_gke_logs":
            cluster = arguments.get("cluster_name")
            if not cluster:
                return [TextContent(type="text", text="Error: cluster_name is required")]
            
            severity = None
            if sev_str := arguments.get("severity"):
                severity = LogSeverity(sev_str)
            
            result = await logs_client.get_logs(
                cluster_name=cluster,
                namespace=arguments.get("namespace"),
                pod_name=arguments.get("pod_name"),
                container_name=arguments.get("container_name"),
                severity=severity,
                search_text=arguments.get("search_text"),
                hours_back=arguments.get("hours_back"),
                max_entries=arguments.get("max_entries"),
            )
            
            # Format output
            meta = result["metadata"]
            logs = result["logs"]
            
            lines = [
                "## Log Query Results",
                f"**Cluster:** {meta['cluster']}",
            ]
            if meta["namespace"]:
                lines.append(f"**Namespace:** {meta['namespace']}")
            if meta["pod_filter"]:
                lines.append(f"**Pod Filter:** {meta['pod_filter']}")
            if meta["severity_minimum"]:
                lines.append(f"**Minimum Severity:** {meta['severity_minimum']}")
            if meta["search_text"]:
                lines.append(f"**Search Text:** {meta['search_text']}")
            
            lines.extend([
                f"**Time Range:** {meta['time_range']['start']} to {meta['time_range']['end']}",
                f"**Entries Returned:** {meta['entries_returned']} (max: {meta['max_entries']})",
                "",
                "---",
                "",
            ])
            
            if not logs:
                lines.append("*No logs found matching the specified criteria.*")
            else:
                for log in logs:
                    ts = log["timestamp"][:19] if log["timestamp"] else "unknown"
                    sev = log["severity"]
                    ns = log["namespace"]
                    pod = log["pod"]
                    container = log["container"]
                    msg = log["message"]
                    
                    lines.append(f"[{ts}] [{sev}] {ns}/{pod}/{container}")
                    lines.append(f"  {msg}")
                    lines.append("")
            
            return [TextContent(type="text", text="\n".join(lines))]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except google_exceptions.PermissionDenied as e:
        logger.error(f"Permission denied: {e}")
        return [TextContent(type="text", text=f"Permission denied. Ensure the service account has 'roles/logging.viewer' on the project.\nError: {e}")]
    except google_exceptions.NotFound as e:
        logger.error(f"Resource not found: {e}")
        return [TextContent(type="text", text=f"Resource not found. Check that the cluster name is correct.\nError: {e}")]
    except Exception as e:
        logger.exception(f"Error executing tool {name}")
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def main():
    """Run the MCP server."""
    logger.info("Starting GKE Logs MCP Server")
    logger.info(f"Project: {config.project_id or 'auto-detect'}")
    logger.info(f"Default max entries: {config.default_max_entries}")
    logger.info(f"Default hours back: {config.default_hours_back}")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
