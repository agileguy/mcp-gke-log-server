#!/usr/bin/env python3
"""
GKE Logs MCP Server

An MCP server that exposes Google Kubernetes Engine logs via Cloud Logging.
Supports querying logs by cluster, namespace, pod, and container with
flexible filtering options.
"""

import asyncio
import functools
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterator

from google.cloud import logging as cloud_logging
from google.cloud import container_v1
from google.cloud.logging_v2 import DESCENDING
from google.cloud.logging_v2.entries import LogEntry
from google.api_core import exceptions as google_exceptions
from google.api_core.retry import Retry
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gke-logs-mcp")

# Constants
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_CACHE_TTL_SECONDS = 300  # 5 minutes
RESOURCE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
MAX_RESOURCE_NAME_LENGTH = 253  # K8s max resource name length


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


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_resource_name(name: str, field_name: str) -> str:
    """
    Validate a Kubernetes resource name.

    Raises ValidationError if the name is invalid.
    """
    if not name:
        raise ValidationError(f"{field_name} cannot be empty")

    if len(name) > MAX_RESOURCE_NAME_LENGTH:
        raise ValidationError(
            f"{field_name} exceeds maximum length of {MAX_RESOURCE_NAME_LENGTH} characters"
        )

    if not RESOURCE_NAME_PATTERN.match(name):
        raise ValidationError(
            f"{field_name} contains invalid characters. "
            "Must start with alphanumeric and contain only alphanumerics, dots, hyphens, and underscores."
        )

    return name


def escape_filter_string(value: str) -> str:
    """
    Escape a string for use in Cloud Logging filter expressions.

    Escapes backslashes and double quotes.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def timed_lru_cache(seconds: int, maxsize: int = 128):
    """LRU cache with time-based expiration."""
    def decorator(func):
        func = functools.lru_cache(maxsize=maxsize)(func)
        func.expiration = time.monotonic() + seconds

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if time.monotonic() > func.expiration:
                func.cache_clear()
                func.expiration = time.monotonic() + seconds
            return func(*args, **kwargs)

        wrapper.cache_clear = func.cache_clear
        wrapper.cache_info = func.cache_info
        return wrapper
    return decorator


class GKELogsConfig(BaseModel):
    """Configuration for the GKE Logs MCP server."""
    project_id: str | None = Field(default=None, description="GCP project ID")
    default_max_entries: int = Field(default=500, ge=1, le=10000)
    default_hours_back: int = Field(default=1, ge=1, le=168)  # Max 1 week
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=300)
    cache_ttl_seconds: int = Field(default=DEFAULT_CACHE_TTL_SECONDS, ge=0, le=3600)

    @field_validator('default_max_entries', 'default_hours_back', 'timeout_seconds', 'cache_ttl_seconds', mode='before')
    @classmethod
    def parse_int_env(cls, v: Any) -> int:
        if isinstance(v, str):
            return int(v)
        return v

    @classmethod
    def from_env(cls) -> "GKELogsConfig":
        return cls(
            project_id=os.getenv("GCP_PROJECT_ID"),
            default_max_entries=os.getenv("DEFAULT_MAX_ENTRIES", "500"),
            default_hours_back=os.getenv("DEFAULT_HOURS_BACK", "1"),
            timeout_seconds=os.getenv("TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)),
            cache_ttl_seconds=os.getenv("CACHE_TTL_SECONDS", str(DEFAULT_CACHE_TTL_SECONDS)),
        )


class GKELogsClient:
    """Client for querying GKE logs from Cloud Logging."""

    def __init__(self, config: GKELogsConfig):
        self.config = config
        self._client: cloud_logging.Client | None = None
        self._container_client: container_v1.ClusterManagerClient | None = None
        self._cache_ttl = config.cache_ttl_seconds
        self._clusters_cache: tuple[float, list[str]] | None = None
        self._namespaces_cache: dict[str, tuple[float, list[str]]] = {}

    @property
    def client(self) -> cloud_logging.Client:
        if self._client is None:
            self._client = cloud_logging.Client(project=self.config.project_id)
            logger.info(f"Initialized Cloud Logging client for project: {self._client.project}")
        return self._client

    @property
    def container_client(self) -> container_v1.ClusterManagerClient:
        if self._container_client is None:
            self._container_client = container_v1.ClusterManagerClient()
            logger.info("Initialized GKE Cluster Manager client")
        return self._container_client

    def _is_cache_valid(self, cache_time: float) -> bool:
        """Check if a cached value is still valid."""
        if self._cache_ttl <= 0:
            return False
        return (time.monotonic() - cache_time) < self._cache_ttl

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._clusters_cache = None
        self._namespaces_cache.clear()
        logger.debug("Cache cleared")

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
        # Validate and escape cluster name
        validate_resource_name(cluster_name, "cluster_name")

        filters = [
            'resource.type="k8s_container"',
            f'resource.labels.cluster_name="{escape_filter_string(cluster_name)}"',
        ]

        if namespace:
            validate_resource_name(namespace, "namespace")
            filters.append(f'resource.labels.namespace_name="{escape_filter_string(namespace)}"')

        if pod_name:
            # Pod name supports regex, so we only escape quotes/backslashes
            escaped_pod = escape_filter_string(pod_name)
            filters.append(f'resource.labels.pod_name=~"{escaped_pod}"')

        if container_name:
            validate_resource_name(container_name, "container_name")
            filters.append(f'resource.labels.container_name="{escape_filter_string(container_name)}"')

        if severity:
            filters.append(f'severity>={severity.value}')

        if search_text:
            escaped = escape_filter_string(search_text)
            filters.append(f'textPayload=~"{escaped}" OR jsonPayload.message=~"{escaped}"')

        if start_time:
            filters.append(f'timestamp>="{start_time.isoformat()}"')
        if end_time:
            filters.append(f'timestamp<="{end_time.isoformat()}"')

        return " AND ".join(filters)

    def _format_entry(self, entry: LogEntry) -> dict[str, Any]:
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

    def _iter_entries(
        self,
        filter_str: str,
        max_results: int,
        order_by: str = DESCENDING,
    ) -> Iterator[LogEntry]:
        """
        Iterate over log entries with retry support.

        Uses streaming to avoid loading all entries into memory at once.
        Retries on transient ServiceUnavailable errors with exponential backoff.
        """
        retry = Retry(
            predicate=lambda exc: isinstance(exc, google_exceptions.ServiceUnavailable),
            initial=1.0,
            maximum=10.0,
            multiplier=2.0,
            deadline=self.config.timeout_seconds,
        )

        entries_iter = self.client.list_entries(
            filter_=filter_str,
            order_by=order_by,
            max_results=max_results,
            retry=retry,
        )

        yield from entries_iter

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
    ) -> dict[str, Any]:
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

        # Run the synchronous API call in a thread pool with timeout
        loop = asyncio.get_running_loop()

        def fetch_entries() -> list[dict[str, Any]]:
            return [self._format_entry(e) for e in self._iter_entries(filter_str, limit)]

        try:
            formatted = await asyncio.wait_for(
                loop.run_in_executor(None, fetch_entries),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Log query timed out after {self.config.timeout_seconds} seconds"
            )

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
        """List available GKE clusters using the GKE Container API.

        Uses the Container API to get an accurate list of all clusters
        in the project, rather than inferring from log entries.
        """
        # Check cache first
        if self._clusters_cache is not None:
            cache_time, cached_clusters = self._clusters_cache
            if self._is_cache_valid(cache_time):
                logger.debug("Returning cached clusters list")
                return cached_clusters

        # Get project ID from logging client (handles auto-detection)
        project_id = self.client.project

        loop = asyncio.get_running_loop()

        def fetch_clusters() -> list[str]:
            # Use "-" as location to list clusters across all zones/regions
            parent = f"projects/{project_id}/locations/-"
            response = self.container_client.list_clusters(parent=parent)
            return sorted(cluster.name for cluster in response.clusters)

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, fetch_clusters),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Cluster listing timed out after {self.config.timeout_seconds} seconds"
            )

        # Update cache
        self._clusters_cache = (time.monotonic(), result)
        return result

    async def list_namespaces(self, cluster_name: str) -> list[str]:
        """List namespaces in a cluster that have logs."""
        validate_resource_name(cluster_name, "cluster_name")

        # Check cache first
        if cluster_name in self._namespaces_cache:
            cache_time, cached_namespaces = self._namespaces_cache[cluster_name]
            if self._is_cache_valid(cache_time):
                logger.debug(f"Returning cached namespaces list for {cluster_name}")
                return cached_namespaces

        escaped_cluster = escape_filter_string(cluster_name)
        filter_str = f'resource.type="k8s_container" AND resource.labels.cluster_name="{escaped_cluster}"'

        loop = asyncio.get_running_loop()

        def fetch_namespaces() -> list[str]:
            namespaces: set[str] = set()
            for entry in self._iter_entries(filter_str, max_results=1000):
                ns = entry.resource.labels.get("namespace_name")
                if ns:
                    namespaces.add(ns)
            return sorted(namespaces)

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, fetch_namespaces),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Namespace listing timed out after {self.config.timeout_seconds} seconds"
            )

        # Update cache
        self._namespaces_cache[cluster_name] = (time.monotonic(), result)
        return result


def create_server(config: GKELogsConfig | None = None) -> tuple[Server, GKELogsClient]:
    """
    Create and configure an MCP server instance.

    This factory function enables dependency injection for testing.

    Args:
        config: Optional configuration. If not provided, loads from environment.

    Returns:
        Tuple of (Server, GKELogsClient) instances.
    """
    if config is None:
        config = GKELogsConfig.from_env()

    app = Server("gke-logs-mcp")
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
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
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

        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return [TextContent(type="text", text=f"Validation error: {e}")]
        except TimeoutError as e:
            logger.error(f"Timeout: {e}")
            return [TextContent(type="text", text=f"Request timed out: {e}")]
        except google_exceptions.PermissionDenied as e:
            logger.error(f"Permission denied: {e}")
            return [TextContent(type="text", text=f"Permission denied. Ensure the service account has 'roles/logging.viewer' on the project.\nError: {e}")]
        except google_exceptions.NotFound as e:
            logger.error(f"Resource not found: {e}")
            return [TextContent(type="text", text=f"Resource not found. Check that the cluster name is correct.\nError: {e}")]
        except Exception as e:
            logger.exception(f"Error executing tool {name}")
            return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]

    return app, logs_client


async def main():
    """Run the MCP server."""
    config = GKELogsConfig.from_env()
    app, _ = create_server(config)

    logger.info("Starting GKE Logs MCP Server")
    logger.info(f"Project: {config.project_id or 'auto-detect'}")
    logger.info(f"Default max entries: {config.default_max_entries}")
    logger.info(f"Default hours back: {config.default_hours_back}")
    logger.info(f"Timeout: {config.timeout_seconds}s")
    logger.info(f"Cache TTL: {config.cache_ttl_seconds}s")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
