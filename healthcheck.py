#!/usr/bin/env python3
"""
Health Check Server for GKE Logs MCP

Runs alongside the MCP server to provide health/readiness endpoints
for Kubernetes deployments.
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import uvicorn

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("healthcheck")

# Track MCP server process
mcp_process: subprocess.Popen | None = None


async def health(request):
    """Liveness probe - is the container running?"""
    return JSONResponse({"status": "healthy"})


async def ready(request):
    """Readiness probe - is the MCP server ready to accept connections?"""
    global mcp_process
    
    if mcp_process is None:
        return JSONResponse({"status": "not_started"}, status_code=503)
    
    if mcp_process.poll() is not None:
        return JSONResponse(
            {"status": "crashed", "returncode": mcp_process.returncode},
            status_code=503
        )
    
    return JSONResponse({"status": "ready"})


async def startup():
    """Start the MCP server as a subprocess."""
    global mcp_process
    
    logger.info("Starting MCP server subprocess...")
    mcp_process = subprocess.Popen(
        [sys.executable, "-m", "gke_logs_mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    logger.info(f"MCP server started with PID {mcp_process.pid}")


async def shutdown():
    """Gracefully shutdown the MCP server."""
    global mcp_process
    
    if mcp_process:
        logger.info("Shutting down MCP server...")
        mcp_process.terminate()
        try:
            mcp_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("MCP server did not terminate gracefully, killing...")
            mcp_process.kill()


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/ready", ready, methods=["GET"]),
    Route("/healthz", health, methods=["GET"]),  # K8s convention
    Route("/readyz", ready, methods=["GET"]),    # K8s convention
]

app = Starlette(
    routes=routes,
    on_startup=[startup],
    on_shutdown=[shutdown],
)


def main():
    port = int(os.getenv("HEALTH_PORT", "8080"))
    logger.info(f"Starting health check server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",  # Reduce noise from health checks
    )


if __name__ == "__main__":
    main()
