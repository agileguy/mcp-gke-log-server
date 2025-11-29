# GKE Logs MCP Server
# Multi-stage build for minimal production image

# === Build Stage ===
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# === Production Stage ===
FROM python:3.12-slim as production

# Security: Run as non-root user
RUN groupadd -r mcp && useradd -r -g mcp mcp

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY gke_logs_mcp/ ./gke_logs_mcp/
COPY healthcheck.py .

# Set ownership
RUN chown -R mcp:mcp /app

USER mcp

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

# Health check endpoint runs on 8080
EXPOSE 8080

# Default command runs the MCP server
# For production with health checks, use the entrypoint script
CMD ["python", "-m", "gke_logs_mcp.server"]
