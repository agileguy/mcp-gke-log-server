#!/bin/bash
# GKE Logs MCP Server - GCP Setup Script
# Sets up service account, IAM bindings, and Workload Identity

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() { echo -e "${GREEN}==>${NC} $1"; }
print_warn() { echo -e "${YELLOW}WARNING:${NC} $1"; }
print_error() { echo -e "${RED}ERROR:${NC} $1"; }

# Configuration - modify these for your environment
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
CLUSTER_NAME="${GKE_CLUSTER_NAME:-}"
SA_NAME="gke-logs-mcp"
K8S_NAMESPACE="mcp-servers"
K8S_SA_NAME="gke-logs-mcp"

# Validate required variables
if [[ -z "$PROJECT_ID" ]]; then
    print_error "GCP_PROJECT_ID environment variable is required"
    exit 1
fi

print_step "Setting up GKE Logs MCP Server for project: $PROJECT_ID"

# 1. Enable required APIs
print_step "Enabling required APIs..."
gcloud services enable \
    logging.googleapis.com \
    container.googleapis.com \
    iam.googleapis.com \
    --project="$PROJECT_ID"

# 2. Create service account
print_step "Creating service account: $SA_NAME"
if gcloud iam service-accounts describe "${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" --project="$PROJECT_ID" &>/dev/null; then
    print_warn "Service account already exists, skipping creation"
else
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="GKE Logs MCP Server" \
        --description="Service account for MCP server to read GKE logs" \
        --project="$PROJECT_ID"
fi

# 3. Grant logging viewer role
print_step "Granting logging.viewer role..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/logging.viewer" \
    --condition=None

# 4. Set up Workload Identity (if cluster name provided)
if [[ -n "$CLUSTER_NAME" ]]; then
    print_step "Setting up Workload Identity binding..."
    
    # Get cluster's workload identity pool
    WI_POOL="${PROJECT_ID}.svc.id.goog"
    
    # Bind the Kubernetes service account to the GCP service account
    gcloud iam service-accounts add-iam-policy-binding \
        "${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/iam.workloadIdentityUser" \
        --member="serviceAccount:${WI_POOL}[${K8S_NAMESPACE}/${K8S_SA_NAME}]" \
        --project="$PROJECT_ID"
    
    print_step "Workload Identity configured for ${K8S_NAMESPACE}/${K8S_SA_NAME}"
else
    print_warn "GKE_CLUSTER_NAME not set, skipping Workload Identity setup"
    print_warn "For local development, create a key file:"
    echo ""
    echo "  gcloud iam service-accounts keys create gcp-key.json \\"
    echo "    --iam-account=${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    echo ""
fi

# 5. Output summary
echo ""
print_step "Setup complete! Summary:"
echo "  Project ID: $PROJECT_ID"
echo "  Service Account: ${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
echo "  Role: roles/logging.viewer"
if [[ -n "$CLUSTER_NAME" ]]; then
    echo "  Workload Identity: ${K8S_NAMESPACE}/${K8S_SA_NAME}"
fi

echo ""
print_step "Next steps:"
echo "  1. Build the container: docker build -t gcr.io/$PROJECT_ID/gke-logs-mcp:latest ."
echo "  2. Push to registry: docker push gcr.io/$PROJECT_ID/gke-logs-mcp:latest"
echo "  3. Update k8s/deployment.yaml with your project ID"
echo "  4. Deploy: kubectl apply -f k8s/deployment.yaml"
