#!/usr/bin/env bash
set -euo pipefail

log()  { echo -e "\n▶️  $*"; }
err()  { echo -e "\n❌ $*" >&2; exit 1; }

# ── 1) Mandatory environment variables (from GitHub Secrets) ───────
: "${GHCR_USER:?GHCR_USER must be set}"
: "${GHCR_TOKEN:?GHCR_TOKEN must be set}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set}"
: "${API_AUTH_TOKEN:?API_AUTH_TOKEN must be set}"
: "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH must be set}"

# ── 2) Prepare image name ─────────────────────────────────────────
IMAGE="ghcr.io/${GHCR_USER,,}/faiss-gpu-api:latest"

# ── 3) Install runpodctl CLI if missing ──────────────────────────
if ! command -v runpodctl &>/dev/null; then
  log "Installing runpodctl CLI…"
  # Download the latest linux-amd64 binary, make it executable, and install it
  curl -sL https://github.com/runpod/cli/releases/latest/download/runpodctl-linux-amd64 \
    -o runpodctl
  chmod +x runpodctl
  sudo mv runpodctl /usr/local/bin/
fi

# ── 4) Log into GHCR, build & push your GPU image ────────────────
log "Building GPU image…"
docker build --no-cache -f docker/Dockerfile.gpu -t faiss-gpu-api:latest .

log "Logging into ghcr.io…"
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin

log "Tagging & pushing $IMAGE…"
docker tag faiss-gpu-api:latest "$IMAGE"
docker push "$IMAGE"

# ── 5) Tear down any existing Spot Pod with our fixed name ───────
POD_NAME="multi-rag-langgraph"
existing=$(runpodctl pod list --apiKey "$RUNPOD_API_KEY" --output json \
  | jq -r --arg n "$POD_NAME" '.[] | select(.name==$n) | .id')

if [[ -n "$existing" ]]; then
  log "Deleting old pod $existing…"
  runpodctl pod delete --apiKey "$RUNPOD_API_KEY" "$existing"
fi

# ── 6) Rent a new Spot Pod with our image ────────────────────────
log "Renting new Spot Pod…"
new_id=$(runpodctl pod rent-interruptable \
  --apiKey "$RUNPOD_API_KEY" \
  --name "$POD_NAME" \
  --imageName "$IMAGE" \
  --gpuTypeName "NVIDIA RTX 3080 Ti" \
  --gpuCount 1 \
  --vcpu 8 \
  --memoryGB 30 \
  --volumeGB 20 \
  --diskGB 5 \
  --ports "8000/http" \
  --env "API_AUTH_TOKEN=$API_AUTH_TOKEN" \
  --env "FAISS_INDEX_PATH=$FAISS_INDEX_PATH" \
  --output json \
  | jq -r '.id')

if [[ -z "$new_id" ]]; then
  err "Failed to rent new Spot Pod"
fi

# ── 7) Fetch its public IP ───────────────────────────────────────
public_ip=$(runpodctl pod get --apiKey "$RUNPOD_API_KEY" "$new_id" \
  --output json | jq -r '.publicIp')

log "✅ Spot Pod created! ID=$new_id  IP=$public_ip"

echo
echo "Your GPU Image‐Search API is live at http://$public_ip:8000/search"
echo "Test with:"
echo "  curl -X POST \\"
echo "    -H \"Authorization: Bearer $API_AUTH_TOKEN\" \\"
echo "    -F \"file=@test.jpg\" \\"
echo "    http://$public_ip:8000/search?top_k=3"
