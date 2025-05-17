#!/usr/bin/env bash
set -euo pipefail

#───────────────────────────────────────────────────────────
# 1) Ensure required env vars are set
#───────────────────────────────────────────────────────────
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set in env}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID must be set in env}"
: "${GHCR_TOKEN:?GHCR_TOKEN must be set in env}"
: "${GHCR_USER:?GHCR_USER must be set in env}"
: "${API_AUTH_TOKEN:?API_AUTH_TOKEN must be set in env}"
: "${GPU_API_URL:?GPU_API_URL must be set in env}"
: "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH must be set in env}"

#───────────────────────────────────────────────────────────
# 2) Derive image name
#───────────────────────────────────────────────────────────
GHCR_NS="${GHCR_USER,,}"
IMAGE="ghcr.io/${GHCR_NS}/faiss-gpu-api:latest"

log() { echo -e "\n▶️  $*"; }
err() { echo -e "\n❌ $*" >&2; exit 1; }

#───────────────────────────────────────────────────────────
# 3) Cleanup old images locally
#───────────────────────────────────────────────────────────
log "Cleaning old local Docker images…"
docker rmi -f faiss-gpu-api:latest "${IMAGE}" 2>/dev/null || true
docker system prune -af

#───────────────────────────────────────────────────────────
# 4) Pre-pull base image (speeds up CI builds)
#───────────────────────────────────────────────────────────
log "Pre-pulling CUDA base image…"
docker pull nvidia/cuda:11.8.0-runtime-ubuntu22.04

#───────────────────────────────────────────────────────────
# 5) Build GPU Docker image
#───────────────────────────────────────────────────────────
log "Building GPU Docker image…"
docker build \
  --network host \
  --no-cache \
  -f docker/Dockerfile.gpu \
  -t faiss-gpu-api:latest \
  .

#───────────────────────────────────────────────────────────
# 6) Login & push to GHCR
#───────────────────────────────────────────────────────────
log "Logging into ghcr.io as ${GHCR_USER}…"
echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

log "Tagging image ${IMAGE}…"
docker tag faiss-gpu-api:latest "${IMAGE}"

log "Pushing to ${IMAGE}…"
docker push "${IMAGE}"

#───────────────────────────────────────────────────────────
# 7) Deploy to RunPod via their REST API
#───────────────────────────────────────────────────────────
REST="https://rest.runpod.io/v1"

# 7a) Fetch current pod desiredStatus
log "Checking RunPod pod status…"
POD_JSON=$(curl -fsSL -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  "${REST}/pods/${RUNPOD_POD_ID}") || err "Failed to GET pod info"
DESIRED=$(jq -r '.desiredStatus // "UNKNOWN"' <<<"$POD_JSON")

log "Pod desired status is: $DESIRED"
if [[ "$DESIRED" != "RUNNING" ]]; then
  log "Starting spot instance…"
  curl -fsSL -X POST \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    "${REST}/pods/${RUNPOD_POD_ID}/start" \
    -d '{}' \
    || err "Failed to start pod"
  log "Waiting 30s for pod initialization…"
  sleep 30
fi

# 7b) Trigger the new container run
log "Deploying GPU container to RunPod…"
DEPLOY_JSON=$(jq -n \
  --arg image "$IMAGE" \
  --arg token "$API_AUTH_TOKEN" \
  --arg path "$FAISS_INDEX_PATH" \
  '{
     image: $image,
     env: {
       API_AUTH_TOKEN: $token,
       FAISS_INDEX_PATH: $path
     },
     ports: ["8000/http"]
   }')

curl -fsSL -X POST \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  "${REST}/pods/${RUNPOD_POD_ID}/run" \
  -d "$DEPLOY_JSON" \
  || err "Failed to deploy new container"

#───────────────────────────────────────────────────────────
# 8) Success!
#───────────────────────────────────────────────────────────
log "✅ GPU API deployed!"
echo "Test with:"
echo "  curl -X POST \\"
echo "    -H \"Authorization: Bearer ${API_AUTH_TOKEN}\" \\"
echo "    -F \"file=@test.jpg\" \\"
echo "    ${GPU_API_URL}/search?top_k=3"
