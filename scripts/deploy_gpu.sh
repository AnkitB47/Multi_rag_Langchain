#!/usr/bin/env bash
set -euo pipefail

log() { echo -e "\n▶️  $*"; }
err() { echo -e "\n❌ $*" >&2; exit 1; }

# 1) Required envs
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set in env}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID must be set in env}"
: "${GHCR_TOKEN:?GHCR_TOKEN must be set in env}"
: "${GHCR_USER:?GHCR_USER must be set in env}"
: "${API_AUTH_TOKEN:?API_AUTH_TOKEN must be set in env}"
: "${GPU_API_URL:?GPU_API_URL must be set in env}"
: "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH must be set in env}"

# 2) Image name
GHCR_NS="${GHCR_USER,,}"
IMAGE="ghcr.io/${GHCR_NS}/faiss-gpu-api:latest"

# 3) Cleanup
log "Pruning old images…"
docker rmi -f faiss-gpu-api:latest "${IMAGE}" 2>/dev/null || true
docker system prune -af

# 4) Pre-pull
log "Pre-pulling CUDA base…"
docker pull nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 5) Build
log "Building GPU image…"
docker build --network host --no-cache -f docker/Dockerfile.gpu -t faiss-gpu-api:latest .

# 6) Push to GHCR
log "Logging into ghcr.io…"
echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin
log "Tagging ${IMAGE}"
docker tag faiss-gpu-api:latest "${IMAGE}"
log "Pushing ${IMAGE}…"
docker push "${IMAGE}"

# 7) Deploy to RunPod
REST="https://rest.runpod.io/v1"

# 7a) Start pod if not RUNNING
log "Checking pod status…"
status_json=$(curl -fsSL -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  "${REST}/pods/${RUNPOD_POD_ID}")
current=$(jq -r '.desiredStatus // "UNKNOWN"' <<<"$status_json")
log "Pod desiredStatus: $current"
if [[ "$current" != "RUNNING" ]]; then
  log "Starting spot instance…"
  curl -fsSL -X POST \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    "${REST}/pods/${RUNPOD_POD_ID}/start" \
    -d '{}' || err "pod start failed"
  log "Waiting 30s for pod init…"; sleep 30
fi

# 7b) Trigger run
log "Preparing deployment payload…"
deploy_payload=$(jq -n \
  --arg image "$IMAGE" \
  --arg token "$API_AUTH_TOKEN" \
  --arg path  "$FAISS_INDEX_PATH" \
  '{
    image: $image,
    env: {
      API_AUTH_TOKEN: $token,
      FAISS_INDEX_PATH: $path
    },
    ports: ["8000/http"]
  }')
echo "$deploy_payload" | jq .   #  show exact payload

log "Deploying to RunPod…"
http_code=$(curl -s -o /tmp/runpod_resp.json -w "%{http_code}" \
  -X POST \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  "${REST}/pods/${RUNPOD_POD_ID}/run" \
  -d "$deploy_payload")

if [[ "$http_code" -ge 200 && "$http_code" -lt 300 ]]; then
  log "✅ Successfully deployed (HTTP $http_code)"
else
  err "RunPod deploy failed (HTTP $http_code). Response:\n$(cat /tmp/runpod_resp.json)"
fi

log "All done—test your GPU API with:"
echo "  curl -X POST \\"
echo "    -H \"Authorization: Bearer ${API_AUTH_TOKEN}\" \\"
echo "    -F \"file=@test.jpg\" \\"
echo "    ${GPU_API_URL}/search?top_k=3"
