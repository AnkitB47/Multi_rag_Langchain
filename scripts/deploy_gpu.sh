#!/usr/bin/env bash
set -euo pipefail

log() { echo -e "\n▶️  $*"; }
err() { echo -e "\n❌ $*" >&2; exit 1; }

#───────────────────────────────────────────────────────────
# 1) Required environment variables (from GitHub secrets)
#───────────────────────────────────────────────────────────
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID must be set}"
: "${GHCR_TOKEN:?GHCR_TOKEN must be set}"
: "${GHCR_USER:?GHCR_USER must be set}"
: "${API_AUTH_TOKEN:?API_AUTH_TOKEN must be set}"
: "${GPU_API_URL:?GPU_API_URL must be set}"
: "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH must be set}"

#───────────────────────────────────────────────────────────
# 2) Compute your GHCR image name
#───────────────────────────────────────────────────────────
GHCR_NS="${GHCR_USER,,}"
IMAGE="ghcr.io/${GHCR_NS}/faiss-gpu-api:latest"

#───────────────────────────────────────────────────────────
# 3) Cleanup any old local images
#───────────────────────────────────────────────────────────
log "Pruning old Docker images…"
docker rmi -f faiss-gpu-api:latest "${IMAGE}" 2>/dev/null || true
docker system prune -af

#───────────────────────────────────────────────────────────
# 4) Pre-pull base image (optional cache-warmer)
#───────────────────────────────────────────────────────────
log "Pre-pulling CUDA base image…"
docker pull nvidia/cuda:11.8.0-runtime-ubuntu22.04

#───────────────────────────────────────────────────────────
# 5) Build the GPU Docker image
#───────────────────────────────────────────────────────────
log "Building GPU image…"
docker build --network host --no-cache \
  -f docker/Dockerfile.gpu \
  -t faiss-gpu-api:latest \
  .

#───────────────────────────────────────────────────────────
# 6) Tag & push to GHCR
#───────────────────────────────────────────────────────────
log "Logging into ghcr.io as ${GHCR_USER}…"
echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

log "Tagging ${IMAGE}…"
docker tag faiss-gpu-api:latest "${IMAGE}"

log "Pushing ${IMAGE}…"
docker push "${IMAGE}"

#───────────────────────────────────────────────────────────
# 7) Deploy to RunPod via REST API
#───────────────────────────────────────────────────────────
REST="https://api.runpod.io/v1"

# 7a) GET the pod object at /v1/{podId}
log "Fetching pod info…"
pod_json=$(curl -fsSL \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  "${REST}/${RUNPOD_POD_ID}") || err "Failed to GET pod info at ${REST}/${RUNPOD_POD_ID}"

desired=$(jq -r '.desiredStatus // "UNKNOWN"' <<<"$pod_json")
log "Pod desiredStatus: $desired"

if [[ "$desired" != "RUNNING" ]]; then
  log "Starting spot instance…"
  curl -fsSL -X POST \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    "${REST}/${RUNPOD_POD_ID}/start" \
    -d '{}' \
    || err "Failed to start pod ${RUNPOD_POD_ID}"
  log "Waiting 30s for pod initialization…"
  sleep 30
fi

# 7b) POST your new container to /v1/{podId}/run
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

echo "$deploy_payload" | jq .   # debug

log "Deploying container to RunPod…"
http_code=$(curl -s -o /tmp/runpod_resp.json -w "%{http_code}" \
  -X POST \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  "${REST}/${RUNPOD_POD_ID}/run" \
  -d "$deploy_payload")

if (( http_code >= 200 && http_code < 300 )); then
  log "✅ Deployed successfully (HTTP $http_code)"
else
  err "RunPod deploy failed (HTTP $http_code):\n$(< /tmp/runpod_resp.json)"
fi

log "All done! Test with:"
echo "  curl -X POST \\"
echo "    -H \"Authorization: Bearer ${API_AUTH_TOKEN}\" \\"
echo "    -F \"file=@test.jpg\" \\"
echo "    ${GPU_API_URL}/search?top_k=3"
