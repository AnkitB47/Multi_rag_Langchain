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

# ─── 7) Deploy to RunPod via REST API ──────────────────────────────────
REST="https://api.runpod.io/v1"

# 7a) Make sure pod is RUNNING
log "Checking RunPod pod status…"
status_json=$(curl -fsSL \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  "${REST}/pods/${RUNPOD_POD_ID}") || err "Failed to GET pod info"
DESIRED=$(jq -r '.desiredStatus // "UNKNOWN"' <<<"$status_json")
log "Pod desiredStatus: $DESIRED"
if [[ "$DESIRED" != "RUNNING" ]]; then
  log "Starting spot instance…"
  curl -fsSL -X POST \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    "${REST}/pods/${RUNPOD_POD_ID}/start" \
    -d '{}' || err "Pod start failed"
  log "Waiting 30s for pod init…"
  sleep 30
fi

# 7b) Trigger the new container run
log "Deploying GPU container to RunPod…"
DEPLOY_PAYLOAD=$(jq -n \
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
echo "$DEPLOY_PAYLOAD" | jq .  # show exactly what we send

HTTP_CODE=$(curl -s -o /tmp/runpod_resp.json -w "%{http_code}" \
  -X POST \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  "${REST}/pods/${RUNPOD_POD_ID}/run" \
  -d "$DEPLOY_PAYLOAD")

if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
  log "✅ Successfully deployed (HTTP $HTTP_CODE)"
else
  err "RunPod deploy failed (HTTP $HTTP_CODE)\n$(cat /tmp/runpod_resp.json)"
fi

log "All done—test your GPU API with:"
echo "  curl -X POST \\"
echo "    -H \"Authorization: Bearer ${API_AUTH_TOKEN}\" \\"
echo "    -F \"file=@test.jpg\" \\"
echo "    ${GPU_API_URL}/search?top_k=3"
