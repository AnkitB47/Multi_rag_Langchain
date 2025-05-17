#!/usr/bin/env bash
set -euo pipefail

# ─── 1) Ensure required env vars are set ────────────────────────────────
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set in env}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID must be set in env}"
: "${GHCR_TOKEN:?GHCR_TOKEN must be set in env}"
: "${GHCR_USER:?GHCR_USER must be set in env}"
: "${API_AUTH_TOKEN:?API_AUTH_TOKEN must be set in env}"
: "${GPU_API_URL:?GPU_API_URL must be set in env}"
: "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH must be set in env}"

# ─── 2) Compute GHCR image name ─────────────────────────────────────────
GHCR_NS="${GHCR_USER,,}"
IMAGE="ghcr.io/${GHCR_NS}/faiss-gpu-api:latest"

# ─── 3) Clean up any old local images ─────────────────────────────────
echo "🗑 Cleaning old Docker images…"
docker rmi -f faiss-gpu-api:latest "${IMAGE}" 2>/dev/null || true
docker system prune -af

# ─── 4) Pre-pull CUDA base ──────────────────────────────────────────────
echo "🔄 Pre-pulling CUDA base image…"
docker pull nvidia/cuda:11.8.0-runtime-ubuntu22.04

# ─── 5) Build GPU API Docker image ────────────────────────────────────
echo "🔨 Building GPU Docker image…"
docker build \
  --network host \
  --no-cache \
  -f docker/Dockerfile.gpu \
  -t faiss-gpu-api:latest \
  .

# ─── 6) Push to GHCR ───────────────────────────────────────────────────
echo "🔑 Logging into ghcr.io…"
echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

echo "🚀 Tagging and pushing ${IMAGE}…"
docker tag faiss-gpu-api:latest "${IMAGE}"
docker push "${IMAGE}"

# ─── 7) Deploy to RunPod via REST API ──────────────────────────────────
REST="https://rest.runpod.io/v1"

# 7a) Make sure pod is running
STATUS=$(curl -s -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  "${REST}/pods/${RUNPOD_POD_ID}" | jq -r .desiredStatus // echo UNKNOWN)

if [[ "$STATUS" != "RUNNING" ]]; then
  echo "⚡ Pod is $STATUS — starting spot instance…"
  curl -s -X POST \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    "${REST}/pods/${RUNPOD_POD_ID}/start" \
    -d '{}'
  echo "⏱ Waiting 30s for pod init…"
  sleep 30
fi

# 7b) Tell RunPod to run our new image
echo "📦 Deploying GPU container on RunPod…"
curl -s -X POST \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  "${REST}/pods/${RUNPOD_POD_ID}/run" \
  -d "$(jq -n \
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
      }')"

echo "✅ GPU API deployed—try:"
echo "   curl -X POST \\"
echo "     -H \"Authorization: Bearer ${API_AUTH_TOKEN}\" \\"
echo "     -F \"file=@test.jpg\" \\"
echo "     ${GPU_API_URL}/search?top_k=3"
