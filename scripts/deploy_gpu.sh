#!/usr/bin/env bash
set -euo pipefail

# ─── 1) Go to repo root (where .env lives)
cd "$(dirname "$0")/.."

# ─── 2) Ensure .env is present
[[ -f .env ]] || { echo "❌ .env missing in $(pwd)"; exit 1; }

# ─── 3) Load KEY=VALUE lines from .env
export $(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env | xargs)

# ─── 4) Mandatory environment variables
: "${RUNPOD_API_KEY:?Missing RUNPOD_API_KEY}"
: "${RUNPOD_POD_ID:?Missing RUNPOD_POD_ID (must be the UUID, not the name)}"
: "${GHCR_TOKEN:?Missing GHCR_TOKEN}"
: "${GHCR_USER:?Missing GHCR_USER}"
: "${API_AUTH_TOKEN:?Missing API_AUTH_TOKEN}"

# ─── 5) Compute lowercase GHCR namespace & image name
GHCR_NS="${GHCR_USER,,}"
IMAGE="ghcr.io/${GHCR_NS}/faiss-gpu-api:latest"

# ─── 6) Clean up old local images
echo "🗑 Cleaning up old Docker images…"
docker rmi -f faiss-gpu-api:latest "${IMAGE}" 2>/dev/null || true
docker system prune -af

# ─── 7) Pre-pull the CUDA base (uses your host DNS)
echo "🔄 Pre-pulling CUDA base image…"
docker pull nvidia/cuda:11.8.0-runtime-ubuntu22.04

# ─── 8) Build your GPU API image
echo "🔨 Building Docker image (host network)…"
docker build \
  --network host \
  --no-cache \
  -f docker/Dockerfile.gpu \
  -t faiss-gpu-api:latest \
  .

# ─── 9) Login to GHCR and push
echo "🔑 Logging into ghcr.io…"
echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

echo "🚀 Tagging and pushing ${IMAGE}…"
docker tag faiss-gpu-api:latest "${IMAGE}"
docker push "${IMAGE}"

# ─── 10) Start & deploy to RunPod via REST API
RUNPOD_REST_URL="https://rest.runpod.io/v1"

# 10a) Check desiredStatus
RESPONSE=$(curl -s \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  "${RUNPOD_REST_URL}/pods/${RUNPOD_POD_ID}")
STATUS=$(echo "$RESPONSE" | (jq -r .desiredStatus 2>/dev/null || echo "UNKNOWN"))

if [[ "$STATUS" != "RUNNING" ]]; then
  echo "⚡ Pod is $STATUS — starting spot instance…"
  curl -s -X POST \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    "${RUNPOD_REST_URL}/pods/${RUNPOD_POD_ID}/start" \
    -d '{}'
  echo "⏱ Waiting 30s for pod init…"
  sleep 30
fi

# 10b) Deploy your container
echo "📦 Deploying container…"
curl -s -X POST \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  "${RUNPOD_REST_URL}/pods/${RUNPOD_POD_ID}/run" \
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

echo "✅ Deployment complete!"
echo
echo "Test your GPU API with:"
echo "  curl -X POST \\"
echo "    -H \"Authorization: Bearer ${API_AUTH_TOKEN}\" \\"
echo "    -F \"file=@test.jpg\" \\"
echo "    ${GPU_API_URL}/search?top_k=3"
