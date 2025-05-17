#!/usr/bin/env bash
set -euo pipefail

log()  { echo -e "\n▶️  $*"; }
fail() { echo -e "\n❌ $*" >&2; exit 1; }

#───────────────────────────────────────────────────────────
# 1) Required env vars (set these from GitHub Actions secrets)
#───────────────────────────────────────────────────────────
: "${GHCR_USER:?GHCR_USER must be set}"
: "${GHCR_TOKEN:?GHCR_TOKEN must be set}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set}"
: "${API_AUTH_TOKEN:?API_AUTH_TOKEN must be set}"
: "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH must be set}"

#───────────────────────────────────────────────────────────
# 2) Prepare image name
#───────────────────────────────────────────────────────────
IMAGE="ghcr.io/${GHCR_USER,,}/faiss-gpu-api:latest"

#───────────────────────────────────────────────────────────
# 3) Install the official runpodctl CLI if missing
#───────────────────────────────────────────────────────────
if ! command -v runpodctl &> /dev/null; then
  log "Installing runpodctl CLI…"
  wget -qO- cli.runpod.net | sudo bash         # ❌ DO NOT use GitHub raw URLs!
fi

# 4) Configure it once
runpodctl config --apiKey="${RUNPOD_API_KEY}"

#───────────────────────────────────────────────────────────
# 5) Build & push your GPU Docker image
#───────────────────────────────────────────────────────────
log "Building GPU image…"
docker build \
  --no-cache \
  -f docker/Dockerfile.gpu \
  -t faiss-gpu-api:latest \
  .

log "Logging into ghcr.io…"
echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

log "Tagging & pushing ${IMAGE}…"
docker tag faiss-gpu-api:latest "${IMAGE}"
docker push "${IMAGE}"

#───────────────────────────────────────────────────────────
# 6) Rent a fresh Spot Pod via RunPod GraphQL ─────────────
#───────────────────────────────────────────────────────────
# We use the real GraphQL endpoint: https://api.runpod.io/graphql

read -r -d '' PAYLOAD <<EOF
{
  "query": "mutation RentSpot(\$in: PodRentInterruptableInput!) { podRentInterruptable(input:\$in) { id publicIp desiredStatus } }",
  "variables": {
    "in": {
      "name": "gpu-search-$(date +%s)",
      "gpuCount": 1,
      "minVcpuCount": 8,
      "minMemoryInGb": 30,
      "volumeInGb": 20,
      "containerDiskInGb": 5,
      "imageName": "${IMAGE}",
      "gpuTypeId": "NVIDIA RTX 3080 Ti",
      "ports": "8000/http",
      "env": [
        { "key": "API_AUTH_TOKEN",   "value": "${API_AUTH_TOKEN}" },
        { "key": "FAISS_INDEX_PATH", "value": "${FAISS_INDEX_PATH}" }
      ]
    }
  }
}
EOF

log "Renting new Spot Pod via GraphQL…"
resp=$(curl -fsSL \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  https://api.runpod.io/graphql \
  -d "${PAYLOAD}")

podId=$(jq -r .data.podRentInterruptable.id    <<<"$resp")
ip  =$(jq -r .data.podRentInterruptable.publicIp<<<"$resp")
status=$(jq -r .data.podRentInterruptable.desiredStatus<<<"$resp")

[[ "$podId" != "null" && "$status" == "RUNNING" ]] \
  || fail "Failed to rent spot pod:\n$resp"

log "✅ Pod created! ID=$podId  IP=$ip"

echo
echo "🖼  GPU Image-Search API is now live at:"
echo "   http://${ip}:8000/search?top_k=3"
echo "   curl -X POST \\"
echo "     -H \"Authorization: Bearer ${API_AUTH_TOKEN}\" \\"
echo "     -F \"file=@test.jpg\" \\"
echo "     http://${ip}:8000/search?top_k=3"
