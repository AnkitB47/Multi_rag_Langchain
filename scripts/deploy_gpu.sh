#!/usr/bin/env bash
set -euo pipefail

log()  { echo -e "\n▶️  $*"; }
fail() { echo -e "\n❌ $*" >&2; exit 1; }

# 1) Env checks
: "${GHCR_USER:?GHCR_USER must be set}"
: "${GHCR_TOKEN:?GHCR_TOKEN must be set}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set}"
: "${API_AUTH_TOKEN:?API_AUTH_TOKEN must be set}"
: "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH must be set}"

# 2) Image name
IMAGE="ghcr.io/${GHCR_USER,,}/faiss-gpu-api:latest"

# 3) Build & push
log "Building GPU image…"
docker build \
  --no-cache \
  -f docker/Dockerfile.gpu \
  -t faiss-gpu-api:latest .

log "Logging into ghcr.io…"
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin

log "Tagging & pushing $IMAGE…"
docker tag faiss-gpu-api:latest "$IMAGE"
docker push "$IMAGE"

# 4) Spin up a fresh spot pod via GraphQL
log "Deploying new Spot Pod via RunPod GraphQL…"
read -r -d '' PAYLOAD <<EOF
{
  "query":"mutation DeploySpot(\$in: PodRentInterruptableInput!){ podRentInterruptable(input:\$in){ id name desiredStatus }}",
  "variables":{
    "in":{
      "name":"multi-rag-langgraph-\$(date +%s)",
      "gpuCount":1,
      "minVcpuCount":8,
      "minMemoryInGb":30,
      "volumeInGb":20,
      "containerDiskInGb":5,
      "imageName":"$IMAGE",
      "gpuTypeId":"NVIDIA RTX 3080 Ti",
      "ports":"8000/http",
      "env":[
        { "key":"API_AUTH_TOKEN",   "value":"$API_AUTH_TOKEN" },
        { "key":"FAISS_INDEX_PATH", "value":"$FAISS_INDEX_PATH" }
      ]
    }
  }
}
EOF

response=$(curl -fsSL -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.io/graphql \
  -d "$PAYLOAD")

podId=$(jq -r '.data.podRentInterruptable.id' <<<"$response")
status=$(jq -r '.data.podRentInterruptable.desiredStatus' <<<"$response")

[[ -n "$podId" && "$status" == "RUNNING" ]] \
  || fail "GraphQL deploy failed:\n$response"

log "✅ Spot Pod created: ID=$podId (running)"

echo
echo "Your GPU Image-Search API is booting up. Test it at:"
echo "  curl -X POST \\"
echo "    -H \"Authorization: Bearer $API_AUTH_TOKEN\" \\"
echo "    -F \"file=@test.jpg\" \\"
echo "    http://<pod-public-ip>:8000/search?top_k=3"
