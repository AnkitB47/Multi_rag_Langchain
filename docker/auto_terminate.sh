#!/usr/bin/env bash
set -eo pipefail

# ──── 1) How long to run before self-terminate ─────────────
# Hard-code to exactly one hour (3600s)
TERMINATE_AFTER=3600

echo "🕒 Pod will self-terminate in $TERMINATE_AFTER seconds (1 hour)"
sleep $TERMINATE_AFTER

# ──── 2) Perform self-termination ─────────────────────────
echo "🔴 Initiating self-termination…"

# Pull RunPod API key from env
API_KEY=${RUNPOD_API_KEY:-}

if [ -z "$API_KEY" ]; then
    echo "❌ Missing RUNPOD_API_KEY" >&2
    exit 1
fi

# Query the local RunPod agent for this pod’s ID
POD_ID=$(curl -fsSL \
  -H "Authorization: Bearer $API_KEY" \
  http://localhost/rp/v1/pod/get \
  | jq -r '.id // empty')

if [ -z "$POD_ID" ]; then
    echo "❌ Could not retrieve pod ID from local RunPod agent" >&2
    exit 1
fi

# Tell RunPod to terminate it
curl -fsSL -X POST \
  -H "Authorization: Bearer $API_KEY" \
  "http://localhost/rp/v1/pod/terminate/$POD_ID" \
  || { echo "❌ Pod termination request failed" >&2; exit 1; }

echo "✅ Termination request sent successfully (pod $POD_ID)"
