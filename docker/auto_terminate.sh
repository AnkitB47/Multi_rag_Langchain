#!/usr/bin/env bash
set -eo pipefail

# Load all environment variables (including secrets)
source /etc/environment

# Verify termination time
if [ -z "$TERMINATE_AT" ]; then
    echo "ℹ️ Running indefinitely (no TERMINATE_AT set)"
    exit 0
fi

# Calculate sleep duration
now=$(date +%s)
terminate=$(date -d "$TERMINATE_AT" +%s)
duration=$((terminate - now))

if [ $duration -le 0 ]; then
    echo "⏰ Termination time already passed" >&2
    exit 1
fi

echo "🕒 Pod will self-terminate at $TERMINATE_AT (in $duration seconds)"

# Sleep in intervals to handle signals
while [ $duration -gt 0 ]; do
    sleep $(( duration > 300 ? 300 : duration ))  # Sleep in 5-minute chunks
    now=$(date +%s)
    duration=$((terminate - now))
done

# Self-termination
echo "🔴 Initiating self-termination..."
API_KEY=${RUNPOD_API_KEY:-$RUNPOD_AI_API_KEY}

if [ -z "$API_KEY" ]; then
    echo "❌ Missing RunPod API key" >&2
    exit 1
fi

# Get pod ID through RunPod's local API
POD_ID=$(curl -sSf -H "Authorization: Bearer $API_KEY" \
    "http://localhost/rp/v1/pod/get" | jq -r '.id')

if [ -z "$POD_ID" ]; then
    echo "❌ Could not retrieve pod ID" >&2
    exit 1
fi

# Send termination request
curl -sSf -X POST -H "Authorization: Bearer $API_KEY" \
    "http://localhost/rp/v1/pod/terminate/$POD_ID" >/dev/null

echo "✅ Termination request sent successfully"
exit 0