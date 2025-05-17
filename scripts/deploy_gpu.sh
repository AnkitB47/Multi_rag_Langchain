#!/usr/bin/env python3
import os, sys
import subprocess
import runpod

# 1) Read env vars
GHCR_USER      = os.environ["GHCR_USER"]
GHCR_TOKEN     = os.environ["GHCR_TOKEN"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
API_AUTH_TOKEN = os.environ["API_AUTH_TOKEN"]
FAISS_INDEX    = os.environ["FAISS_INDEX_PATH"]

# 2) Build & push Docker image
image = f"ghcr.io/{GHCR_USER.lower()}/faiss-gpu-api:latest"
subprocess.run([
    "docker", "build", "-f", "docker/Dockerfile.gpu", "-t", "faiss-gpu-api:latest", "."
], check=True)
subprocess.run(
    ["docker", "login", "ghcr.io", "-u", GHCR_USER, "--password-stdin"],
    input=GHCR_TOKEN.encode(), check=True
)
subprocess.run(["docker", "tag", "faiss-gpu-api:latest", image], check=True)
subprocess.run(["docker", "push", image], check=True)

# 3) Configure runpod
runpod.api_key = RUNPOD_API_KEY

# 4) Delete old pod if exists
pods = runpod.get_pods()
for p in pods:
    if p.name == "multi-rag-langgraph":
        print(f"▶️ Deleting old pod {p.id}")
        runpod.terminate_pod(p.id)

# 5) Create new interruptible (spot) GPU pod
print("▶️ Creating new spot pod…")
pod = runpod.create_pod(
    name="multi-rag-langgraph",
    image_name=image,
    gpu_type="NVIDIA RTX 3080 Ti",
    gpu_count=1,
    vcpu_count=8,
    memory_gb=30,
    volume_gb=20,
    container_disk_gb=5,
    ports=["8000/http"],
    env={
        "API_AUTH_TOKEN": API_AUTH_TOKEN,
        "FAISS_INDEX_PATH": FAISS_INDEX
    },
    interruptible=True,
)

if pod.status not in ("RUNNING", "RESUMED"):
    print("❌ Pod did not start:", pod)
    sys.exit(1)

print("✅ Pod created:", pod.id, "IP:", pod.public_ip)
print(f"Test with: curl -X POST -H \"Authorization: Bearer {API_AUTH_TOKEN}\" "
      f"-F file=@test.jpg http://{pod.public_ip}:8000/search?top_k=3")
