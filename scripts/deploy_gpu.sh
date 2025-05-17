#!/usr/bin/env python3
import os, sys, subprocess
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
    "docker", "build", "-f", "docker/Dockerfile.gpu",
    "-t", "faiss-gpu-api:latest", "."
], check=True)

login = subprocess.Popen(
    ["docker", "login", "ghcr.io", "-u", GHCR_USER, "--password-stdin"],
    stdin=subprocess.PIPE
)
login.communicate(input=GHCR_TOKEN.encode())
if login.returncode:
    sys.exit("❌ Docker login failed!")

subprocess.run(["docker", "tag", "faiss-gpu-api:latest", image], check=True)
subprocess.run(["docker", "push", image], check=True)

# 3) Configure runpod SDK
runpod.api_key = RUNPOD_API_KEY

# 4) Delete old pod if it exists
for p in runpod.get_pods():
    if p.name == "multi-rag-langgraph":
        print(f"▶️ Deleting old pod {p.id}")
        runpod.terminate_pod(p.id)

# 5) Create new interruptible GPU pod
print("▶️ Creating new spot pod…")
pod = runpod.create_pod(
    name="multi-rag-langgraph",
    image_name=image,
    gpu_type_id="NVIDIA GeForce RTX 3080 Ti",  
    cloud_type="SECURE",               # Required parameter
    gpu_count=1,                       # Number of GPUs
    volume_in_gb=50,                   # Storage size
    container_disk_in_gb=20,           # Container disk size
    ports="8000/http",                 # Expose port 8000
    volume_mount_path="/data",         # Mount path for volume
    env={"API_AUTH_TOKEN": API_AUTH_TOKEN, "FAISS_INDEX_PATH": FAISS_INDEX},
    support_public_ip=True             # Enable public IP
)

if pod.status not in ("RUNNING", "RESUMED"):
    print("❌ Pod did not start:", pod)
    sys.exit(1)

# 6) Fetch public IP
public_ip = runpod.get_pod(pod.id).public_ip

print("✅ Pod created:", pod.id, "IP:", public_ip)
print(f"""
🚀 GPU Image-Search API is live at:
   http://{public_ip}:8000/search?top_k=3

Test with:
  curl -X POST \\
    -H "Authorization: Bearer {API_AUTH_TOKEN}" \\
    -F file=@test.jpg \\
    http://{public_ip}:8000/search?top_k=3
""")