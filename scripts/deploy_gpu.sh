#!/usr/bin/env python3
import os, sys, subprocess
import runpod
from time import sleep

# 1) Read env vars
GHCR_USER = os.environ["GHCR_USER"]
GHCR_TOKEN = os.environ["GHCR_TOKEN"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
API_AUTH_TOKEN = os.environ["API_AUTH_TOKEN"]
FAISS_INDEX = os.environ["FAISS_INDEX_PATH"]

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

# 5) Define GPU priority list with fallback options
GPU_PRIORITY_LIST = [
    "NVIDIA GeForce RTX 3080 Ti",  # First choice
    "NVIDIA GeForce RTX 3080",     # Second choice
    "NVIDIA RTX A4000",            # Third choice
    "NVIDIA GeForce RTX 3090",     # Fourth choice
    "NVIDIA GeForce RTX 3070",     # Fifth choice
    "NVIDIA RTX A5000"            # Final fallback
]

# 6) Try creating pod with different GPUs
max_retries = 3
retry_delay = 30  # seconds
pod = None

for attempt in range(max_retries):
    for gpu_type in GPU_PRIORITY_LIST:
        try:
            print(f"▶️ Attempt {attempt + 1}: Trying GPU {gpu_type}...")
            response = runpod.create_pod(
                name="multi-rag-langgraph",
                image_name=image,
                gpu_type_id=gpu_type,
                cloud_type="SECURE",
                gpu_count=1,
                volume_in_gb=50,
                container_disk_in_gb=20,
                ports="8000/http",
                volume_mount_path="/data",
                env={
                    "API_AUTH_TOKEN": API_AUTH_TOKEN,
                    "FAISS_INDEX_PATH": FAISS_INDEX
                },
                support_public_ip=True,
                min_vcpu_count=8,
                min_memory_in_gb=30,
                bid_percent=50  # Added spot pricing
            )
            
            # Handle both new and old API response formats
            if hasattr(response, 'id'):
                # New SDK version with Pod object
                pod = response
                if pod.status in ("RUNNING", "RESUMED"):
                    break
                print(f"⚠️ Pod created but not running. Status: {pod.status}")
                runpod.terminate_pod(pod.id)
            else:
                # Old SDK version with dict response
                pod_id = response['data']['podFindAndDeployOnDemand']['id']
                print(f"ℹ️ Pod created with ID: {pod_id}")
                # Wait for pod to be ready
                for _ in range(10):
                    sleep(10)
                    pod = runpod.get_pod(pod_id)
                    if pod and pod.status in ("RUNNING", "RESUMED"):
                        break
                if pod and pod.status in ("RUNNING", "RESUMED"):
                    break
                print(f"⚠️ Pod not ready after waiting. Status: {pod.status if pod else 'unknown'}")
                if pod:
                    runpod.terminate_pod(pod.id)
            
        except Exception as e:
            print(f"⚠️ Failed with {gpu_type}: {str(e)}")
            continue
    
    if pod and pod.status in ("RUNNING", "RESUMED"):
        break  # Success!
    
    if attempt < max_retries - 1:
        print(f"🔄 Retrying in {retry_delay} seconds...")
        sleep(retry_delay)
else:
    sys.exit("❌ Failed to create pod after multiple attempts")

# 7) Success - get public IP and print info
try:
    if not hasattr(pod, 'public_ip'):
        pod = runpod.get_pod(pod.id if hasattr(pod, 'id') else pod['data']['podFindAndDeployOnDemand']['id'])
    
    public_ip = pod.public_ip
    print("✅ Pod created successfully!")
    print(f"GPU Type: {pod.gpu_type_id if hasattr(pod, 'gpu_type_id') else 'unknown'}")
    print(f"Pod ID: {pod.id}")
    print(f"Public IP: {public_ip}")
    print(f"""
🚀 GPU Image-Search API is live at:
   http://{public_ip}:8000/search?top_k=3

Test with:
  curl -X POST \\
    -H "Authorization: Bearer {API_AUTH_TOKEN}" \\
    -F file=@test.jpg \\
    http://{public_ip}:8000/search?top_k=3
""")
except Exception as e:
    sys.exit(f"❌ Failed to get pod info: {str(e)}")