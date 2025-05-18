#!/usr/bin/env python3
import os, sys, subprocess
import runpod
from time import sleep

# Configuration
CONFIG = {
    "service_name": "multi-rag-image-service",
    "service_port": 8000,
    "gpu_type": "NVIDIA GeForce RTX 3080 Ti",
    "volume_size_gb": 50,
    "container_disk_gb": 20,
    "min_vcpu": 8,
    "min_memory_gb": 30,
    "max_retries": 3,
    "retry_delay": 30
}

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

# 4) Delete old pod if exists
print("▶️ Cleaning up existing pods...")
for p in runpod.get_pods():
    if isinstance(p, dict):
        if p.get('name') == CONFIG["service_name"]:
            print(f"Deleting pod {p['id']}")
            runpod.terminate_pod(p['id'])
    elif hasattr(p, 'name') and p.name == CONFIG["service_name"]:
        print(f"Deleting pod {p.id}")
        runpod.terminate_pod(p.id)

# 5) Create new pod
print(f"▶️ Creating new pod with {CONFIG['gpu_type']}...")
for attempt in range(CONFIG["max_retries"]):
    try:
        pod = runpod.create_pod(
            name=CONFIG["service_name"],
            image_name=image,
            gpu_type_id=CONFIG["gpu_type"],
            cloud_type="SECURE",
            gpu_count=1,
            volume_in_gb=CONFIG["volume_size_gb"],
            container_disk_in_gb=CONFIG["container_disk_gb"],
            ports=f"{CONFIG['service_port']}/http",
            volume_mount_path="/data",
            env={
                "API_AUTH_TOKEN": API_AUTH_TOKEN,
                "FAISS_INDEX_PATH": FAISS_INDEX,
                "SERVICE_TYPE": "image",
                "PORT": str(CONFIG["service_port"])
            },
            support_public_ip=True,
            min_vcpu_count=CONFIG["min_vcpu"],
            min_memory_in_gb=CONFIG["min_memory_gb"],
            bid_percent=50
        )

        # Handle response format
        pod_id = pod.id if hasattr(pod, 'id') else pod['data']['podFindAndDeployOnDemand']['id']
        
        # Wait for pod to be ready
        print(f"🔄 Waiting for pod {pod_id} to be ready...")
        for _ in range(10):  # Wait up to 5 minutes
            sleep(30)
            current_pod = runpod.get_pod(pod_id)
            if current_pod and current_pod.status in ("RUNNING", "RESUMED"):
                public_ip = current_pod.public_ip
                print(f"✅ Pod ready at {public_ip}")
                print(f"🚀 Image service available at: http://{public_ip}:{CONFIG['service_port']}")
                sys.exit(0)
        
        print(f"⚠️ Pod not ready after waiting. Status: {current_pod.status if current_pod else 'unknown'}")
        runpod.terminate_pod(pod_id)
        
    except Exception as e:
        print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
        if attempt < CONFIG["max_retries"] - 1:
            print(f"🔄 Retrying in {CONFIG['retry_delay']} seconds...")
            sleep(CONFIG["retry_delay"])

sys.exit("❌ Failed to create pod after multiple attempts")