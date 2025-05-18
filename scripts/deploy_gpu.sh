#!/usr/bin/env python3
import os
import sys
import subprocess
import runpod
from datetime import datetime, timedelta

CONFIG = {
    "service_name": "multi-rag-image-service",
    "service_port": 8000,
    "gpu_type": "NVIDIA GeForce RTX 3080 Ti",
    "volume_size_gb": 50,
    "container_disk_gb": 20,
    "min_vcpu": 8,
    "min_memory_gb": 30,
    "pod_lifetime_minutes": 60
}

def build_and_push_image():
    """Build and push Docker image using GitHub Secrets"""
    image = f"ghcr.io/{os.environ['GHCR_USER'].lower()}/faiss-gpu-api:latest"
    
    subprocess.run([
        "docker", "build", "-f", "docker/Dockerfile.gpu",
        "-t", image, "."
    ], check=True)
    
    subprocess.run(["docker", "push", image], check=True)
    return image

def terminate_existing_pods():
    """Clean up any existing pods"""
    print("🔍 Checking for existing pods...")
    for pod in runpod.get_pods():
        pod_id = pod.get('id') if isinstance(pod, dict) else getattr(pod, 'id', None)
        if pod_id and (pod.get('name') == CONFIG["service_name"] or 
                      getattr(pod, 'name', None) == CONFIG["service_name"]):
            print(f"🗑️ Terminating existing pod {pod_id}")
            runpod.terminate_pod(pod_id)

def deploy_pod(image):
    """Deploy new pod with all required secrets"""
    terminate_time = datetime.now() + timedelta(minutes=CONFIG["pod_lifetime_minutes"])
    
    print("🚀 Deploying new pod...")
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
            # Required core variables
            "API_AUTH_TOKEN": os.environ["API_AUTH_TOKEN"],
            "FAISS_INDEX_PATH": os.environ["FAISS_INDEX_PATH"],
            "TERMINATE_AT": terminate_time.isoformat(),
            
            # All other secrets your app needs
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
            
            # Add any other required environment variables
            "SERVICE_TYPE": "image",
            "PORT": str(CONFIG["service_port"])
        },
        support_public_ip=True,
        min_vcpu_count=CONFIG["min_vcpu"],
        min_memory_in_gb=CONFIG["min_memory_gb"]
    )
    
    pod_id = pod.id if hasattr(pod, 'id') else pod['id']
    print(f"✅ Pod deployed successfully. ID: {pod_id}")
    print(f"⏰ Will auto-terminate at: {terminate_time}")
    
    return pod_id

if __name__ == "__main__":
    # Verify required secrets
    required_secrets = ["GHCR_USER", "GHCR_TOKEN", "RUNPOD_API_KEY", 
                       "API_AUTH_TOKEN", "FAISS_INDEX_PATH"]
    missing = [secret for secret in required_secrets if secret not in os.environ]
    if missing:
        sys.exit(f"❌ Missing required secrets: {', '.join(missing)}")

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    
    try:
        image = build_and_push_image()
        terminate_existing_pods()
        deploy_pod(image)
    except Exception as e:
        sys.exit(f"❌ Deployment failed: {str(e)}")