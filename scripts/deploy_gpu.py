import os
import sys
import subprocess
import runpod
from datetime import datetime, timedelta

CONFIG = {
    "service_name": "multi-rag-image-service",
    "service_port": 8000,
    "gpu_type": "NVIDIA RTX A5000",
    "volume_size_gb": 50,
    "container_disk_gb": 20,
    "min_vcpu": 8,
    "min_memory_gb": 30,
    "pod_lifetime_minutes": 60
}

def verify_ghcr_access():
    """Verify GHCR credentials and image accessibility"""
    image_name = os.environ["IMAGE_NAME"].lower()  # Force lowercase
    try:
        subprocess.run(
            ["docker", "pull", image_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ Verified GHCR image accessibility")
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to access GHCR image: {str(e)}")

def terminate_existing_pods():
    """Clean up any existing pods with matching name"""
    print("🔍 Checking for existing pods...")
    for pod in runpod.get_pods():
        pod_id = pod.get('id') if isinstance(pod, dict) else getattr(pod, 'id', None)
        if pod_id and (pod.get('name') == CONFIG["service_name"] or 
                      getattr(pod, 'name', None) == CONFIG["service_name"]):
            print(f"🗑️ Terminating existing pod {pod_id}")
            runpod.terminate_pod(pod_id)

def deploy_pod():
    """Deploy new pod with all required configuration"""
    terminate_time = datetime.now() + timedelta(minutes=CONFIG["pod_lifetime_minutes"])
    image_name = os.environ["IMAGE_NAME"].lower()  # Force lowercase
    
    print("🚀 Deploying new pod...")
    pod = runpod.create_pod(
        name=CONFIG["service_name"],
        image_name=image_name,
        container_registry_auth={
            "username": os.environ["GHCR_USER"],
            "password": os.environ["GHCR_TOKEN"],
            "server": "ghcr.io"
        },
        gpu_type_id=CONFIG["gpu_type"],
        cloud_type="SECURE",
        gpu_count=1,
        volume_in_gb=CONFIG["volume_size_gb"],
        container_disk_in_gb=CONFIG["container_disk_gb"],
        ports=f"{CONFIG['service_port']}/http",
        volume_mount_path="/data",
        env={
            "API_AUTH_TOKEN": os.environ["API_AUTH_TOKEN"],
            "FAISS_INDEX_PATH": os.environ["FAISS_INDEX_PATH"],
            "TERMINATE_AT": terminate_time.isoformat(),
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
    # Verify required environment variables
    required_vars = [
        "GHCR_USER", "GHCR_TOKEN", "RUNPOD_API_KEY",
        "API_AUTH_TOKEN", "FAISS_INDEX_PATH", "IMAGE_NAME"
    ]
    missing = [var for var in required_vars if var not in os.environ]
    if missing:
        sys.exit(f"❌ Missing required environment variables: {', '.join(missing)}")

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    
    try:
        verify_ghcr_access()
        terminate_existing_pods()
        deploy_pod()
    except Exception as e:
        sys.exit(f"❌ Deployment failed: {str(e)}")