.PHONY: dev lint deploy-pdf deploy-gpu deploy

dev:
	streamlit run src/langgraphagenticai/ui/app.py

lint:
	flake8 src/

# Deploy the CPU‐based PDF service to Fly.io
deploy-pdf:
	flyctl deploy --remote-only

# Deploy the GPU image‐search service to RunPod
deploy-gpu:
	./scripts/deploy_gpu.sh

# Convenience: do both
deploy: deploy-pdf deploy-gpu
