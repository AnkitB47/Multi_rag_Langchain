# ──── BASE CUDA IMAGE ───────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV TMPDIR=/tmp
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ──── BUILDER STAGE ─────────────────────────────────────────────────
FROM base as builder

WORKDIR /app

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch with CUDA 11.8 first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install FAISS-GPU (official version)
RUN pip install --no-cache-dir faiss-gpu==1.7.2

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# ──── RUNTIME STAGE ─────────────────────────────────────────────────
FROM base as runtime

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY ./src /app/src
COPY .streamlit /app/.streamlit

# Create /tmp and set permissions
RUN mkdir -p /tmp && chmod 777 /tmp

# Expose Streamlit port
EXPOSE 8501

# Launch command
CMD ["streamlit", "run", "src/langgraphagenticai/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]