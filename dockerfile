# Real-ESRGAN Dockerfile — GPU inference
# Requires: NVIDIA GPU + nvidia-container-toolkit on the host
# Build:  docker build -t real-esrgan .
# Run:    docker run --gpus all \
#           -v /path/to/inputs:/app/inputs \
#           -v /path/to/results:/app/results \
#           real-esrgan \
#           python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs -o results

# ── Base image ────────────────────────────────────────────────────────────────
# CUDA 11.8 + cuDNN 8, Ubuntu 22.04. Matches PyTorch 2.x wheels.
# For CPU-only: replace with python:3.10-slim and remove all CUDA references.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ── System deps ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-dev \
        git \
        wget \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Workdir ───────────────────────────────────────────────────────────────────
WORKDIR /app

# ── Clone repo ────────────────────────────────────────────────────────────────
RUN git clone https://github.com/xinntao/Real-ESRGAN.git .

# ── Python deps ───────────────────────────────────────────────────────────────
# Install PyTorch first with the matching CUDA index to avoid CPU-only fallback.
RUN pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies exactly as the README specifies.
RUN pip install --no-cache-dir basicsr facexlib gfpgan
RUN pip install --no-cache-dir -r requirements.txt
RUN python setup.py develop

# ── Download default model weights ────────────────────────────────────────────
# Pre-bake the most common model so the container is self-contained.
# Comment out any you don't need to keep the image smaller.
RUN mkdir -p weights && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
        -O weights/RealESRGAN_x4plus.pth && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth \
        -O weights/RealESRGAN_x4plus_anime_6B.pth

# ── I/O directories ───────────────────────────────────────────────────────────
RUN mkdir -p inputs results
VOLUME ["/app/inputs", "/app/results"]

# ── Default command ───────────────────────────────────────────────────────────
# Override at runtime with any inference_realesrgan.py flags you need.
#CMD ["python", "inference_realesrgan.py", \
#     "-n", "RealESRGAN_x4plus", \
#     "-i", "inputs", \
#     "-o", "results"]
