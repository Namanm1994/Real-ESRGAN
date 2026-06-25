# ════════════════════════════════════════════════════════════════════════════
# Dockerfile v2
#
# Repos included:
#   v1 — Real-ESRGAN  (xinntao/Real-ESRGAN)
#   v2 — RealBasicVSR (ckkelvinchan/RealBasicVSR)  ← added
#
# ⚠ DEPENDENCY NOTE
# Real-ESRGAN uses basicsr + its own pip stack.
# RealBasicVSR uses mmedit (MMEditing) which pins mmcv-full<=1.6.0.
# These two stacks DO NOT share runtime state; treat them as separate tools
# installed side-by-side. Do not mix their Python APIs in custom scripts.
#
# Build:
#   docker build -t sr-toolkit:v2 .
#
# Run Real-ESRGAN (image upscaling):
#   docker run --gpus all \
#     -v /path/to/inputs:/app/real-esrgan/inputs \
#     -v /path/to/results:/app/real-esrgan/results \
#     sr-toolkit:v2 \
#     bash -c "cd /app/real-esrgan && python inference_realesrgan.py \
#       -n RealESRGAN_x4plus -i inputs -o results"
#
# Run RealBasicVSR (video super-resolution):
#   docker run --gpus all \
#     -v /path/to/input_video:/app/realbasicvsr/data/input.mp4 \
#     -v /path/to/results:/app/realbasicvsr/results \
#     sr-toolkit:v2 \
#     bash -c "cd /app/realbasicvsr && python inference_realbasicvsr.py \
#       configs/realbasicvsr_x4.py checkpoints/RealBasicVSR_x4.pth \
#       data/input.mp4 results/output.mp4 --fps=24"
# ════════════════════════════════════════════════════════════════════════════

# ── Base image ────────────────────────────────────────────────────────────────
# CUDA 11.3 chosen for mmcv-full compat (mmcv-full<=1.6.0 builds against cu113).
# Real-ESRGAN is fine with cu113 PyTorch as well.
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# ── System deps ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3-pip \
        python3.8-dev \
        git \
        wget \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch (shared, cu113) ───────────────────────────────────────────────────
# mmcv-full 1.6.x requires torch>=1.7; Real-ESRGAN requires torch>=1.7.
# torch 1.12 + cu113 is the highest torch that mmcv-full 1.6.x supports.
RUN pip install --no-cache-dir \
        torch==1.12.1+cu113 \
        torchvision==0.13.1+cu113 \
        torchaudio==0.12.1 \
        --extra-index-url https://download.pytorch.org/whl/cu113

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Real-ESRGAN
# ══════════════════════════════════════════════════════════════════════════════
WORKDIR /app/real-esrgan

RUN git clone https://github.com/xinntao/Real-ESRGAN.git .

RUN pip install --no-cache-dir basicsr facexlib gfpgan && \
    pip install --no-cache-dir -r requirements.txt && \
    python setup.py develop

# Pre-download default model weights
RUN mkdir -p weights && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
        -O weights/RealESRGAN_x4plus.pth && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth \
        -O weights/RealESRGAN_x4plus_anime_6B.pth

RUN mkdir -p inputs results
VOLUME ["/app/real-esrgan/inputs", "/app/real-esrgan/results"]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RealBasicVSR
# ══════════════════════════════════════════════════════════════════════════════
WORKDIR /app/realbasicvsr

RUN git clone https://github.com/ckkelvinchan/RealBasicVSR.git .

# openmim is the MMLab package manager; used to install mmcv-full with CUDA wheels.
# mmcv-full must be <=1.6.0 (mmedit hard constraint).
# mmedit 0.15.x is the last release that uses the mmcv-full 1.x API.
RUN pip install --no-cache-dir openmim && \
    mim install "mmcv-full>=1.3.13,<=1.6.0" && \
    pip install --no-cache-dir "mmedit==0.15.2"

# checkpoints directory — user must supply the weights (too large to bake in:
# RealBasicVSR_x4.pth is ~300MB). Download instructions:
#   wget "https://www.dropbox.com/s/eufigxmmkv5woop/RealBasicVSR.pth" \
#       -O checkpoints/RealBasicVSR_x4.pth
# or use the Google Drive / OneDrive links from the README.
RUN mkdir -p checkpoints data results
VOLUME ["/app/realbasicvsr/checkpoints", \
        "/app/realbasicvsr/data", \
        "/app/realbasicvsr/results"]

# ── Default workdir and command ───────────────────────────────────────────────
WORKDIR /app

# No single default CMD makes sense for a multi-tool image.
# Override at runtime (see usage examples at the top of this file).
CMD ["bash"]
