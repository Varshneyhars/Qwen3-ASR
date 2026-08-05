# RunPod Serverless image. For Google Cloud Run, build Dockerfile.cloudrun
# instead — same core, HTTP server instead of the RunPod queue worker.
#
# Base: PyTorch 2.8.0 + CUDA 12.8 + cuDNN 9. Qwen3-ASR needs transformers>=5.13
# (torch>=2.4), and CUDA 12.8 covers Turing through Blackwell, so the image runs
# on every GPU RunPod Serverless offers.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface \
    QWEN_MODEL_ID=Qwen/Qwen3-ASR-1.7B-hf \
    QWEN_ALIGNER_MODEL_ID=Qwen/Qwen3-ForcedAligner-0.6B-hf \
    QWEN_PRECISION=bfloat16 \
    QWEN_ENABLE_ALIGNER=true \
    DO_NOT_TRACK=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

# ffmpeg does all container/codec normalization to 16 kHz mono; libsndfile
# backs soundfile; libgomp is required by numba/librosa.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-runpod.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-runpod.txt

COPY asr_core.py .
COPY handler.py .
COPY test_input.json .

# Optional: pre-download weights into the image so cold starts don't block on
# HF. ~4.1 GB for the ASR model + ~1.8 GB for the aligner. Set false to lazy-
# download onto a RunPod network volume instead.
ARG PREFETCH_MODEL=true
ARG MODEL_ID=Qwen/Qwen3-ASR-1.7B-hf
ARG ALIGNER_MODEL_ID=Qwen/Qwen3-ForcedAligner-0.6B-hf
ARG PREFETCH_ALIGNER=true
RUN if [ "$PREFETCH_MODEL" = "true" ]; then \
        python -c "from huggingface_hub import snapshot_download; \
snapshot_download('$MODEL_ID', cache_dir='/opt/hf-cache')" && \
        echo "Prefetched $MODEL_ID"; \
    fi && \
    if [ "$PREFETCH_MODEL" = "true" ] && [ "$PREFETCH_ALIGNER" = "true" ]; then \
        python -c "from huggingface_hub import snapshot_download; \
snapshot_download('$ALIGNER_MODEL_ID', cache_dir='/opt/hf-cache')" && \
        echo "Prefetched $ALIGNER_MODEL_ID"; \
    fi

ENV HF_HOME=/opt/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache \
    QWEN_MODEL_ID=${MODEL_ID} \
    QWEN_ALIGNER_MODEL_ID=${ALIGNER_MODEL_ID}

CMD ["python", "-u", "handler.py"]
