# GroundedGuard API image.
# UNTESTED: docker could not run on the machine this was written on (snap docker refuses a home
# directory outside /home), so neither the build nor the base-image tag below has been verified.
#
# No model weights in the image. Mount at run time (read-only):
#   checkpoints/                          -> /app/checkpoints
#   outputs/module5/frozen_choice.json    -> /app/outputs/module5/frozen_choice.json   (DistilBERT threshold)
#   outputs/module5b/frozen_choice.json   -> /app/outputs/module5b/frozen_choice.json  (ModernBERT threshold)
#
#   docker build -t groundedguard-api .
#   docker run --gpus '"device=0"' -p 8000:8000 \
#     -v "$PWD/checkpoints:/app/checkpoints:ro" \
#     -v "$PWD/outputs/module5/frozen_choice.json:/app/outputs/module5/frozen_choice.json:ro" \
#     -v "$PWD/outputs/module5b/frozen_choice.json:/app/outputs/module5b/frozen_choice.json:ro" \
#     groundedguard-api

# PyTorch 2.11.0 with CUDA 12.8 to match the torch==2.11.0+cu128 the model was trained and tested with.
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    GG_MODEL=modernbert \
    GG_PRECISION=fp32 \
    GG_DEVICE=auto \
    GG_LOG_CONTENT=false

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY groundedguard/ groundedguard/
COPY api/ api/
COPY configs/inference.yaml configs/inference.yaml

RUN useradd --create-home --uid 1000 gg && chown -R gg /app
USER gg

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
# One worker: one model per process; GPU calls are serialized inside the process.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
