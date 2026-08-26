# =============================================================================
# lodevem benchmark container
# =============================================================================
# This Dockerfile is bundled inside the lodevem Python package.
# It gets used automatically when Docker mode is active.
# Build context is the lodevem/ package directory.
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# PyTorch CPU-only and ONNX Runtime
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    onnxruntime==1.17.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the entire lodevem package so container_measure can access lodevem.backends
COPY . /app/lodevem

WORKDIR /app

ENTRYPOINT ["python", "-m", "lodevem.container_measure"]
