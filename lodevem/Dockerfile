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

# PyTorch CPU-only — significantly smaller than the CUDA build
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# container_measure.py lives alongside this Dockerfile inside the package
COPY container_measure.py /app/container_measure.py

WORKDIR /app

ENTRYPOINT ["python", "/app/container_measure.py"]
