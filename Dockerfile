# =============================================================================
# lodevem benchmark container
# =============================================================================
#
# What this file is for:
#   This defines the environment inside the Docker container that lodevem
#   spins up to measure memory usage. The container is where the actual
#   "this is a Nokia C1 with 1GB RAM" constraint gets enforced.
#
# Why a separate container?
#   We can't safely cap the memory of the main Python process — your terminal
#   session, VS Code, and other tools are running there too. Docker lets us
#   create an isolated process with a hard memory ceiling that only affects
#   the benchmark, not your system.
#
# Image size note:
#   PyTorch CPU-only (no CUDA) is ~500MB installed. We use the CPU-only
#   version because our target devices don't have NVIDIA GPUs — and because
#   we want to measure CPU inference, which is what those phones do.
# =============================================================================

FROM python:3.11-slim

# Keeps Python from writing .pyc files to disk (cleaner container)
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python output unbuffered so we see logs immediately
ENV PYTHONUNBUFFERED=1

# Install PyTorch CPU-only — much smaller than the full CUDA build
# (We pin to a stable version for reproducibility)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy only the measurement script into the container.
# The model file gets mounted as a volume at runtime (see measure.py).
COPY lodevem/container_measure.py /app/container_measure.py

WORKDIR /app

# The entry point runs the measurement script with the model path as argument.
# Example: docker run ... benchmark_runner /models/cocoa_int8.pt 50
ENTRYPOINT ["python", "/app/container_measure.py"]
