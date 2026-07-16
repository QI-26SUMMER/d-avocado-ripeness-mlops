# Avocado ripeness model training image (Vertex AI Custom Training)
#
# Reproducibility: torch/torchvision are pinned to the same cu124 wheels as requirements.txt.
#   The torch cu124 wheels bundle the CUDA userspace, so no separate CUDA base image is needed
#   (it works as long as the Vertex GPU node has the NVIDIA driver).
# The data images are not in the repo → the container unzips a single zip from GCS at startup (docker/entrypoint.sh).
# Python 3.12: numpy==2.5.1 in requirements.txt requires Python>=3.12 (3.11 cannot install it).
# torch 2.6.0+cu124 provides a cp312 wheel.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# unzip: for extracting the dataset zip from the /gcs mount
RUN apt-get update && apt-get install -y --no-install-recommends unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1) Install torch/torchvision first from the cu124 index (same versions as the header comment in requirements.txt)
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 2) Remaining pure-Python dependencies (torch/torchvision already satisfied → pip skips them)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Copy the repository (.dockerignore excludes images/outputs/.git; the xlsx under data/ is included)
COPY . .

RUN chmod +x docker/entrypoint.sh
ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
