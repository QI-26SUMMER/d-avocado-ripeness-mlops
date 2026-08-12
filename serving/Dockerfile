# Vertex AI Endpoint prediction container. Separate from ../Dockerfile (training):
# that one runs the training pipeline as its ENTRYPOINT, which is why the first
# deploy attempt crash-looped (Endpoints expect an HTTP predict/health server,
# and there is no /gcs/ auto-mount outside Custom Training Jobs).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY serving/requirements-serving.txt .
RUN pip install -r requirements-serving.txt

# Background-removal crop (ENABLE_CROP=1): InSPyReNet (default) + rembg/u2net (fallback) deps.
# Pre-download both models at build time so the first scan after a cold start doesn't pay the
# download (~370 MB InSPyReNet + ~176 MB u2net) on the request path (best-effort — falls back to
# a runtime download if a fetch fails during the build). TRANSPARENT_BACKGROUND_FILE_PATH pins the
# checkpoint cache dir so the build-time download location matches where the app looks at runtime.
ENV TRANSPARENT_BACKGROUND_FILE_PATH=/workspace
COPY requirements-preprocess.txt .
RUN pip install -r requirements-preprocess.txt
# transparent-background pulls the GUI build of opencv-python, which needs X11 libs
# (libxcb.so.1 etc.) that python:3.12-slim doesn't have -> `import cv2` crashes the container at
# startup (confirmed via a failed Cloud Run deploy). Swap in opencv-python-headless: same cv2 API
# for our usage (resize/blur/color-convert, no GUI calls), no system libs needed.
# PINNED to 4.10.0.84, NOT latest: opencv-python-headless 5.0.0.93 (latest as of this writing) has
# a broken wheel -- pip reports it installed but site-packages/cv2/ ends up with no native module,
# so `import cv2` raises ModuleNotFoundError (reproduced identically in two separate environments;
# --force-reinstall --no-deps confirmed it isn't a stale-cache issue). 4.10.0.84 installs cleanly.
RUN pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall --no-deps "opencv-python-headless==4.10.0.84"
RUN python -c "from transparent_background import Remover; Remover(mode='base')" \
    || echo "[build] InSPyReNet pre-download skipped — will fetch at runtime"
RUN python -c "from rembg import new_session; new_session('u2net')" \
    || echo "[build] u2net pre-download skipped — will fetch at runtime"

# Only the modules the server actually imports (labels, model def, transforms, shelf-life, crop).
# One COPY per package directory, NOT one flat COPY: with multiple sources Docker copies each file
# in by basename and flattens the tree, which would put common/labels.py at src/labels.py and break
# `from common.labels import ...`.
# src/data/ is deliberately absent — the serving image needed it only for LABELS, which now lives
# in common/labels.py, so pandas-adjacent dataset code no longer ships to production.
COPY src/__init__.py src/
COPY src/common/__init__.py src/common/labels.py src/common/models.py \
     src/common/transforms.py src/common/shelf_life.py src/common/
COPY src/inference/__init__.py src/inference/preprocess.py src/inference/
COPY serving/app.py serving/app.py

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
