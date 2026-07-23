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

# Background-removal crop (ENABLE_CROP=1): rembg/u2net deps. Pre-download the u2net model at build
# time so the first scan after a cold start doesn't pay the ~176 MB download (best-effort — falls
# back to a runtime download if the fetch fails during the build).
COPY requirements-preprocess.txt .
RUN pip install -r requirements-preprocess.txt
RUN python -c "from rembg import new_session; new_session('u2net')" \
    || echo "[build] u2net pre-download skipped — will fetch at runtime"

# Only the modules the server actually imports (model def, transforms, labels, shelf-life, crop).
COPY src/__init__.py src/data.py src/models.py src/transforms.py src/shelf_life.py src/preprocess.py src/
COPY serving/app.py serving/app.py

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
