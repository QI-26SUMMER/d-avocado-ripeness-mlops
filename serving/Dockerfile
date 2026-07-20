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

# Only the modules the server actually imports (model def, transforms, labels, shelf-life).
COPY src/__init__.py src/data.py src/models.py src/transforms.py src/shelf_life.py src/
COPY serving/app.py serving/app.py

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
