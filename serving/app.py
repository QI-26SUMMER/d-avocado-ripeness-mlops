"""Vertex AI custom-container prediction server for the avocado ripeness model.

Separate from the training container (../Dockerfile): this one starts an HTTP
server instead of running the training pipeline, matching what Vertex AI
Endpoints actually require (health + predict routes), and downloads the model
artifacts from GCS itself since Endpoints do NOT get the automatic /gcs mount
that Custom Training Jobs get.

Vertex AI sets these env vars at deploy time:
  AIP_STORAGE_URI   gs:// URI of the model artifact directory (same as the
                    --artifact-uri given at `gcloud ai models upload`)
  AIP_HTTP_PORT     port to listen on
  AIP_HEALTH_ROUTE  health-check path
  AIP_PREDICT_ROUTE prediction path

Request body (Vertex custom-container contract, {"instances": [...]}):
  {"instances": [{"b64": "<base64-encoded jpg/png bytes>"}, ...],
   "parameters": {"storage_group": "T20"}}   # optional, applies to all instances

Response: {"predictions": [{"predicted_stage": 3, "label": "Ripe(1)", "hint": "Ready to eat",
                             "confidence": 0.82, "stage_probs": [.., .., .., .., ..],
                             "model_version": "P1_general_resnet18_paper_aug_oversample",
                             "days_left": 2.3}, ...]}
  Field names match the Spring backend's DB columns (predicted_stage, stage_probs,
  confidence, model_version) so it can store the response as-is. stage_probs is a
  list indexed 0..4 for stage 1..5. days_left is only present when "storage_group"
  is given, and is this repo's days-until-stage-5 metric - NOT the backend's
  days_to_target (days until the user's chosen target stage); see the comment
  above its assignment in _predict_one.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import LABELS  # noqa: E402
from models import build_model  # noqa: E402
from shelf_life import ALPHA_5STAGE, estimate_days_left  # noqa: E402
from transforms import evaluation_transform  # noqa: E402

STAGE_HINT = {1: "Unripe (firm)", 2: "Breaking", 3: "Ready to eat",
              4: "Peak (end of shelf life)", 5: "Overripe (too late)"}

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
HTTP_PORT = int(os.environ.get("AIP_HTTP_PORT", "8080"))
STORAGE_URI = os.environ.get("AIP_STORAGE_URI")
LOCAL_MODEL_DIR = Path(os.environ.get("LOCAL_MODEL_DIR", "/tmp/model"))

app = FastAPI()
_state: dict = {"model": None, "transform": None, "device": "cpu", "cfg": None}


def _download_artifacts(gcs_uri: str, dest: Path) -> None:
    """Download every blob under gcs_uri (a gs://bucket/prefix dir) into dest.

    No /gcs/ auto-mount exists for Endpoints (unlike Custom Training Jobs),
    so the container has to fetch the artifacts itself.
    """
    from google.cloud import storage

    assert gcs_uri.startswith("gs://"), f"AIP_STORAGE_URI must start with gs://, got {gcs_uri!r}"
    bucket_name, _, prefix = gcs_uri[5:].partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    dest.mkdir(parents=True, exist_ok=True)
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No objects found under {gcs_uri}")
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix):].lstrip("/")
        out_path = dest / (rel or Path(blob.name).name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(out_path))


def load_model() -> None:
    """Download (if needed) and load the checkpoint. Raises on failure (kept
    unhealthy) rather than silently falling back — a wrong/missing model
    must not look like it's serving fine."""
    ckpt_path = LOCAL_MODEL_DIR / "best.pt"
    if not ckpt_path.exists():
        if not STORAGE_URI:
            raise RuntimeError("AIP_STORAGE_URI not set and no local checkpoint found")
        _download_artifacts(STORAGE_URI, LOCAL_MODEL_DIR)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best.pt not found under {LOCAL_MODEL_DIR} after download")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg["model"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    _state["model"] = model
    _state["transform"] = evaluation_transform(cfg.get("img_size", 224))
    _state["device"] = device
    _state["cfg"] = cfg


@app.on_event("startup")
def _startup() -> None:
    load_model()


@app.get(HEALTH_ROUTE)
def health():
    if _state["model"] is None:
        return JSONResponse({"status": "not ready"}, status_code=503)
    return {"status": "ok", "experiment_id": _state["cfg"]["experiment_id"]}


def _predict_one(image_bytes: bytes, storage_group: str | None) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _state["transform"](img).unsqueeze(0).to(_state["device"])
    with torch.no_grad():
        probs = torch.softmax(_state["model"](x), 1)[0].cpu().numpy()
    stage = int(probs.argmax()) + 1
    out = {
        "predicted_stage": stage,
        "label": LABELS[stage],
        "hint": STAGE_HINT[stage],
        "confidence": float(probs[stage - 1]),
        "stage_probs": [float(probs[k - 1]) for k in range(1, 6)],
        "model_version": _state["cfg"]["experiment_id"],
    }
    if storage_group:
        if storage_group not in ALPHA_5STAGE:
            out["shelf_life_error"] = f"unknown storage_group {storage_group!r} (expected T10|T20|Tam|Tamb)"
        else:
            # Days until stage 5 (this repo's shelf-life endpoint), NOT the backend's
            # days_to_target (days until the user's chosen target stage). The
            # days-until-target calculation is a separate, still-undecided design
            # question - out of scope here; do not rename this to days_to_target.
            out["days_left"] = estimate_days_left(stage, storage_group)
    return out


@app.post(PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()
    instances = body.get("instances", [])
    params = body.get("parameters") or {}
    storage_group = params.get("storage_group")

    predictions = []
    for inst in instances:
        try:
            b64 = inst["b64"] if isinstance(inst, dict) else inst
            image_bytes = base64.b64decode(b64)
            predictions.append(_predict_one(image_bytes, storage_group))
        except Exception as e:
            predictions.append({"error": str(e)})
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT)
