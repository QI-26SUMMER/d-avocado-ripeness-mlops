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
  {"instances": [{"b64": "<base64-encoded jpg/png bytes>",
                  "target_stage": 4,        # optional, per-image; enables days_to_target
                  "temp_celsius": 21.0,      # optional, per-image; user's storage temp
                  "storage_group": "T20"}],  # optional; discrete-α fallback if no temp
   "parameters": {"target_stage": 4, "temp_celsius": 21.0}}  # optional shared defaults
  A per-instance field overrides the same field in "parameters".

Response: {"predictions": [{"predicted_stage": 3, "label": "Ripe(1)", "hint": "Ready to eat",
                             "confidence": 0.82, "stage_probs": [.., .., .., .., ..],
                             "model_version": "P1_general_resnet18_paper_aug_oversample",
                             "days_to_target": 4.4,           # only if target_stage sent
                             "estimated_peak_date": "2026-07-25",
                             "days_left": 2.3}, ...]}          # only if storage_group sent
  Field names match the Spring backend's DB columns (predicted_stage, stage_probs,
  confidence, model_version, days_to_target, estimated_peak_date) so it can store the
  response as-is. stage_probs is a LIST indexed 0..4 for stage 1..5 (not a dict).

  days_to_target = days until the CURRENT stage reaches the user's target_stage,
  = α × (predicted_stage − target_stage). Emitted only when target_stage is given
  (absent -> field omitted -> backend stores null). Needs an α source: temp_celsius
  (continuous, interpolated - see shelf_life.alpha_from_temp, ⚠ provisional) or
  storage_group (discrete paper coefficients).

  days_left is the SEPARATE paper-reproduction metric (days until stage 5); it is
  NOT days_to_target and the two must never be conflated (different endpoints).
  Per-image failures return {"error": "..."} -> backend maps to NO_AVOCADO_DETECTED.
"""
from __future__ import annotations

import base64
import io
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import LABELS  # noqa: E402
from models import build_model  # noqa: E402
from shelf_life import (  # noqa: E402
    ALPHA_5STAGE,
    alpha_from_temp,
    estimate_days_left,
    estimate_days_to_target,
)
from transforms import evaluation_transform  # noqa: E402

STAGE_HINT = {1: "Unripe (firm)", 2: "Breaking", 3: "Ready to eat",
              4: "Peak (end of shelf life)", 5: "Overripe (too late)"}

# estimated_peak_date is a calendar date the Korean end-user reads, so anchor it
# to KST rather than the Cloud Run container's UTC clock (up to ~9h / one calendar
# day off at midnight otherwise). Fixed offset avoids a tzdata dependency.
KST = timezone(timedelta(hours=9))

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


def _predict_one(image_bytes: bytes, storage_group: str | None,
                 target_stage=None, temp_celsius=None) -> dict:
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
            # Days until stage 5 (this repo's paper-reproduction endpoint), NOT the
            # backend's days_to_target. Kept for parity with src/predict.py; the
            # backend ignores it. Do not rename this to days_to_target.
            out["days_left"] = estimate_days_left(stage, storage_group)

    # days_to_target: days until the CURRENT stage reaches the user's target_stage.
    # Only emitted when target_stage is explicitly given (agreed contract fallback:
    # absent target_stage -> field omitted -> backend stores null).
    if target_stage is not None:
        _add_days_to_target(out, stage, target_stage, temp_celsius, storage_group)
    return out


def _add_days_to_target(out: dict, stage: int, target_stage,
                        temp_celsius, storage_group: str | None) -> None:
    """Fill out['days_to_target'] + ['estimated_peak_date'], or a *_error field.

    α source, in order: temp_celsius (continuous, provisional interp) → storage_group
    (discrete paper coefficients). estimated_peak_date = KST today + round(days).
    """
    try:
        ts = int(target_stage)
    except (TypeError, ValueError):
        out["days_to_target_error"] = f"target_stage must be an int 1-5, got {target_stage!r}"
        return
    if not 1 <= ts <= 5:
        out["days_to_target_error"] = f"target_stage out of range 1-5: {ts}"
        return

    if temp_celsius is not None:
        try:
            alpha = alpha_from_temp(float(temp_celsius))
            basis = f"temp_interp(provisional):{float(temp_celsius):g}C"
        except (TypeError, ValueError):
            out["days_to_target_error"] = f"temp_celsius must be numeric, got {temp_celsius!r}"
            return
    elif storage_group and storage_group in ALPHA_5STAGE:
        alpha = ALPHA_5STAGE[storage_group]
        basis = f"storage:{storage_group}"
    else:
        out["days_to_target_error"] = "days_to_target needs temp_celsius or a valid storage_group"
        return

    days = estimate_days_to_target(stage, ts, alpha)
    out["days_to_target"] = days
    out["estimated_peak_date"] = (datetime.now(KST).date() + timedelta(days=round(days))).isoformat()
    out["days_to_target_basis"] = basis  # debug only; backend ignores unknown fields


@app.post(PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()
    instances = body.get("instances", [])
    params = body.get("parameters") or {}

    predictions = []
    for inst in instances:
        try:
            inst = inst if isinstance(inst, dict) else {"b64": inst}
            image_bytes = base64.b64decode(inst["b64"])
            # Per-instance value wins over the shared `parameters` default, so a batch
            # may carry per-image target_stage/temp while still allowing one global setting.
            storage_group = inst.get("storage_group", params.get("storage_group"))
            target_stage = inst.get("target_stage", params.get("target_stage"))
            temp_celsius = inst.get("temp_celsius", params.get("temp_celsius"))
            predictions.append(
                _predict_one(image_bytes, storage_group, target_stage, temp_celsius))
        except Exception as e:
            predictions.append({"error": str(e)})
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT)
