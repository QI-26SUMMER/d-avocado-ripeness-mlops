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

Request body (Vertex custom-container contract, {"instances": [...]}), fixed to
match the Spring backend exactly:
  {"instances": [{"b64": "<base64-encoded jpg bytes>"}],
   "parameters": {"target_stage": 4, "temp_celsius": 17.0}}
  target_stage is an int 1-5, always sent by the backend. temp_celsius is a
  float and may be absent, in which case it defaults to 20.0°C.

Response: {"predictions": [{"predicted_stage": 3, "label": "Ripe(1)", "hint": "Ready to eat",
                             "confidence": 0.82, "stage_probs": [.., .., .., .., ..],
                             "model_version": "P1_general_resnet18_paper_aug_oversample",
                             "days_to_target": 4.4,           # only if target_stage sent
                             "cropped_b64": "<base64 jpg>"}, ...]}  # only if ENABLE_CROP=1
  Field names match the Spring backend's DB columns (predicted_stage, stage_probs,
  confidence, model_version, days_to_target) so it can store the response as-is.
  stage_probs is a LIST indexed 0..4 for stage 1..5 (not a dict). The backend
  derives estimated_peak_date itself; this service does NOT return it.

  days_to_target = α × (predicted_stage − target_stage), NOT clipped at 0 (negative
  means already past target, i.e. overripe), rounded to 1 decimal place (the DB
  column is NUMERIC(4,1)). α comes from shelf_life.alpha_from_temp(temp_celsius)
  (Q10 log-linear interpolation, see shelf_life.py). Emitted only when target_stage
  is given (absent -> field omitted -> backend stores null).

  cropped_b64 (method A): when ENABLE_CROP=1, the image is background-removal cropped
  (src/preprocess.py, rembg by default) BEFORE classification, and the 224px crop is
  returned as raw base64 JPEG. The backend saves it (cropped/{user_id}/{scan_id}.jpg)
  and sets images.cropped_url. Absent when cropping is off (current CPU deploy) ->
  backend leaves cropped_url null.
  Per-image failures return {"error": "..."} -> backend maps to NO_AVOCADO_DETECTED.

Classifier backend (MODEL_BACKEND env, internal — does NOT change the wire contract above):
  'resnet' (default) - the local torch checkpoint (existing behavior, unchanged).
  'automl' - calls out to a teammate's Vertex AI AutoML Vision Endpoint (cross-project) instead
    of running local inference. Swapped in 2026-07 because the local ResNet (P1) was observed
    collapsing to stage-1/confidence~1.0 on real phone photos (domain-gap failure vs the light-
    box training data). The ResNet checkpoint/code is kept, NOT deleted — MODEL_BACKEND=resnet
    switches back to it with no code change. Needs AUTOML_PROJECT/AUTOML_LOCATION/
    AUTOML_ENDPOINT_ID, and the deploy service account must be granted roles/aiplatform.user on
    the teammate's project (cross-project IAM; no key file — Cloud Run uses its attached SA via
    ADC). The AutoML model's displayNames are expected to be the literal strings "1".."5"
    (validated at request time; a mismatch fails loudly rather than silently mismapping stages).
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
from shelf_life import alpha_from_temp, estimate_days_to_target  # noqa: E402
from transforms import evaluation_transform  # noqa: E402

STAGE_HINT = {1: "Unripe (firm)", 2: "Breaking", 3: "Ready to eat",
              4: "Peak (end of shelf life)", 5: "Overripe (too late)"}

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
HTTP_PORT = int(os.environ.get("AIP_HTTP_PORT", "8080"))
STORAGE_URI = os.environ.get("AIP_STORAGE_URI")
LOCAL_MODEL_DIR = Path(os.environ.get("LOCAL_MODEL_DIR", "/tmp/model"))
# Background-removal crop (src/preprocess.py). Off by default so the current CPU Cloud Run deploy
# is unchanged; turn on with ENABLE_CROP=1 (also needs the preprocess deps in the image, see
# requirements-preprocess.txt). SEGMENTER=rembg is CPU-friendly (current default); 'sam3' needs a GPU.
ENABLE_CROP = os.environ.get("ENABLE_CROP", "0") == "1"
SEGMENTER = os.environ.get("SEGMENTER", "rembg")   # rembg (CPU) | sam3 (GPU) — swap later
SAM3_WEIGHTS = os.environ.get("SAM3_WEIGHTS", "sam3.pt")
# img_size used only for the returned cropped_b64 preview; decoupled from any local model cfg
# since the automl backend has no local checkpoint/cfg to read img_size from.
CROP_IMG_SIZE = int(os.environ.get("CROP_IMG_SIZE", "224"))

# Classifier backend. Default 'resnet' is the safe/explicit choice: forgetting to set this env
# on a redeploy must NOT silently start calling out to another team's GCP project.
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "resnet")   # resnet | automl
AUTOML_PROJECT = os.environ.get("AUTOML_PROJECT")
AUTOML_LOCATION = os.environ.get("AUTOML_LOCATION", "us-central1")
AUTOML_ENDPOINT_ID = os.environ.get("AUTOML_ENDPOINT_ID")

app = FastAPI()
_state: dict = {"model": None, "transform": None, "device": "cpu", "cfg": None, "segmenter": None,
                "automl_client": None, "automl_endpoint_path": None}


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


def load_automl_client() -> None:
    """Build the Vertex AI PredictionServiceClient + resource path for the teammate's
    cross-project AutoML Endpoint. No key file: Cloud Run authenticates as its attached service
    account via ADC, which must be granted roles/aiplatform.user on AUTOML_PROJECT."""
    from google.cloud import aiplatform

    if not AUTOML_PROJECT or not AUTOML_ENDPOINT_ID:
        raise RuntimeError("MODEL_BACKEND=automl requires AUTOML_PROJECT and AUTOML_ENDPOINT_ID")
    client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": f"{AUTOML_LOCATION}-aiplatform.googleapis.com"})
    path = client.endpoint_path(project=AUTOML_PROJECT, location=AUTOML_LOCATION,
                                endpoint=AUTOML_ENDPOINT_ID)
    _state["automl_client"] = client
    _state["automl_endpoint_path"] = path


@app.on_event("startup")
def _startup() -> None:
    if MODEL_BACKEND == "resnet":
        load_model()
    elif MODEL_BACKEND == "automl":
        load_automl_client()
        print(f"[serving] backend=automl project={AUTOML_PROJECT} location={AUTOML_LOCATION} "
              f"endpoint={AUTOML_ENDPOINT_ID}")
    else:
        raise ValueError(f"unknown MODEL_BACKEND {MODEL_BACKEND!r} (expected resnet|automl)")

    if ENABLE_CROP:
        # lazy: pulls the segmenter's deps (rembg / ultralytics) only when cropping is on.
        from preprocess import load_segmenter

        kw = {"weights": SAM3_WEIGHTS, "device": _state["device"]} if SEGMENTER == "sam3" else {}
        _state["segmenter"] = load_segmenter(SEGMENTER, **kw)
        print(f"[serving] crop ENABLED (segmenter={SEGMENTER!r}, device={_state['device']})")


@app.get(HEALTH_ROUTE)
def health():
    if MODEL_BACKEND == "automl":
        if _state["automl_client"] is None:
            return JSONResponse({"status": "not ready"}, status_code=503)
        return {"status": "ok", "backend": "automl", "endpoint": AUTOML_ENDPOINT_ID}
    if _state["model"] is None:
        return JSONResponse({"status": "not ready"}, status_code=503)
    return {"status": "ok", "backend": "resnet", "experiment_id": _state["cfg"]["experiment_id"]}


def _encode_jpeg_b64(pil_img) -> str:
    """base64-encoded JPEG bytes of the crop (method A: backend decodes and saves it, then
    sets images.cropped_url). Raw base64, no data: prefix."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _classify_resnet(img) -> tuple[int, list[float]]:
    x = _state["transform"](img).unsqueeze(0).to(_state["device"])
    with torch.no_grad():
        probs = torch.softmax(_state["model"](x), 1)[0].cpu().numpy()
    stage = int(probs.argmax()) + 1
    return stage, [float(probs[k - 1]) for k in range(1, 6)]


def _classify_automl(img) -> tuple[int, list[float]]:
    """Call the teammate's cross-project AutoML Vision Endpoint. Builds the standard AutoML
    image-classification request (base64 JPEG content, confidence_threshold=0 + max_predictions=5
    so all 5 stages come back), then maps displayName -> stage index explicitly (AutoML does NOT
    guarantee the returned order matches stage order) rather than trusting positional order.

    response.predictions[0] is a proto-plus MapComposite (a google.protobuf.Value struct that
    proto-plus auto-unwraps into a dict-like object) -- it supports .get()/indexing directly, no
    json_format/ClassificationPredictionResult parsing needed (verified against the installed
    google-cloud-aiplatform; going through ._pb + MessageToDict throws on this field type)."""
    from google.cloud.aiplatform.gapic.schema import predict as automl_schema

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    content_b64 = base64.b64encode(buf.getvalue()).decode()

    instance = automl_schema.instance.ImageClassificationPredictionInstance(
        content=content_b64).to_value()
    parameters = automl_schema.params.ImageClassificationPredictionParams(
        confidence_threshold=0.0, max_predictions=5).to_value()

    response = _state["automl_client"].predict(
        endpoint=_state["automl_endpoint_path"], instances=[instance], parameters=parameters)
    if not response.predictions:
        raise RuntimeError("AutoML endpoint returned no predictions")
    result = response.predictions[0]
    names = result.get("displayNames") or []
    confs = result.get("confidences") or []
    if not names or len(names) != len(confs):
        raise RuntimeError(f"AutoML response missing displayNames/confidences: {dict(result)!r}")

    probs = [0.0] * 5
    for name, conf in zip(names, confs):
        try:
            idx = int(name) - 1
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                f"AutoML displayName {name!r} is not an int 1-5 -- label schema mismatch, "
                "refusing to guess a mapping") from e
        if not 0 <= idx <= 4:
            raise RuntimeError(f"AutoML displayName {name!r} out of range 1-5")
        probs[idx] = float(conf)
    stage = probs.index(max(probs)) + 1
    return stage, probs


def _predict_one(image_bytes: bytes, target_stage=None, temp_celsius: float = 20.0) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cropped_b64 = None
    seg = _state.get("segmenter")
    if seg is not None:
        # Background-removal crop (§3 domain gap). Classify the CROP, not the raw photo, and hand
        # the crop back to the backend. A NoAvocadoDetected raise propagates to predict()'s
        # per-image try/except -> {"error": ...} -> backend NO_AVOCADO_DETECTED.
        from preprocess import preprocess_real_photo

        img_size = _state["cfg"].get("img_size", CROP_IMG_SIZE) if _state["cfg"] else CROP_IMG_SIZE
        img = preprocess_real_photo(img, seg, img_size=img_size)
        cropped_b64 = _encode_jpeg_b64(img)

    if MODEL_BACKEND == "automl":
        stage, probs = _classify_automl(img)
        model_version = f"automl:{AUTOML_PROJECT}:{AUTOML_ENDPOINT_ID}"
    else:
        stage, probs = _classify_resnet(img)
        model_version = _state["cfg"]["experiment_id"]

    out = {
        "predicted_stage": stage,
        "label": LABELS[stage],
        "hint": STAGE_HINT[stage],
        "confidence": probs[stage - 1],
        "stage_probs": probs,
        "model_version": model_version,
    }
    if cropped_b64 is not None:
        out["cropped_b64"] = cropped_b64  # method A: backend saves -> cropped_url
    # days_to_target: days until the CURRENT stage reaches the user's target_stage.
    # NOT clipped at 0 (negative = already past target, i.e. overripe — the app
    # needs that), rounded to 1 decimal (DB column is NUMERIC(4,1)). Only emitted
    # when target_stage is given (agreed contract: absent -> field omitted ->
    # backend stores null). Do NOT invent a default target_stage.
    if target_stage is not None:
        alpha = alpha_from_temp(temp_celsius)
        days = estimate_days_to_target(stage, target_stage, alpha, clip_negative=False)
        out["days_to_target"] = round(days, 1)
    return out


@app.post(PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()
    instances = body.get("instances", [])
    params = body.get("parameters") or {}
    target_stage = params.get("target_stage")
    temp_celsius = params.get("temp_celsius", 20.0)

    predictions = []
    for inst in instances:
        try:
            inst = inst if isinstance(inst, dict) else {"b64": inst}
            image_bytes = base64.b64decode(inst["b64"])
            predictions.append(_predict_one(image_bytes, target_stage, temp_celsius))
        except Exception as e:
            predictions.append({"error": str(e)})
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT)
