#!/usr/bin/env bash
# Vertex AI Custom Training container entrypoint.
#   1) Unzip the dataset zip from GCS (/gcs auto-mount) to the container's local SSD
#      → training reads repeatedly from local SSD (avoids per-read FUSE latency)
#   2) validate_data → split (generates splits.csv; needs image-existence check → after unzip)
#   3) train (durably saves best.pt / history.json to AVOCADO_OUTPUT_DIR=/gcs/...)
#
# Behavior is controlled via env (default = P1: general × ResNet-18). Override at job submission.
set -euo pipefail

BUCKET="${BUCKET:-qi-2026summer-avocado}"
CONFIG="${CONFIG:-configs/paper/general_resnet18.yaml}"
EXP_ID="${EXP_ID:-P1_general_resnet18_paper_aug_oversample}"
ZIP_PATH="${ZIP_PATH:-/gcs/${BUCKET}/data/raw/avocado.zip}"
MANIFEST_PATH="${MANIFEST_PATH:-/gcs/${BUCKET}/data/avocado_complete_states.csv}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-}"                     # config epochs override (for practical runs)
VAL_FREQ="${VAL_FREQ:-}"                 # validation frequency override
SKIP_IMAGE_CHECK="${SKIP_IMAGE_CHECK:-0}"  # if 1, skip item 12 (full corrupted-image scan)
SKIP_TRAIN="${SKIP_TRAIN:-0}"           # if 1, skip training (evaluate only with existing best.pt)
RUN_EVAL="${RUN_EVAL:-1}"               # if 1, auto-evaluate the test set after training (evaluate.py). smoke auto-skipped

export AVOCADO_OUTPUT_DIR="/gcs/${BUCKET}/outputs/${EXP_ID}"

echo "[entrypoint] BUCKET=$BUCKET EXP_ID=$EXP_ID CONFIG=$CONFIG SMOKE=$SMOKE"
echo "[entrypoint] OUTPUT_DIR=$AVOCADO_OUTPUT_DIR"

# --- 1) unzip -------------------------------------------------------------
echo "[entrypoint] unzip $ZIP_PATH -> /workspace/images_extract"
if [ ! -f "$ZIP_PATH" ]; then
  echo "ERROR: zip not found: $ZIP_PATH (upload it first with scripts/02_upload_data.sh)"; exit 1
fi
mkdir -p /workspace/images_extract
unzip -q -o "$ZIP_PATH" -d /workspace/images_extract

# Find the directory containing *.jpg wherever it is in the nested folders (handles CLAUDE.md §1 single-level nesting)
FIRST_JPG="$(find /workspace/images_extract -name '*.jpg' -print -quit)"
if [ -z "$FIRST_JPG" ]; then echo "ERROR: no .jpg inside zip"; exit 1; fi
export AVOCADO_IMAGE_DIR="$(dirname "$FIRST_JPG")"
JPG_COUNT="$(find "$AVOCADO_IMAGE_DIR" -maxdepth 1 -name '*.jpg' | wc -l)"
echo "[entrypoint] IMAGE_DIR=$AVOCADO_IMAGE_DIR  jpg=$JPG_COUNT (expected 14710)"

# --- 2) validate_data + split --------------------------------------------
# Curated keep-list manifest (excludes never-reached-5 / gap samples, CLAUDE.md §2.6).
# Read straight from the /gcs mount, same as the zip. Absent -> validate_data trains on
# the full dataset (filter_to_manifest is a no-op), so this degrades safely.
if [ -f "$MANIFEST_PATH" ]; then
  export AVOCADO_MANIFEST="$MANIFEST_PATH"
  echo "[entrypoint] manifest (GCS) -> $AVOCADO_MANIFEST"
else
  # data.py then falls back to the image-bundled data/avocado_complete_states.csv if it
  # was baked in (it is, unless removed); only if neither exists is the full dataset used.
  echo "[entrypoint] manifest not in GCS ($MANIFEST_PATH) -> using image-bundled copy if present, else FULL dataset"
fi

VD_ARGS=()
[ "$SKIP_IMAGE_CHECK" = "1" ] && VD_ARGS+=(--skip-image-check)
echo "[entrypoint] python -m src.validate_data ${VD_ARGS[*]-}"
python -m src.validate_data "${VD_ARGS[@]}"   # bash 5: an empty array expands to 0 arguments
echo "[entrypoint] python -m src.split"
python -m src.split

# --- 3) train -------------------------------------------------------------
if [ "$SKIP_TRAIN" = "1" ]; then
  echo "[entrypoint] SKIP_TRAIN=1 → skip training, evaluate only with existing best.pt"
else
  TR_ARGS=(--config "$CONFIG")
  [ "$SMOKE" = "1" ] && TR_ARGS+=(--smoke-test)
  [ -n "$EPOCHS" ] && TR_ARGS+=(--epochs "$EPOCHS")
  [ -n "$VAL_FREQ" ] && TR_ARGS+=(--val-freq "$VAL_FREQ")
  echo "[entrypoint] python -m src.train ${TR_ARGS[*]}"
  python -m src.train "${TR_ARGS[@]}"
fi

# --- 4) evaluate (test-set model analysis: confusion matrix, QWK, per-class, best/two-side, shelf-life) ----
# Unlike AutoML, Custom Training does not provide analysis automatically, so generate it here via evaluate.py.
# smoke is skipped because its checkpoint is meaningless.
if [ "$RUN_EVAL" = "1" ] && [ "$SMOKE" != "1" ]; then
  echo "[entrypoint] python -m src.evaluate --config $CONFIG"
  python -m src.evaluate --config "$CONFIG"
else
  echo "[entrypoint] skip evaluate (RUN_EVAL=$RUN_EVAL SMOKE=$SMOKE)"
fi

echo "[entrypoint] DONE. outputs -> $AVOCADO_OUTPUT_DIR"
