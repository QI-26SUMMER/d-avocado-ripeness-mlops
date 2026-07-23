#!/usr/bin/env bash
# Submit a Vertex AI Custom Training Job (T4 ×1).
# Experiments are controlled via env. Default = P1 (general × ResNet-18). SMOKE=1 runs a small 1-epoch smoke.
#
# Usage:
#   scripts/04_submit_job.sh                 # P1 full run
#   SMOKE=1 scripts/04_submit_job.sh         # smoke (pipeline check, cheap)
#   EXP_ID=... CONFIG=... scripts/04_submit_job.sh   # extend to other experiments
set -euo pipefail

PROJECT="${PROJECT:-qi-2026summer}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-avocado}"
TAG="${TAG:-latest}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/train:${TAG}"

BUCKET="${BUCKET:-qi-2026summer-avocado}"
CONFIG="${CONFIG:-configs/paper/general_resnet18.yaml}"
EXP_ID="${EXP_ID:-P1_general_resnet18_paper_aug_oversample}"
SMOKE="${SMOKE:-0}"
MACHINE="${MACHINE:-n1-standard-8}"
GPU="${GPU:-1}"                          # GPUs to attach; GPU=0 -> CPU-only (the GMM baseline needs no GPU)
ACCEL="${ACCEL:-NVIDIA_TESLA_T4}"
DISPLAY_NAME="${DISPLAY_NAME:-avocado-${EXP_ID}$([ "$SMOKE" = "1" ] && echo -smoke)}"

# Generate a config YAML holding the worker-pool spec + container env (the --worker-pool-spec shorthand doesn't support env)
# Created in the current directory (avoids the issue where Windows gcloud can't read /tmp paths). Deleted on exit.
JOB_YAML="./.avocado_job_$$.yaml"
trap 'rm -f "$JOB_YAML"' EXIT
# Emit a container env entry only when the var is set, so a caller can forward extra controls
# without editing this file, e.g.  SKIP_TRAIN=1 RUN_EVAL=0 RUN_GMM=1 scripts/04_submit_job.sh
emit_env() { if [ -n "${2:-}" ]; then printf '        - name: %s\n          value: "%s"\n' "$1" "$2"; fi; }
{
  echo "workerPoolSpecs:"
  echo "  - machineSpec:"
  echo "      machineType: ${MACHINE}"
  if [ "$GPU" != "0" ]; then
    echo "      acceleratorType: ${ACCEL}"
    echo "      acceleratorCount: ${GPU}"
  fi
  echo "    replicaCount: 1"
  echo "    containerSpec:"
  echo "      imageUri: ${IMAGE}"
  echo "      env:"
  emit_env BUCKET "$BUCKET"
  emit_env EXP_ID "$EXP_ID"
  emit_env CONFIG "$CONFIG"
  emit_env SMOKE "$SMOKE"
  emit_env SKIP_TRAIN "${SKIP_TRAIN:-}"
  emit_env RUN_EVAL "${RUN_EVAL:-}"
  emit_env CV_FOLDS "${CV_FOLDS:-}"
  emit_env CV_RECOVER "${CV_RECOVER:-}"
  emit_env RUN_GMM "${RUN_GMM:-}"
  emit_env GMM_DATASET "${GMM_DATASET:-}"
  emit_env GMM_COMPONENTS "${GMM_COMPONENTS:-}"
  emit_env SKIP_IMAGE_CHECK "${SKIP_IMAGE_CHECK:-}"
  emit_env EPOCHS "${EPOCHS:-}"
  emit_env VAL_FREQ "${VAL_FREQ:-}"
} > "$JOB_YAML"

echo "== Submit Job =="
echo "  display-name: $DISPLAY_NAME"
echo "  image:        $IMAGE"
echo "  exp/config:   $EXP_ID / $CONFIG  (SMOKE=$SMOKE)"
echo "  machine:      $MACHINE  gpu=$GPU$([ "$GPU" != "0" ] && echo " ($ACCEL)")"
echo "  outputs:      gs://${BUCKET}/outputs/${EXP_ID}"
echo "--- job.yaml ---"; cat "$JOB_YAML"; echo "----------------"

gcloud ai custom-jobs create \
  --project="$PROJECT" \
  --region="$REGION" \
  --display-name="$DISPLAY_NAME" \
  --config="$JOB_YAML"

echo
echo "Stream logs:   gcloud ai custom-jobs stream-logs <JOB_ID> --region=${REGION}"
echo "List status:   gcloud ai custom-jobs list --region=${REGION} --limit=5"
