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
DISPLAY_NAME="${DISPLAY_NAME:-avocado-${EXP_ID}$([ "$SMOKE" = "1" ] && echo -smoke)}"

# Generate a config YAML holding the worker-pool spec + container env (the --worker-pool-spec shorthand doesn't support env)
# Created in the current directory (avoids the issue where Windows gcloud can't read /tmp paths). Deleted on exit.
JOB_YAML="./.avocado_job_$$.yaml"
trap 'rm -f "$JOB_YAML"' EXIT
cat > "$JOB_YAML" <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE}
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: ${IMAGE}
      env:
        - name: BUCKET
          value: "${BUCKET}"
        - name: EXP_ID
          value: "${EXP_ID}"
        - name: CONFIG
          value: "${CONFIG}"
        - name: SMOKE
          value: "${SMOKE}"
EOF

echo "== Submit Job =="
echo "  display-name: $DISPLAY_NAME"
echo "  image:        $IMAGE"
echo "  exp/config:   $EXP_ID / $CONFIG  (SMOKE=$SMOKE)"
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
