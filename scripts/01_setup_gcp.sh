#!/usr/bin/env bash
# One-time GCP setup (idempotent — safe to run multiple times).
#   - Enable the required APIs
#   - Create the GCS bucket / Artifact Registry repo
#   - Check the Vertex AI T4 custom-training GPU quota (new projects start at 0 → request an increase)
set -euo pipefail

PROJECT="${PROJECT:-qi-2026summer}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-qi-2026summer-avocado}"
AR_REPO="${AR_REPO:-avocado}"

gcloud config set project "$PROJECT" >/dev/null

echo "== Enable APIs =="
gcloud services enable \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com

echo "== GCS bucket (create if absent) =="
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://${BUCKET}" \
       --location="$REGION" --uniform-bucket-level-access

echo "== Artifact Registry repo (create if absent) =="
gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$AR_REPO" \
       --repository-format=docker --location="$REGION" \
       --description="Avocado training images"

echo "== T4 custom-training GPU quota (us-central1) =="
echo "  If the current limit is 0, request an increase as below (approval is on Google's side, minutes to hours):"
cat <<EOF
  gcloud alpha quotas preferences create \\
    --service=aiplatform.googleapis.com \\
    --quota-id=CustomModelTrainingT4GPUsPerProjectPerRegion \\
    --preferred-value=1 --dimensions=region=${REGION} \\
    --project=${PROJECT} --email=<your-email> \\
    --justification="Reproduce avocado ripeness ResNet-18 training on Vertex."
EOF

echo "== Done =="
