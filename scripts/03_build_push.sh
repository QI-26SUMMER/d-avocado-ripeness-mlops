#!/usr/bin/env bash
# Build the training image with Cloud Build and push it to Artifact Registry.
# (No local docker needed — the build runs on GCP)
set -euo pipefail

PROJECT="${PROJECT:-qi-2026summer}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-avocado}"
TAG="${TAG:-latest}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/train:${TAG}"

echo "Cloud Build -> $IMAGE"
# Uses the Dockerfile at the repo root. .dockerignore minimizes the build context.
gcloud builds submit --tag "$IMAGE" --project "$PROJECT" .

echo "Done: $IMAGE"
echo "Next: scripts/04_submit_job.sh   (smoke: SMOKE=1 scripts/04_submit_job.sh)"
