#!/usr/bin/env bash
# Upload a single dataset zip to GCS (resumable — resumes if interrupted).
# The container unzips this zip at startup (docker/entrypoint.sh). No need to upload 14k individual files.
#
# Usage:
#   scripts/02_upload_data.sh <local-zip-path>
#   e.g. scripts/02_upload_data.sh "C:/Users/.../avocado.zip"
#
# If you don't have the zip locally, download it directly from Mendeley in Cloud Shell:
#   place the zip from "Download all" on the DOI 10.17632/3xd9n945v8.1 page into your Cloud Shell home and
#   run this script from Cloud Shell (avoids uploading from a home connection).
set -euo pipefail

BUCKET="${BUCKET:-qi-2026summer-avocado}"
ZIP_LOCAL="${1:?Usage: 02_upload_data.sh <local-zip-path>}"
DEST="gs://${BUCKET}/data/raw/avocado.zip"

if [ ! -f "$ZIP_LOCAL" ]; then echo "ERROR: file not found: $ZIP_LOCAL"; exit 1; fi

echo "Uploading: $ZIP_LOCAL -> $DEST"
gcloud storage cp "$ZIP_LOCAL" "$DEST"

echo "Verify:"
gcloud storage ls -l "$DEST"
echo "Done. Next: scripts/03_build_push.sh"
