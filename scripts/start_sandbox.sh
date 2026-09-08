#!/usr/bin/env bash
set -euo pipefail
if ! command -v docker >/dev/null; then
  echo 'Docker is required for the Sandbox Fusion service. Run this script on a Docker host and set SANDBOX_URL in the trainer.' >&2
  exit 1
fi
# No GPU or model/dataset mounts are needed. The service receives code and test stdin over HTTP.
exec docker run --rm --name verl-coding-sandbox \
  -p "${SANDBOX_BIND:-127.0.0.1}:8080:8080" --cpus=4 --memory=8g --pids-limit=256 \
  volcengine/sandbox-fusion:server-20250609
