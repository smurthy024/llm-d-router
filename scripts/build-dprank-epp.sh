#!/usr/bin/env bash
# Build and push dprank EPP image from a Mac (linux/amd64).
# Requires: docker buildx, github.com login to ghcr.io
#
#   echo "$GITHUB_PAT" | docker login ghcr.io -u smurthy024 --password-stdin
#   ./scripts/build-dprank-epp.sh
#
# PAT needs: write:packages, read:packages (github.com classic token)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SHORT_SHA="$(git rev-parse --short HEAD)"
TAG="${TAG:-dprank-${SHORT_SHA}}"
IMAGE="ghcr.io/smurthy024/llm-d-router-epp:${TAG}"

echo "Building ${IMAGE} (linux/amd64)..."
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.epp \
  -t "${IMAGE}" \
  -t ghcr.io/smurthy024/llm-d-router-epp:dprank-latest \
  --build-arg "COMMIT_SHA=$(git rev-parse HEAD)" \
  --build-arg "BUILD_REF=${TAG}" \
  --build-arg 'LDFLAGS=-s -w' \
  --push \
  .

echo "Pushed ${IMAGE}"
