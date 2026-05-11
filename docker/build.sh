#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build sdcat Docker images locally (and optionally push to Docker Hub).

Usage:
  docker/build_images.sh [VERSION] [--push] [--login]

Examples:
  docker/build_images.sh 1.29.1
  docker/build_images.sh 1.29.1 --push --login
  docker/build_images.sh --push --login   # VERSION defaults to "latest"

Tags produced:
  CUDA 13:
    mbari/sdcat:<VERSION>-cuda13
    mbari/sdcat:cuda13
  CPU:
    mbari/sdcat:<VERSION>
    mbari/sdcat:latest

Environment for --login:
  DOCKERHUB_USERNAME
  DOCKERHUB_TOKEN
EOF
}

VERSION="${1:-}"
PUSH=0
LOGIN=0

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push) PUSH=1 ;;
    --login) LOGIN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if [[ -z "${VERSION}" || "${VERSION}" == "--push" || "${VERSION}" == "--login" ]]; then
  VERSION="latest"
fi

if [[ "${PUSH}" -eq 1 && "${LOGIN}" -eq 1 ]]; then
  : "${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME is required for --login}"
  : "${DOCKERHUB_TOKEN:?DOCKERHUB_TOKEN is required for --login}"
  echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin
fi

BUILDX_BUILDER_NAME="${BUILDX_BUILDER_NAME:-sdcat-builder}"
docker buildx inspect "${BUILDX_BUILDER_NAME}" >/dev/null 2>&1 || docker buildx create --name "${BUILDX_BUILDER_NAME}" --use

PLATFORM="${PLATFORM:-linux/amd64}"

# If we're not pushing, load images into the local Docker daemon.
EXTRA_FLAGS=(--no-cache --platform "${PLATFORM}")
if [[ "${PUSH}" -eq 1 ]]; then
  EXTRA_FLAGS+=(--push)
else
  EXTRA_FLAGS+=(--load)
fi

echo "Building CUDA 13 image for VERSION=${VERSION} ..."
docker buildx build "${EXTRA_FLAGS[@]}" \
  -t "mbari/sdcat:${VERSION}-cuda13" \
  -t "mbari/sdcat:cuda13" \
  --label "GIT_VERSION=${VERSION}" \
  --label "IMAGE_URI=mbari/sdcat:${VERSION}-cuda13" \
  -f docker/Dockerfile.cuda .

docker push "mbari/sdcat:${VERSION}-cuda13"

echo "Building CPU image for VERSION=${VERSION} ..."
docker buildx build "${EXTRA_FLAGS[@]}" \
  -t "mbari/sdcat:${VERSION}" \
  -t "mbari/sdcat:latest" \
  --label "GIT_VERSION=${VERSION}" \
  --label "IMAGE_URI=mbari/sdcat:${VERSION}" \
  -f docker/Dockerfile .

docker push "mbari/sdcat:${VERSION}"

echo "Done."