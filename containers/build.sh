#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
image=${IMAGE_TAG:-pi05-qwen:thor-candidate}
if [[ $(uname -m) != aarch64 ]]; then
  echo 'Run the initial build on Thor/aarch64; this recipe does not configure emulation.' >&2
  exit 1
fi
if [[ -f "$repo/BUILD_KIT_MANIFEST.json" ]]; then
  revision=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_commit"])' "$repo/BUILD_KIT_MANIFEST.json")
else
  revision=$(git -C "$repo" rev-parse HEAD)
  test -z "$(git -C "$repo" status --porcelain -- containers baseline deployment dual_stack profiling reporting)"
fi
artifacts=${BUILD_ARTIFACTS:-$repo/containers/local-artifacts/build-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$artifacts"
base=${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}
docker pull --platform linux/arm64 "$base"
docker image inspect "$base" > "$artifacts/base-image.json"
base_digest=$(docker image inspect "$base" --format '{{index .RepoDigests 0}}')
test -n "$base_digest"
docker build --platform linux/arm64 -f "$repo/containers/Dockerfile.thor" \
  --build-arg "BASE_IMAGE=$base_digest" \
  --build-arg "BUILD_JOBS=${BUILD_JOBS:-2}" --build-arg "BENCHMARK_COMMIT=$revision" \
  -t "$image" "$repo"
docker image inspect "$image" > "$artifacts/built-image.json"
echo "Built candidate $image from $revision; GPU acceptance still required. Metadata: $artifacts"
