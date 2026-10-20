#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
image=${IMAGE_TAG:-pi05-qwen:nvidia-candidate}
architectures=${CUDA_ARCHITECTURES:-89;120}
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo 'Build on x86_64 Linux. This recipe does not configure cross-compilation/emulation.' >&2
  exit 1
fi
if [[ ! $architectures =~ ^[0-9]+(\;[0-9]+)*$ ]]; then
  echo 'CUDA_ARCHITECTURES must be a semicolon-separated list, for example 89;120.' >&2
  exit 1
fi
if [[ -f "$repo/BUILD_KIT_MANIFEST.json" ]]; then
  revision=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_commit"])' "$repo/BUILD_KIT_MANIFEST.json")
else
  revision=$(git -C "$repo" rev-parse HEAD)
  test -z "$(git -C "$repo" status --porcelain -- containers baseline deployment dual_stack flashrt profiling reporting standalone_tokenizer.py README.md summary.md reports/engine-comparison-2026-10-16/baseline-latency)"
fi
artifacts=${BUILD_ARTIFACTS:-$repo/containers/local-artifacts/nvidia-build-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$artifacts"
base=${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.03-py3}
docker pull --platform linux/amd64 "$base"
docker image inspect "$base" > "$artifacts/base-image.json"
base_digest=$(docker image inspect "$base" --format '{{index .RepoDigests 0}}')
test -n "$base_digest"
docker build --platform linux/amd64 -f "$repo/containers/nvidia/Dockerfile" \
  --build-arg "BASE_IMAGE=$base_digest" --build-arg "CUDA_ARCHITECTURES=$architectures" \
  --build-arg "BUILD_JOBS=${BUILD_JOBS:-2}" --build-arg "BENCHMARK_COMMIT=$revision" \
  -t "$image" "$repo"
docker image inspect "$image" > "$artifacts/built-image.json"
echo "Built candidate $image from $revision, targets $architectures; GPU acceptance still required. Metadata: $artifacts"
