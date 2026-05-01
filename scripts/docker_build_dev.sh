#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

sha="${1:-$(git rev-parse --short=8 HEAD)}"
image="arle-dev:${sha}"

docker build \
  --file Dockerfile \
  --target dev \
  --tag "${image}" \
  .

echo "${image}"
