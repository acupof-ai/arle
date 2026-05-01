#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

sha="${1:-$(git rev-parse --short=8 HEAD)}"
local_image="arle-dev:${sha}"
remote_image="ghcr.io/cklxx/arle:dev-${sha}"

docker image inspect "${local_image}" >/dev/null
docker tag "${local_image}" "${remote_image}"
docker push "${remote_image}"

echo "${remote_image}"
