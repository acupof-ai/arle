#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:?usage: scripts/validate_release.sh vX.Y.Z [kernel-bundle-dir-or-release]}"
SOURCE="${2:-kernel-artifacts}"
head="$(git -C "$ROOT" rev-parse HEAD)"

[[ "$TAG" == v* && "$(git -C "$ROOT" rev-parse "$TAG^{commit}")" == "$head" ]] || {
    echo "release tag must resolve to checkout HEAD: $TAG" >&2
    exit 1
}
jq -e '.schema == 1 and (keys | sort == ["blockers", "schema"]) and (.blockers | type == "array" and length == 0)' \
    "$ROOT/release-blockers.json" >/dev/null || {
    echo "open or invalid release blockers in release-blockers.json" >&2
    exit 1
}
workspace_version="$(perl -ne 'if (/^\[workspace\.package\]/) {$in=1; next} if ($in && /^version = "([^"]+)"/) {print $1; exit} if ($in && /^\[/) {exit}' "$ROOT/Cargo.toml")"
tag_version="${TAG#v}"
[[ "${tag_version%%-*}" == "$workspace_version" ]] || {
    echo "tag/product version mismatch: tag=$TAG workspace=v$workspace_version" >&2
    exit 1
}

"$ROOT/scripts/kernel_artifacts.sh" fetch-qualified "$SOURCE"
evidence="$ROOT/crates/cuda-kernels/generated/correctness-evidence.json"
bundle_id="$("$ROOT/scripts/kernel_artifacts.sh" id)"
tested_commit="$(jq -er '.source_commit' "$evidence")"
[[ "$tested_commit" == "$head" ]] || {
    echo "kernel evidence must test the release commit exactly" >&2
    exit 1
}
printf 'release-qualified tag=%s commit=%s bundle=%s tested_commit=%s\n' \
    "$TAG" "$head" "$bundle_id" "$tested_commit"
