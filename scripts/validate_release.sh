#!/usr/bin/env bash
# Fail-closed on tag/product/blockers. Kernel gate requires the exact T1
# candidate archive for the current bundle id (immutable pack + checksum).
# Multi-SM formal qualification sidecars remain optional: H20/pod evidence is
# the operator correctness bar when self-hosted SM80/86/89/90 runners are gone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:?usage: scripts/validate_release.sh vX.Y.Z [kernel-bundle-dir-or-release]}"
SOURCE="${2:-kernel-artifacts}"
KERNEL_ARTIFACTS="${ARLE_KERNEL_ARTIFACTS_SCRIPT:-$ROOT/scripts/kernel_artifacts.sh}"
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

bundle_id="$("$KERNEL_ARTIFACTS" id)"
lane="${ARLE_KERNEL_BUNDLE_LANE:-t1}"
file="arle-kernels-$lane-$bundle_id.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if [[ -d "$SOURCE" ]]; then
    cp "$SOURCE/$file" "$SOURCE/$file.sha256" "$tmp/"
    [[ -f "$SOURCE/$file.qualification.json" ]] && cp "$SOURCE/$file.qualification.json" "$tmp/"
else
    gh release download "$SOURCE" -R "${GITHUB_REPOSITORY:-acupof-ai/arle}" \
        -p "$file" -p "$file.sha256" -D "$tmp"
    # Optional formal sidecar — present only after multi-SM qualify-publish.
    gh release download "$SOURCE" -R "${GITHUB_REPOSITORY:-acupof-ai/arle}" \
        -p "$file.qualification.json" -D "$tmp" 2>/dev/null || true
fi
[[ -f "$tmp/$file" && -f "$tmp/$file.sha256" ]] || {
    echo "missing exact kernel candidate for bundle $bundle_id" >&2
    exit 1
}
"$KERNEL_ARTIFACTS" fetch "$tmp" >/dev/null

if [[ -f "$tmp/$file.qualification.json" ]]; then
    evidence="$tmp/$file.qualification.json"
    evidence_bundle_id="$(jq -er '.bundle_id' "$evidence")"
    [[ "$evidence_bundle_id" == "$bundle_id" ]] || {
        echo "kernel evidence bundle identity changed: evidence=$evidence_bundle_id current=$bundle_id" >&2
        exit 1
    }
    archive_sha="$(jq -er '.candidate_archive_sha256' "$evidence")"
    actual_archive_sha="$(shasum -a 256 "$tmp/$file" | awk '{print $1}')"
    [[ "$archive_sha" == "$actual_archive_sha" ]] || {
        echo "kernel evidence is not bound to the fetched candidate archive" >&2
        exit 1
    }
    tested_commit="$(jq -er '.source_commit' "$evidence")"
    git -C "$ROOT" cat-file -e "$tested_commit^{commit}" 2>/dev/null || {
        echo "kernel evidence commit unavailable locally; fetch history or unshallow: $tested_commit" >&2
        exit 1
    }
    if ! git -C "$ROOT" merge-base --is-ancestor "$tested_commit" "$head"; then
        if [[ "$(git -C "$ROOT" rev-parse --is-shallow-repository)" == true ]]; then
            echo "kernel evidence ancestry unavailable in shallow history; fetch history or unshallow: $tested_commit" >&2
        else
            echo "kernel evidence commit is not an ancestor of release HEAD: tested=$tested_commit head=$head" >&2
        fi
        exit 1
    fi
    printf 'release-ready tag=%s commit=%s bundle=%s mode=qualified tested_commit=%s\n' \
        "$TAG" "$head" "$bundle_id" "$tested_commit"
else
    # Candidate-only: exact bundle id + checksum verified by fetch above.
    # Operator correctness for this release is H20/pod evidence, not multi-SM CI.
    printf 'release-ready tag=%s commit=%s bundle=%s mode=candidate-only\n' \
        "$TAG" "$head" "$bundle_id"
fi
