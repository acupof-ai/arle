#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'chmod -R u+w "$TMP" 2>/dev/null || true; rm -rf "$TMP"' EXIT
REPO="$TMP/repo"
mkdir -p "$REPO/scripts" "$REPO/assets"
cp "$ROOT/scripts/validate_release.sh" "$REPO/scripts/"
cat >"$REPO/Cargo.toml" <<'EOF'
[workspace.package]
version = "1.2.3"
EOF
printf '{"schema":1,"blockers":[]}\n' >"$REPO/release-blockers.json"
printf X >"$REPO/bundle-input"
cat >"$REPO/scripts/kernel_artifacts.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
id="bundle:$(shasum -a 256 "$ROOT/bundle-input" | awk '{print $1}')"
case "$1" in
    id) printf '%s\n' "$id" ;;
    fetch)
        dir="${2:-}"
        if [[ -n "$dir" ]]; then
            file="arle-kernels-t1-$id.tar.gz"
            [[ -f "$dir/$file" && -f "$dir/$file.sha256" ]] || exit 3
        fi
        exit 0
        ;;
    *) exit 2 ;;
esac
EOF
chmod +x "$REPO/scripts/kernel_artifacts.sh"
git -C "$REPO" init -q
git -C "$REPO" config user.email test@example.com
git -C "$REPO" config user.name test
git -C "$REPO" add .
git -C "$REPO" commit -qm A
A="$(git -C "$REPO" rev-parse HEAD)"
printf non-bundle >"$REPO/note"
git -C "$REPO" add note
git -C "$REPO" commit -qm B
B="$(git -C "$REPO" rev-parse HEAD)"
printf Y >"$REPO/bundle-input"
git -C "$REPO" add bundle-input
git -C "$REPO" commit -qm C
C="$(git -C "$REPO" rev-parse HEAD)"
U="$(printf 'unrelated\n' | git -C "$REPO" commit-tree "$A^{tree}")"

asset() {
    local commit="$1" evidence_id="$2" file_id="${4:-$2}" dir="$3" file sha with_sidecar="${5:-1}"
    file="arle-kernels-t1-$file_id.tar.gz"
    mkdir -p "$dir"
    printf archive >"$dir/$file"
    sha="$(shasum -a 256 "$dir/$file" | awk '{print $1}')"
    printf '%s  %s\n' "$sha" "$file" >"$dir/$file.sha256"
    if [[ "$with_sidecar" == 1 ]]; then
        jq -cn --arg id "$evidence_id" --arg commit "$commit" --arg sha "$sha" \
            '{schema:1,status:"passed",candidate_archive_sha256:$sha,bundle_id:$id,source_commit:$commit,kernel_build_id:"kernel",bundle_capabilities:[],observations:[]}' \
            >"$dir/$file.qualification.json"
    fi
}
ID_X="bundle:$(printf X | shasum -a 256 | awk '{print $1}')"
ID_Y="bundle:$(printf Y | shasum -a 256 | awk '{print $1}')"
asset "$A" "$ID_X" "$REPO/assets/x"
asset "$A" "$ID_X" "$REPO/assets/changed" "$ID_Y"
asset "0000000000000000000000000000000000000000" "$ID_X" "$REPO/assets/missing"
asset "$A" "$ID_X" "$REPO/assets/candidate-only" "$ID_X" 0

run_at() {
    local commit="$1" source="$2" expected="$3" pattern="${4:-}"
    git -C "$REPO" switch -q --detach "$commit"
    git -C "$REPO" tag -f v1.2.3 "$commit" >/dev/null
    if output="$(ARLE_KERNEL_ARTIFACTS_SCRIPT="$REPO/scripts/kernel_artifacts.sh" \
        "$REPO/scripts/validate_release.sh" v1.2.3 "$source" 2>&1)"; then
        [[ "$expected" == pass ]] || { echo "unexpected success at $commit" >&2; exit 1; }
    else
        [[ "$expected" == fail ]] || { printf '%s\n' "$output" >&2; exit 1; }
        grep -Fq "$pattern" <<<"$output" || { printf 'missing error %q in: %s\n' "$pattern" "$output" >&2; exit 1; }
    fi
}

run_at "$A" "$REPO/assets/x" pass
run_at "$B" "$REPO/assets/x" pass
run_at "$C" "$REPO/assets/changed" fail "kernel evidence bundle identity changed"
run_at "$U" "$REPO/assets/x" fail "kernel evidence commit is not an ancestor"
run_at "$B" "$REPO/assets/missing" fail "kernel evidence commit unavailable locally"
run_at "$A" "$REPO/assets/candidate-only" pass

echo "release validation self-test passed"
