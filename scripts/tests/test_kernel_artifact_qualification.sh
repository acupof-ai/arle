#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/cuda_prebuilt_manifest.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
TREE="$TMP/tree"
mkdir -p "$TREE/generated"
printf kernel >"$TREE/generated/kernel.bin"
BUNDLE_ID="bundle:$(printf bundle | cuda_prebuilt_hash_stream)"
KERNEL_ID="bundle:$(printf kernels | cuda_prebuilt_hash_stream)"
COMMIT="$(printf commit | cuda_prebuilt_hash_stream)"
cat >"$TREE/generated/arle-cuda-kernels.manifest" <<EOF
schema=3
package=cuda-kernels
capabilities=deepgemm-native,fa3,flashmla
kernel_build_id=$KERNEL_ID
EOF
cat >"$TREE/correctness-evidence.json" <<EOF
{"schema":2,"status":"not-run","bundle_id":"$BUNDLE_ID","source_commit":"$COMMIT","artifact_sha256":null,"tested_sms":[],"capabilities":[]}
EOF
printf symbol >"$TREE/symbols.txt"
printf '{}\n' >"$TREE/placeholder"
SYMBOL_SHA="$(cuda_prebuilt_hash_file "$TREE/symbols.txt")"
EVIDENCE_SHA="$(cuda_prebuilt_hash_file "$TREE/correctness-evidence.json")"
cat >"$TREE/manifest.json" <<EOF
{
  "schema": 3,
  "bundle_id": "$BUNDLE_ID",
  "source_commit": "$COMMIT",
  "abi_sha256": "$(cuda_prebuilt_hash_file "$ROOT/crates/cuda-kernels/kernels.toml")",
  "symbol_allowlist_sha256": "$SYMBOL_SHA",
  "correctness_status": "not-run",
  "correctness_evidence_sha256": "$EVIDENCE_SHA"
}
EOF
(
    cd "$TREE"
    find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z |
        while IFS= read -r -d '' file; do
            printf '%s  %s\n' "$(cuda_prebuilt_hash_file "$file")" "${file#./}"
        done
) >"$TREE/SHA256SUMS"
tar -C "$TREE" -czf "$TMP/candidate.tar.gz" .

printf product >"$TMP/arle"
chmod +x "$TMP/arle"
PRODUCT_SHA="$(cuda_prebuilt_hash_file "$TMP/arle")"
stats() {
    printf '{"build_identity":{"product_binary_sha256":"sha256:%s","kernel_bundle_id":"%s"}}\n' "$1" "$2" >"$3"
}
stats "$PRODUCT_SHA" "$KERNEL_ID" "$TMP/stats.json"

fragment() {
    local sm="$1" profile="$2" caps="$3" output="$4"
    ARLE_KERNEL_TEST_BINARY="$TMP/arle" ARLE_KERNEL_TESTED_SM="$sm" \
        ARLE_KERNEL_QUALIFICATION_PROFILE="$profile" ARLE_KERNEL_TESTED_CAPABILITIES="$caps" \
        "$ROOT/scripts/kernel_artifacts.sh" qualify-fragment \
        "$TMP/candidate.tar.gz" "$TMP/stats.json" "$output"
}
expect_fail() {
    if "$@" >/dev/null 2>&1; then
        echo "unexpected success: $*" >&2
        exit 1
    fi
}

stats "$(printf wrong | cuda_prebuilt_hash_stream)" "$KERNEL_ID" "$TMP/wrong-binary.json"
expect_fail env ARLE_KERNEL_TEST_BINARY="$TMP/arle" ARLE_KERNEL_TESTED_SM=8.0 \
    ARLE_KERNEL_QUALIFICATION_PROFILE=qwen ARLE_KERNEL_TESTED_CAPABILITIES= \
    "$ROOT/scripts/kernel_artifacts.sh" qualify-fragment "$TMP/candidate.tar.gz" \
    "$TMP/wrong-binary.json" "$TMP/wrong.json"
stats "$PRODUCT_SHA" "bundle:$(printf wrong | cuda_prebuilt_hash_stream)" "$TMP/wrong-kernel.json"
expect_fail env ARLE_KERNEL_TEST_BINARY="$TMP/arle" ARLE_KERNEL_TESTED_SM=8.0 \
    ARLE_KERNEL_QUALIFICATION_PROFILE=qwen ARLE_KERNEL_TESTED_CAPABILITIES= \
    "$ROOT/scripts/kernel_artifacts.sh" qualify-fragment "$TMP/candidate.tar.gz" \
    "$TMP/wrong-kernel.json" "$TMP/wrong.json"
expect_fail env ARLE_KERNEL_TEST_BINARY="$TMP/arle" ARLE_KERNEL_TESTED_SM=8.9 \
    ARLE_KERNEL_QUALIFICATION_PROFILE=qwen-fa3 ARLE_KERNEL_TESTED_CAPABILITIES=fa3 \
    "$ROOT/scripts/kernel_artifacts.sh" qualify-fragment "$TMP/candidate.tar.gz" \
    "$TMP/stats.json" "$TMP/wrong.json"

fragment 8.0 qwen '' "$TMP/80.json"
fragment 8.6 qwen '' "$TMP/86.json"
fragment 8.9 qwen '' "$TMP/89.json"
fragment 9.0 qwen-fa3 fa3 "$TMP/90.json"
fragment 9.0 dsv4 flashmla,deepgemm-native "$TMP/dsv4.json"
AGG=("$ROOT/scripts/kernel_artifacts.sh" aggregate-qualification "$TMP/candidate.tar.gz")
expect_fail "${AGG[@]}" "$TMP/incomplete.json" "$TMP/80.json" "$TMP/86.json" "$TMP/89.json"
expect_fail "${AGG[@]}" "$TMP/duplicate.json" "$TMP/80.json" "$TMP/80.json" "$TMP/86.json" "$TMP/89.json" "$TMP/90.json" "$TMP/dsv4.json"

jq '.bundle_id="bundle:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"' \
    "$TMP/80.json" >"$TMP/mixed.json"
expect_fail "${AGG[@]}" "$TMP/mixed-out.json" "$TMP/mixed.json" "$TMP/86.json" "$TMP/89.json" "$TMP/90.json" "$TMP/dsv4.json"
jq '.profile="qwen" | .tested_capabilities=["fa3"]' "$TMP/90.json" >"$TMP/false.json"
expect_fail "${AGG[@]}" "$TMP/false-out.json" "$TMP/80.json" "$TMP/86.json" "$TMP/89.json" "$TMP/false.json" "$TMP/dsv4.json"
jq '.tested_capabilities=["nccl"]' "$TMP/80.json" >"$TMP/overclaim.json"
expect_fail "${AGG[@]}" "$TMP/overclaim-out.json" "$TMP/overclaim.json" "$TMP/86.json" "$TMP/89.json" "$TMP/90.json" "$TMP/dsv4.json"

"${AGG[@]}" "$TMP/a.json" "$TMP/dsv4.json" "$TMP/90.json" "$TMP/89.json" "$TMP/80.json" "$TMP/86.json"
"${AGG[@]}" "$TMP/b.json" "$TMP/86.json" "$TMP/80.json" "$TMP/89.json" "$TMP/dsv4.json" "$TMP/90.json"
cmp "$TMP/a.json" "$TMP/b.json"
jq -e '.schema == 1 and .status == "passed" and (.observations | length == 5) and
    [.observations[].tested_sm] == ["8.0","8.6","8.9","9.0","9.0"]' "$TMP/a.json" >/dev/null

echo "kernel artifact qualification self-test passed"
