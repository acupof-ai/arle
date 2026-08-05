#!/usr/bin/env bash
# Run the FP8 small-M component probe without manufacturing qualification.
#
# Required identity:
#   ARLE_SMALLM_MODEL_KIND=actual|synthetic
#   ARLE_SMALLM_MODEL_REVISION=<exact revision>
#   ARLE_SMALLM_E2E_STATUS=passed|failed|not_run
#
# passed/failed E2E requires ARLE_SMALLM_E2E_ARTIFACT pointing to an
# arle.operator-e2e/v1 JSON artifact for this exact binary and kernel bundle.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="${1:-/dev/stdout}"
cd "$ROOT"
source "$ROOT/scripts/export_prebuilt_cuda_kernels.sh"

die() {
    echo "[run_fp8_probe] ERROR: $*" >&2
    exit 1
}

PROBE_BIN="${ARLE_SMALLM_PROBE_BIN:-target/release/examples/fp8_smallm_gemm_probe}"
if [[ ! -x "$PROBE_BIN" ]]; then
    cargo build -p infer-cuda --release --features cuda --example fp8_smallm_gemm_probe
fi
[[ -x "$PROBE_BIN" ]] || die "probe binary missing: $PROBE_BIN"
BINARY_ID="sha256:$(cuda_prebuilt_hash_file "$PROBE_BIN")"

MODEL_KIND="${ARLE_SMALLM_MODEL_KIND:?set ARLE_SMALLM_MODEL_KIND=actual|synthetic}"
[[ "$MODEL_KIND" == "actual" || "$MODEL_KIND" == "synthetic" ]] ||
    die "ARLE_SMALLM_MODEL_KIND must be actual or synthetic"
MODEL_REVISION="${ARLE_SMALLM_MODEL_REVISION:?set exact ARLE_SMALLM_MODEL_REVISION}"
[[ "$MODEL_REVISION" != "unreported" ]] || die "model revision cannot be unreported"

MANIFEST="${ARLE_SMALLM_KERNEL_MANIFEST:-}"
if [[ -z "$MANIFEST" && -n "${ARLE_CUDA_KERNELS_PREBUILT_DIR:-}" ]]; then
    MANIFEST="$ARLE_CUDA_KERNELS_PREBUILT_DIR/arle-cuda-kernels.manifest"
fi
if [[ -n "$MANIFEST" ]]; then
    [[ -f "$MANIFEST" ]] || die "kernel manifest missing: $MANIFEST"
    MANIFEST_DIR="$(cd "$(dirname "$MANIFEST")" && pwd)"
    cuda_prebuilt_validate_bundle "$MANIFEST_DIR" || die "invalid kernel producer manifest or artifact hashes"
    BUNDLE_ID="$(cuda_prebuilt_manifest_value "$MANIFEST" kernel_build_id)"
    [[ "$BUNDLE_ID" =~ ^bundle:[0-9a-f]{64}$ ]] || die "kernel manifest has invalid kernel_build_id"
    # First line only: `--kernel-build-id` also prints `capabilities:` (06a27527e).
    EMBEDDED_BUNDLE_ID="$("$PROBE_BIN" --kernel-build-id | head -n1)" || die "probe binary cannot report its kernel build ID"
    [[ "$EMBEDDED_BUNDLE_ID" == "$BUNDLE_ID" ]] || die "kernel manifest ID does not match the probe binary"
    BUNDLE_ID_SOURCE="verified_binary"
    BUNDLE_MANIFEST_SHA256="$(cuda_prebuilt_hash_file "$MANIFEST")"
else
    BUNDLE_ID=""
    BUNDLE_ID_SOURCE="unverified"
    BUNDLE_MANIFEST_SHA256=""
fi

E2E_STATUS="${ARLE_SMALLM_E2E_STATUS:?set ARLE_SMALLM_E2E_STATUS=passed|failed|not_run}"
case "$E2E_STATUS" in
    not_run)
        [[ -z "${ARLE_SMALLM_E2E_ARTIFACT:-}" ]] ||
            die "not_run must not carry an E2E artifact"
        E2E_PASS=0
        E2E_ARTIFACT_SHA256=""
        E2E_MODEL_KIND=""
        E2E_MODEL_REVISION=""
        ;;
    passed|failed)
        E2E_ARTIFACT="${ARLE_SMALLM_E2E_ARTIFACT:?passed/failed requires a real E2E artifact}"
        [[ -f "$E2E_ARTIFACT" ]] || die "E2E artifact missing: $E2E_ARTIFACT"
        E2E_PASS=0
        [[ "$E2E_STATUS" == "passed" ]] && E2E_PASS=1
        E2E_MODEL_REVISION="$(python3 - "$E2E_ARTIFACT" "$E2E_PASS" "$BINARY_ID" "$BUNDLE_ID" <<'PY'
import json, sys
path, expected_pass, binary_id, bundle_id = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    artifact = json.load(handle)
if artifact.get("schema_version") != "arle.operator-e2e/v1":
    raise SystemExit("E2E artifact has wrong schema")
if artifact.get("passed") is not (expected_pass == "1"):
    raise SystemExit("E2E artifact verdict differs from requested status")
if artifact.get("binary_id") != binary_id or artifact.get("bundle_id") != bundle_id:
    raise SystemExit("E2E artifact identity differs from probe binary/bundle")
model = artifact.get("model_revision", {})
if model.get("kind") != "actual" or not model.get("id"):
    raise SystemExit("E2E artifact must name an actual model revision")
print(model["id"])
PY
)" || die "invalid E2E artifact"
        E2E_ARTIFACT_SHA256="$(cuda_prebuilt_hash_file "$E2E_ARTIFACT")"
        E2E_MODEL_KIND="actual"
        ;;
    *) die "ARLE_SMALLM_E2E_STATUS must be passed, failed, or not_run" ;;
esac

export ARLE_SMALLM_BINARY_ID="$BINARY_ID"
export ARLE_SMALLM_BUNDLE_ID_SOURCE="$BUNDLE_ID_SOURCE"
export ARLE_SMALLM_MODEL_KIND="$MODEL_KIND"
export ARLE_SMALLM_MODEL_REVISION="$MODEL_REVISION"
export ARLE_SMALLM_E2E_STATUS="$E2E_STATUS"
export ARLE_SMALLM_E2E_PASS="$E2E_PASS"
export ARLE_SMALLM_PROBE_ITERS="${ARLE_SMALLM_PROBE_ITERS:-200}"
export ARLE_SMALLM_PROBE_SAMPLES="${ARLE_SMALLM_PROBE_SAMPLES:-5}"
if [[ -n "$BUNDLE_MANIFEST_SHA256" ]]; then
    export ARLE_SMALLM_BUNDLE_ID="$BUNDLE_ID"
    export ARLE_SMALLM_BUNDLE_MANIFEST_SHA256="$BUNDLE_MANIFEST_SHA256"
else
    unset ARLE_SMALLM_BUNDLE_ID ARLE_SMALLM_BUNDLE_MANIFEST_SHA256 || true
fi
if [[ -n "$E2E_ARTIFACT_SHA256" ]]; then
    export ARLE_SMALLM_E2E_ARTIFACT_SHA256="$E2E_ARTIFACT_SHA256"
    export ARLE_SMALLM_E2E_MODEL_KIND="$E2E_MODEL_KIND"
    export ARLE_SMALLM_E2E_MODEL_REVISION="$E2E_MODEL_REVISION"
else
    unset ARLE_SMALLM_E2E_ARTIFACT_SHA256 ARLE_SMALLM_E2E_MODEL_KIND \
        ARLE_SMALLM_E2E_MODEL_REVISION || true
fi

echo "[run_fp8_probe] binary_id=$BINARY_ID bundle_id=$BUNDLE_ID model=$MODEL_KIND:$MODEL_REVISION e2e=$E2E_STATUS" >&2
"$PROBE_BIN" >"$OUTPUT"
echo "[run_fp8_probe] wrote $OUTPUT" >&2
