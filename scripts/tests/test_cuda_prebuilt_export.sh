#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/export_prebuilt_cuda_kernels.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
SRC="$TMP/out"
DEST="$TMP/export"
mkdir -p "$SRC" "$DEST"
python3 - "$ROOT/crates/cuda-kernels/build.rs" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()

def function_body(name):
    start = source.index(f"fn {name}(")
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        depth += source[index] == "{"
        depth -= source[index] == "}"
        if depth == 0:
            return source[brace:index + 1]
    raise AssertionError(f"unterminated function: {name}")

contract = function_body("producer_contract")
capabilities = function_body("configured_capabilities")
# FlashQLA GDR is gated by sm_90 presence, not a build env opt-in — the
# capability lands in the producer contract's `capabilities` field.
assert 'capabilities.insert("flashqla"' in capabilities
assert 'spec.sm == "90"' in capabilities
assert 'env_nonempty("ARLE_DEEPEP_DIR")' in capabilities
assert "ARLE_DEEPEP_SIDECAR_PREBUILT" not in capabilities
required = function_body("required_symbols")
assert 'capabilities.contains("flashmla")' in required
assert 'capabilities.contains("fa3")' in required
assert '"arle_flashmla_sm90_sparse_decode_real_kernel_marker_cuda"' not in source[
    source.index("const PREBUILT_REQUIRED_DSV4_SYMBOLS"):source.index("const PREBUILT_MANIFEST")
]
PY
printf kernels >"$SRC/libkernels_cuda.a"
printf tilelang >"$SRC/libtilelang_kernels_aot.a"
printf sidecar >"$SRC/arle_deepep_sidecar"
chmod 755 "$SRC/arle_deepep_sidecar"
cat >"$SRC/arle-cuda-kernels.manifest" <<EOF
schema=3
kernel_build_id=bundle:test
artifact.libkernels_cuda.a.size=$(wc -c <"$SRC/libkernels_cuda.a" | tr -d ' ')
artifact.libkernels_cuda.a.sha256=$(cuda_prebuilt_hash_file "$SRC/libkernels_cuda.a")
artifact.libtilelang_kernels_aot.a.size=$(wc -c <"$SRC/libtilelang_kernels_aot.a" | tr -d ' ')
artifact.libtilelang_kernels_aot.a.sha256=$(cuda_prebuilt_hash_file "$SRC/libtilelang_kernels_aot.a")
artifact.arle_deepep_sidecar.size=$(wc -c <"$SRC/arle_deepep_sidecar" | tr -d ' ')
artifact.arle_deepep_sidecar.sha256=$(cuda_prebuilt_hash_file "$SRC/arle_deepep_sidecar")
EOF

cuda_prebuilt_validate_bundle "$SRC"

cat >>"$SRC/arle-cuda-kernels.manifest" <<'EOF'
kernel_build_id=duplicate
EOF
if cuda_prebuilt_validate_bundle "$SRC" 2>/dev/null; then
    echo "validator accepted duplicate manifest key" >&2
    exit 1
fi
# Restore the valid fixture for the remaining checks.
python3 - "$SRC/arle-cuda-kernels.manifest" <<'EOF'
from pathlib import Path
import sys
path = Path(sys.argv[1])
lines = path.read_text().splitlines()
path.write_text("\n".join(lines[:-1]) + "\n")
EOF

mkdir -p "$TMP/bin"
cat >"$TMP/bin/nm" <<'EOF'
#!/usr/bin/env bash
case "$*" in
    *exact.a) printf '00000000 T required_symbol\n' ;;
    *substring.a) printf '00000000 T required_symbol_suffix\n' ;;
esac
EOF
chmod +x "$TMP/bin/nm"
: >"$TMP/exact.a"
: >"$TMP/substring.a"
PATH="$TMP/bin:$PATH" cuda_prebuilt_archive_has_symbol "$TMP/exact.a" required_symbol
if PATH="$TMP/bin:$PATH" cuda_prebuilt_archive_has_symbol "$TMP/substring.a" required_symbol; then
    echo "validator accepted a substring symbol match" >&2
    exit 1
fi
cuda_prebuilt_export "$DEST" "$SRC"
cmp "$SRC/arle-cuda-kernels.manifest" "$DEST/arle-cuda-kernels.manifest"
cmp "$SRC/libkernels_cuda.a" "$DEST/libkernels_cuda.a"
cmp "$SRC/libtilelang_kernels_aot.a" "$DEST/libtilelang_kernels_aot.a"
cmp "$SRC/arle_deepep_sidecar" "$DEST/arle_deepep_sidecar"

printf corrupt >>"$DEST/libkernels_cuda.a"
if cuda_prebuilt_validate_bundle "$DEST" 2>/dev/null; then
    echo "validator accepted corrupt artifact" >&2
    exit 1
fi


BUNDLE_ID="bundle:$(printf test | cuda_prebuilt_hash_stream)"
PROBE="$TMP/probe"
cat >"$PROBE" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" == --kernel-build-id ]]; then printf '%s\n' '$BUNDLE_ID'; else printf '{}\n'; fi
EOF
chmod +x "$PROBE"
PROBE_DIR="$TMP/probe-bundle"
mkdir "$PROBE_DIR"
printf kernels >"$PROBE_DIR/libkernels_cuda.a"
printf tilelang >"$PROBE_DIR/libtilelang_kernels_aot.a"
cat >"$PROBE_DIR/arle-cuda-kernels.manifest" <<EOF
schema=3
kernel_build_id=$BUNDLE_ID
artifact.libkernels_cuda.a.size=$(wc -c <"$PROBE_DIR/libkernels_cuda.a" | tr -d ' ')
artifact.libkernels_cuda.a.sha256=$(cuda_prebuilt_hash_file "$PROBE_DIR/libkernels_cuda.a")
artifact.libtilelang_kernels_aot.a.size=$(wc -c <"$PROBE_DIR/libtilelang_kernels_aot.a" | tr -d ' ')
artifact.libtilelang_kernels_aot.a.sha256=$(cuda_prebuilt_hash_file "$PROBE_DIR/libtilelang_kernels_aot.a")
EOF
ARLE_SMALLM_PROBE_BIN="$PROBE" ARLE_SMALLM_KERNEL_MANIFEST="$PROBE_DIR/arle-cuda-kernels.manifest" \
    ARLE_SMALLM_MODEL_KIND=synthetic ARLE_SMALLM_MODEL_REVISION=test ARLE_SMALLM_E2E_STATUS=not_run \
    "$ROOT/scripts/run_fp8_probe.sh" "$TMP/probe.json" >/dev/null 2>&1
python3 - "$PROBE_DIR/arle-cuda-kernels.manifest" <<'PY2'
from pathlib import Path
import sys
path = Path(sys.argv[1])
path.write_text(path.read_text().replace("bundle:", "bundle:0", 1))
PY2
if ARLE_SMALLM_PROBE_BIN="$PROBE" ARLE_SMALLM_KERNEL_MANIFEST="$PROBE_DIR/arle-cuda-kernels.manifest" \
    ARLE_SMALLM_MODEL_KIND=synthetic ARLE_SMALLM_MODEL_REVISION=test ARLE_SMALLM_E2E_STATUS=not_run \
    "$ROOT/scripts/run_fp8_probe.sh" "$TMP/probe.json" >/dev/null 2>&1; then
    echo "probe accepted a manifest ID not embedded in its binary" >&2
    exit 1
fi

echo "cuda prebuilt export self-test passed"
