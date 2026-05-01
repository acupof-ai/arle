#!/usr/bin/env bash
# Build/package an H20 single-node Phase 1 smoke bundle.
#
# Default behavior builds the CUDA release binary with SM90 requested, then
# creates dist/arle-h20-phase1-<commit>.tar.gz. Set SKIP_BUILD=1 to package an
# existing binary from INFER_BIN or target/release/infer.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMMIT="$(git rev-parse --short HEAD)"
OUT_DIR="${OUT_DIR:-$ROOT/dist}"
BUNDLE_NAME="${BUNDLE_NAME:-arle-h20-phase1-${COMMIT}}"
STAGE="$OUT_DIR/$BUNDLE_NAME"
TARBALL="$OUT_DIR/${BUNDLE_NAME}.tar.gz"
INFER_BIN="${INFER_BIN:-$ROOT/target/release/infer}"

mkdir -p "$OUT_DIR"
rm -rf "$STAGE" "$TARBALL"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
    export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90}"
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" cargo build --release -p infer --features cuda
fi

if [[ ! -x "$INFER_BIN" ]]; then
    echo "error: infer binary not found or not executable: $INFER_BIN" >&2
    echo "       set INFER_BIN=/path/to/infer or run without SKIP_BUILD=1" >&2
    exit 2
fi

mkdir -p \
    "$STAGE/bin" \
    "$STAGE/scripts" \
    "$STAGE/infer/models/Qwen3-4B" \
    "$STAGE/docs/experience/wins"

cp "$INFER_BIN" "$STAGE/bin/infer"
cp scripts/bench_guidellm.sh "$STAGE/scripts/bench_guidellm.sh"
cp scripts/bench_sglang_longctx.sh "$STAGE/scripts/bench_sglang_longctx.sh"
cp scripts/smoke_h20_phase1.sh "$STAGE/smoke_h20_phase1.sh"
cp requirements-bench.txt "$STAGE/requirements-bench.txt"
cp docs/experience/wins/TEMPLATE-bench-guidellm.md \
    "$STAGE/docs/experience/wins/TEMPLATE-bench-guidellm.md"
chmod +x "$STAGE/bin/infer" "$STAGE/scripts/"*.sh "$STAGE/smoke_h20_phase1.sh"

cat >"$STAGE/README-H20.md" <<EOF
# ARLE H20 Phase 1 Smoke Bundle

Commit: $COMMIT

Before running:

1. Confirm the node is H20/SM90-class:

   \`\`\`bash
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
   \`\`\`

   The first value must be >= 9.0.

2. Place Qwen3-4B weights under:

   \`\`\`text
   infer/models/Qwen3-4B/
   \`\`\`

   Or run with \`MODEL_PATH=/absolute/path/to/Qwen3-4B\`.

3. Install benchmark-side tools if they are not already present:

   \`\`\`bash
   python3 -m pip install -r requirements-bench.txt
   # jq is also required by scripts/bench_guidellm.sh.
   # Ubuntu/Debian: sudo apt-get install -y jq
   # Conda:         conda install -c conda-forge jq
   \`\`\`

4. Run:

   \`\`\`bash
   ./smoke_h20_phase1.sh
   \`\`\`

The smoke starts \`bin/infer --kv-cache-dtype fp8 --num-slots 16
--max-seq-len 131072 --mem-fraction-static 0.85
--max-num-batched-tokens 16384 --max-prefill-tokens 16384\` and runs the
longctx-32k c=4 300s GuideLLM envelope through \`scripts/bench_guidellm.sh\`.
EOF

cat >"$STAGE/infer/models/Qwen3-4B/README.md" <<'EOF'
Place Qwen3-4B config, tokenizer, and safetensor shards in this directory, or
set MODEL_PATH when running ./smoke_h20_phase1.sh.
EOF

cat >"$STAGE/manifest.txt" <<EOF
bundle=$BUNDLE_NAME
commit=$COMMIT
infer_bin=$INFER_BIN
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST:-}
cmake_cuda_architectures=${CMAKE_CUDA_ARCHITECTURES:-}
EOF

tar -C "$OUT_DIR" -czf "$TARBALL" "$BUNDLE_NAME"

echo "$TARBALL"
