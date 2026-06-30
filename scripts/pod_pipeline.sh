#!/usr/bin/env bash
# pod_pipeline.sh — sync-changed-source → incremental build (exit-gated) → verify on GPUs 4-7.
#
# The dev loop for pod (H20) work: edit code locally (you or codex), run this; it
# pushes only the CHANGED source files (so cargo stays incremental), rebuilds with
# the proven CUDA env, gates on the real build exit code, and runs a verify command
# pinned to GPUs 4-7. Re-runnable: each call syncs the delta since the last run.
#
# Usage:
#   scripts/pod_pipeline.sh                       # default verify = cargo check
#   scripts/pod_pipeline.sh "<verify command>"    # e.g. a smoke/test/gate, runs in the pod tree
#   FULL=1 scripts/pod_pipeline.sh ...            # force-resync all tracked source (ignore marker)
#
# State: .pod_pipeline_ref (local) records the last-synced commit. GPUs 4-7 only.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
POD=/data01/arle-build
MARK="$ROOT/.pod_pipeline_ref"
VERIFY="${1:-cargo check --release --no-default-features --features cuda,nccl --bin arle 2>&1 | tail -5}"
HEAD="$(git rev-parse HEAD)"

log(){ echo "[pipe $(date +%H:%M:%S)] $*"; }

# 1) SYNC — push only changed source files (committed since the marker + uncommitted + untracked).
if [ "${FULL:-0}" = "1" ]; then
  git ls-files -- '*.rs' '*.toml' '*.cu' '*.cuh' Cargo.lock '**/build.rs' > /tmp/pipe_changed.txt
else
  LAST="$(cat "$MARK" 2>/dev/null || ~/bin/pod "cd $POD && git rev-parse HEAD" 2>/dev/null || echo "")"
  { [ -n "$LAST" ] && git diff --name-only "$LAST" HEAD 2>/dev/null
    git diff --name-only HEAD 2>/dev/null
    git ls-files -o --exclude-standard 2>/dev/null
  } | grep -E '\.(rs|toml|cu|cuh|c|h)$|(^|/)(Cargo\.lock|build\.rs)$' | sort -u > /tmp/pipe_changed.txt
fi
N=$(grep -c . /tmp/pipe_changed.txt || true)
log "sync ${LAST:0:8}..${HEAD:0:8} : $N changed source file(s)"
while IFS= read -r f; do
  [ -f "$f" ] || continue
  tn push "$f" "$POD/$f" >/dev/null 2>&1 && echo "    + $f" || echo "    ! push failed: $f"
done < /tmp/pipe_changed.txt

# 2) BUILD — incremental, the proven CUDA env, gate on the REAL exit code (not a wrapper echo).
log "build (incremental, exit-gated)…"
~/bin/pod "cd $POD && \
  CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 \
  ARLE_DEEPGEMM_ROOT=$POD/crates/cuda-kernels/vendor/deepgemm \
  ARLE_DEEPGEMM_LIBRARY_ROOT=$POD/crates/cuda-kernels/vendor/deepgemm/deep_gemm \
  INFER_TILELANG_PYTHON=/usr/bin/python3 \
  OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu OPENSSL_INCLUDE_DIR=/usr/include \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-} \
  cargo build --release --no-default-features --features cuda,nccl --bin arle 2>&1 | tail -30; \
  echo INCR_BUILD_EXIT=\${PIPESTATUS[0]}" 2>&1 | tee /tmp/pipe_build.log
BEC="$(grep -oE 'INCR_BUILD_EXIT=[0-9]+' /tmp/pipe_build.log | tail -1 | cut -d= -f2)"
if [ "${BEC:-1}" != "0" ]; then log "BUILD FAILED (ec=${BEC:-?}) — see /tmp/pipe_build.log; NOT verifying (stale-binary guard)"; exit 3; fi
echo "$HEAD" > "$MARK"
log "build OK"

# 3) VERIFY — GPUs 4-7 only, in the pod tree.
log "verify (GPU 4-7): $VERIFY"
~/bin/pod "cd $POD && CUDA_VISIBLE_DEVICES=4,5,6,7 bash -lc '$VERIFY'" 2>&1 | tee /tmp/pipe_verify.log
VEC=${PIPESTATUS[0]:-$?}
log "DONE (verify ec=$VEC). build→/tmp/pipe_build.log  verify→/tmp/pipe_verify.log"
exit "$VEC"
