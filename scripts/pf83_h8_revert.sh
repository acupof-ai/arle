#!/usr/bin/env bash
# PF8.3 H8 revert — symmetrically undo the diagnostic patch (81672c3) if
# pf83_h8_verify.sh shows H8 DISPROVEN (no diagnostic fires + kernel still
# fails). Pivots to H1' static-scratch refactor next.
#
# Per docs/plans/M_pf83_h8_fix_patch.md §3 "if H8 disproven: pivot to H1'".

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Sanity: confirm we're at expected commit
HEAD_SHA=$(git rev-parse --short HEAD)
echo "Current HEAD: $HEAD_SHA"

# Show what's between bb4f5c8 (pre-H8) and HEAD
echo
echo "=== Commits between bb4f5c8 and HEAD ==="
git log --oneline bb4f5c8..HEAD

echo
echo "=== Files modified by 81672c3 (H8 patch) ==="
git show --stat 81672c3 | head -10

# Revert just the H8 substrate edit, keep the docs + scripts
echo
echo "=== Reverting 81672c3 only (keep docs/scripts) ==="
echo "(use: git revert 81672c3 to create a revert commit)"
echo "OR (manual): git checkout bb4f5c8 -- crates/cuda-kernels/csrc/gemm/marlin_w4_fp8_kernel.cu"
echo
echo "Then commit: 'revert(cuda): PF8.3 H8 disproven, pivot H1' static-scratch refactor'"
echo
echo "Followed by: cargo build --release -p infer --features cuda (rebuild without diagnostic)"
echo
echo "(This script is read-only — explicit user/codex action required to actually revert)"
