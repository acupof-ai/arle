# dsv4_parity was suspected of an unconditional pass; the verdict was already real

Date: 2026-09-13. Investigation of the `dsv4-parity-batch-wiring` agenda row,
whose exit assumed the multi-rank `dsv4_parity` gate reported PASS without
observing a verdict.

## Finding: no unconditional success exists

The positive verdict is enforced at three layers, and it is shell-tested
both directions.

1. **Example** (`crates/infer-cuda/examples/dsv4_parity.rs`): each rank
   returns `Err` on load, NCCL rendezvous timeout, or forward failure and
   prints `clean_tokens=[...]`. The non-CUDA stub returns `Ok(())` after an
   `eprintln!`, but the multi-rank launcher only ever invokes a
   cuda,nccl-built binary.
2. **Launcher** (`scripts/dsv4_multigpu_parity.sh`): `set -euo pipefail`;
   waits every rank and exits 1 on any rank failure; parses rank 0's first
   `clean_tokens` entry against the oracle `11111`; prints the verdict and
   exits 1 on mismatch.
3. **Batch** (`scripts/parity_gpu_batch.sh`, `run_model_gate`): captures
   the launcher's unpiped `$?` and an `ALL PASS` marker; PASS requires both;
   the overall batch exits 0 only when `n_fail == 0`. SKIP is a separate
   recorded state with a "NEVER EXECUTED" section, never counted as pass.

`scripts/tests/test_parity_gpu_batch.sh` mocks the binary both ways: the
correct token yields `results.tsv ... ALL PASS ... PASS` and rc 0; token
`99999` yields the launcher's `FAIL: rank0 first token 99999 != 11111`,
a FAIL row, and rc 1.

A row can be wrong about what is broken. The honest outcome of the
investigation is to disprove the premise, not to manufacture a fix.

## What was actually wrong: the verdict text overstated its scope

The gate verifies only the FIRST of 16 oracle tokens — the full-prefix
prefill argmax over SW/CSA/HCA recomputation. The other 15 require
incremental `start_pos > 0` decode with per-step KV reuse, which currently
bails (the example records the bail step in `bail_at` and says so on
rank-0 stderr). A bare `ALL PASS` line therefore read as 16/16 to anyone
looking at `results.tsv` months later without the launcher source in front.

Fix (verdict text only, no comparison or bound change): rank 0 now emits a
machine-readable `gate_scope=` line —

- no bail: `gate_scope=first-token+<k>-incremental(<n> tokens total)`;
- bail: `gate_scope=first-token-only(incremental bail at step <k>)` —

and the launcher emits `ALL PASS gated=<scope>` / `FAIL ... (gated=<scope>)`.
The scope is stated positively (what was checked) and the bail is visible
in the TSV cell, not only in rank-0 stderr. The batch shell test asserts
the scoped string. `pos_marker` still keys off the `ALL PASS` fixed
substring, so the runner integration is unchanged.

## The recurring shape: correct plumbing, zero executions

This is the third population of its kind found the same day. The
`dsv4_parity` wiring is correct and shell-tested with mocks, but the real
multi-rank gate has never executed: it SKIPs unless `INFER_DSV4_MODEL_PATH`
is set AND a free N-GPU SM90 set can be reserved, and neither holds today
(model absent; all eight GPUs assigned to the long 27B/30B serving job).
The same "wiring right, nothing has exercised it" form was found for 16
device-requiring `cuda-kernels` tests and for the Vulkan/fixture
require-knobs that no runner sets. Real execution of dsv4_parity folds into
the GPU-assignment-blocked queue; the 1-of-16 token coverage question is a
scope decision tied to the incremental-decode follow-up, not a verdict
defect.

## Rule

Before "fixing a false pass", read the verdict source and prove the pass is
false — a row can misname the defect. When a gate's verdict covers a
deliberately smaller scope than its name implies, the verdict text itself
must state the scope positively; a future reader sees the results table,
not the source.
