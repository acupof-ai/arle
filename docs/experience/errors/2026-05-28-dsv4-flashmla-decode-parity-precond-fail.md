# DSv4 FlashMLA decode parity/perf probe — STOP at precondition: pod tree is desynchronized, `tn` push broken

## SLO-shape probed? — N (precondition failed before any build/probe could run)

## TL;DR

Brief requested a pod-side parity + perf probe of the `8ebe3ff5`
FlashMLA decode dispatch (with the two ancestor commits `ed4a7b38` arena
+ `b3a33188` indices builder), and a default-on flip of
`ARLE_DSV4_FLASHMLA_DECODE` if both clear. **STOPPED at preconditions
per the brief's hard "If anything fails: stop, write errors entry,
don't flip default" rule.** Default knob stays OFF.

## Context — what the brief assumed vs pod reality

The brief specified:

```bash
~/bin/pod-exec 'cd /sgl-workspace/arle-fresh && git fetch origin \
    && git worktree add /tmp/arle-d4-parity 8ebe3ff5 …'
```

Pod reality (verified `2026-05-28` via `pod-exec`):

| Assumption | Reality |
|---|---|
| `/sgl-workspace/arle-fresh` is the canonical pod workspace | Exists, but contains only `crates/` and `infer/` shells — no `Cargo.toml`, no `scripts/`, **not a git repo** |
| Pod workspace is a git checkout supporting `git fetch origin` | No `.git` anywhere on the pod that ARLE lives in. `/root/.git` is an empty unrelated repo (`master` with no commits); `/data01/build/DeepEP/.git` is DeepEP, unrelated |
| `git worktree add 8ebe3ff5` will materialize the target commit | Impossible — no git history to worktree from |
| Pod source ≈ `origin/main HEAD` (`8ebe3ff5`) with the three target commits' source present | `/data01/build/arle/` (the *actual* working dir with `target/release/infer` + `scripts/bench_dsv4_trace_http.py`) has the Rust side of the three commits, **but is missing the CUDA-side dependencies they need** |

## What's actually missing on `/data01/build/arle/`

Diff vs local `origin/main`:

```
crates/cuda-kernels/csrc/attention/dsv4_fp8_kv_pack.cu        (14318 B)  MISSING
crates/cuda-kernels/csrc/gemm/dsv4_grouped_gemm.cu            (16802 B)  MISSING
crates/cuda-kernels/csrc/misc/arle_dtype_convert.cu             (1095 B)  MISSING
crates/cuda-kernels/csrc/misc/arle_flashmla_csa_prep.cu       (14386 B)  MISSING
crates/cuda-kernels/csrc/misc/arle_flashmla_decode_shim.cu    (16598 B)  MISSING  ← decode dispatch target
crates/cuda-kernels/csrc/misc/arle_flashmla_shim.cu             (5008 B)  MISSING  ← prefill shim
crates/cuda-kernels/csrc/misc/dsv4_tp_attention_repack.cu       (4736 B)  MISSING
crates/cuda-kernels/vendor/flashmla/                       (entire dir)  MISSING  (~27 MB, 871 files)
crates/cuda-kernels/build.rs                                  (md5 diff)  STALE   (1537 B pod vs 1616 B local)
```

The Rust side has `dsv4_flashmla_decode_enabled`, `ensure_fm_decode_arena`,
`use_flashmla_decode`, and the FFI declarations — but every shim and
the entire FlashMLA vendor tree they FFI into is absent. `cargo build`
cannot link.

Existing pod binary `/data01/build/arle/target/release/infer`
(timestamp `May 27 16:38`) confirms this: `strings` on the binary shows
**none** of `ARLE_DSV4_FLASHMLA_DECODE`,
`arle_flashmla_sm90_sparse_decode_*`, or
`arle_dsv4_flashmla_decode_build_indices` — it predates the D-4
dispatch landings. Probing this binary with the env knob on would be a
no-op.

## Root cause

`tn` (the file-sync tunnel) is broken (per the brief itself: *"the tn
tunnel is broken in a way I can't diagnose remotely"*). The pod tree
is a partial snapshot from a `tn push` that completed before the D-4
contract findings (F1-F4 in
[`docs/experience/wins/2026-05-28-dsv4-flashmla-decode-d4-plumbing.md`](../wins/2026-05-28-dsv4-flashmla-decode-d4-plumbing.md))
and the dispatch + arena + indices landings (`b3a33188 → ed4a7b38 →
8ebe3ff5`) needed the new shims + vendor tree to be pushed.

The brief's `scp -J jumpecs-hl.byted.org …` path is available (and
works), but transmitting 27 MB + 871 vendor files + 7 csrc files +
build.rs without a manifest/checksum protocol risks a fresh
half-state. Per CLAUDE.md §0 ("80% SOLID 不够") and
`feedback_no_half_states.md`, that's not an acceptable rescue path —
the correct fix is restoring `tn` (or whatever full-tree sync the user
has been using) so the pod tree matches `origin/main` cleanly.

## What I did NOT do (deliberately, per the brief)

- **No default-on flip.** `dsv4_flashmla_decode_enabled` stays
  `Ok(false)`. No edit to `infer/src/model/deepseek/weights.rs`.
- **No scp dump.** Did not attempt to manually mirror the missing 871
  vendor files + 7 csrc files + build.rs delta via scp — that path
  would create a third partially-synced state on top of the existing
  partial sync.
- **No `git init` on the pod.** Force-creating a git repo in
  `/sgl-workspace/arle-fresh` or `/data01/build/arle/` to satisfy the
  worktree protocol would be a half-state hack; the user's real
  workflow predicts a `tn push` from a local git checkout, not an
  in-place pod-side fetch.
- **No Mac local-tree touches.** The brief's "DO NOT touch Mac local
  tree" + `feedback_commit_only_own_files.md` rule held — sister INT4
  KIVI WIP under `crates/cuda-kernels/csrc/{attention,kv}/...cu`,
  `crates/cuda-kernels/src/{ffi/{attention,kv},kv_quant}.rs`, and
  `infer/src/model/qwen35/{batch_decode,prefill}.rs` left untouched in
  `git status`.

## Fix (out-of-scope for this entry, surfaced for the user)

To unblock the D-4 parity/perf probe:

1. **Diagnose and restore `tn`** — the user-side blocker. Without
   a full-tree push primitive, every pod cycle needs a
   manifest/checksum verification step which is more work than fixing
   the sync tool itself.
2. After `tn push` works, `tn push` from the local `8ebe3ff5` HEAD to
   `/sgl-workspace/arle-fresh/` (or `/data01/build/arle/`), then:
   - `find crates/cuda-kernels -name "._*" -delete` (clean macOS
     AppleDouble artifacts that polluted `find` listings on pod)
   - `~/bin/pod-exec 'cd <workspace> && CUDA_HOME=/usr/local/cuda
     CUDARC_CUDA_VERSION=12060 cargo build --release --features
     cuda,nccl -p infer --bin infer 2>&1 | tail -40'`
   - Resume the brief's parity (env=0 vs env=1, byte-equal or
     `abs_tol=8e-4`) and perf (TPOT ≤ 12 ms/token at 4K) probes
3. Default-on flip lands only after both gates clear.

Alternative if `tn` stays broken: the brief's scp path is workable but
requires manifest discipline (checksum every file, deny on mismatch);
that's an unrelated tooling investment, not in scope for this probe.

## Rule

**Pod state is not a given.** Before running any pod-side probe whose
result will gate a default-on flip:

1. Verify the pod workspace **is** a git repo and at HEAD via
   `git rev-parse HEAD`.
2. Verify the target binary on disk contains the symbol(s) the probe
   exercises (`strings target/release/infer | grep <symbol>`).
3. If either check fails: STOP, write errors entry, do not flip
   defaults — even if the rebuild "looks easy" via scp/manual sync.
4. The presence of `target/release/infer` is not evidence the source
   tree is current; it's evidence *some* source tree was current
   *whenever it was last built*.

## Refs

- Brief (this conversation): pod-side parity + perf probe + default-on flip.
- Plan: [`docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`](../../plans/2026-05-28-dsv4-flashmla-decode-integration.md)
  (hardened-policy — Flash is permanent default once parity validates).
- Dispatch wins entry (the commit under test): [`docs/experience/wins/2026-05-28-dsv4-flashmla-decode-dispatch.md`](../wins/2026-05-28-dsv4-flashmla-decode-dispatch.md).
- CLAUDE.md §0 SOLID — "80% SOLID 不够",
  [`feedback_no_half_states.md`](../../../.claude/memory/feedback_no_half_states.md).
