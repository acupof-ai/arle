# Multi-GPU + cutover roadmap — porting TP/EP/DeepEP/DeepGEMM into the new seam

Synthesized from a 4-agent parallel port-design analysis (workflow `wpw2xu66s`). This is the
execution plan for the expanded goal: **delete legacy `infer/` · elegant/extensible arch ·
Metal+CUDA verified · TP+EP verified · DeepEP+DeepGEMM working.** Honest: this is multi-phase,
not one session. The current new stack is **BF16-dense-Qwen3 only**; the four axes below were
deliberately dropped in the deletion-refactor and must be cleanly re-ported, each verified, *then*
legacy deleted.

## Extensibility verdict ("易于扩展") — VALIDATED, with one known gap

Every axis ports **cleanly below the executor seam** via the existing contracts — adding an axis is a
`ModelArch`/`Communicator` impl, **not a scheduler change** (the deep-module thesis holds):

| Axis | Seam fit | Effort | Plugs into |
|---|---|---|---|
| **TP** (tensor parallel) | below executor, inside `ModelArch::forward` | medium | `Communicator::all_reduce` per layer |
| **EP + DeepEP** (MoE all-to-all) | below executor, inside the MoE `ModelArch` | medium | `Communicator::all_to_all` (wrap DeepEP in the CUDA `Communicator` impl) |
| **DeepGEMM** (FP8 grouped GEMM) | below executor, an `infer-cuda` ops variant | medium | format-dispatched expert FFN in `CudaModel::forward_tokens` |
| **DSv4 + full quant** (MLA/FP8-KV/INT8) | below executor (KV variant + model) | **xlarge** | `KvPool` variant + model |

**The one design gap (already recorded, arch §8 Fix 4):** `Communicator` is **flat** (one implicit group).
TP *alone* works flat; **TP×EP×PP composition needs hierarchical process-groups / a device mesh**
(named groups per axis, SGLang `parallel_state` pattern). Upgrade `Communicator` with
`new_process_group(ranks, tag)` before multi-axis paths (DP-attention+EP for Qwen3.6-MoE). Single-axis
ports don't need it.

## Universal prerequisite

**CUDA GPU parity on H20** (Qwen3 BF16 greedy bit-identical vs legacy) gates *everything* — it proves the
clean `infer-cuda` forward is correct on real hardware. **In progress** (Codex, pod tmux: Qwen3-0.6B on
`/data01/models`, building with `RUSTUP_TOOLCHAIN=1.92.0`). Fix 0 sampling is done (+ HTTP wiring).

## Phases (sequenced; each ends with an H20 verification gate)

0. **CUDA Qwen H20 parity** — *in progress*. Gate: new vs legacy greedy token ids match.
1. **TP** — port `tensor_parallel.rs` sharding math + `LayerCommunicator` call sites → `Communicator::all_reduce` inside the Qwen3 `ModelArch`; NCCL `Communicator` impl wrapping `cuda-kernels/collective.rs`. Gate: TP=1 == single-rank; TP=8 mock == TP=1; TP=8 real NCCL == mock (8×H20).
2. **EP + DeepEP** — Qwen3.5/3.6 MoE `ModelArch` (single-GPU grouped-expert first) → then `all_to_all` dispatch/combine wrapping `native_deepep` + `deepep-sys`. Gate: MoE single-GPU parity, then EP multi-rank.
3. **DeepGEMM** — FP8 grouped/blockwise GEMM expert variant in `infer-cuda` ops, format-dispatched; build-gated (`ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE`). Gate: FP8-expert parity vs BF16 within tol.
4. **DSv4 full coverage** (xlarge) — MLA + FP8/INT8 KV + quant variants. Gate: DSv4 long-context needle + greedy parity on H20.
5. **Communicator hierarchy** (Fix 4) — flat→process-groups, when EP+TP must compose.
6. **Cutover** — once the new stack covers legacy's serving surface (HTTP done) + the model/quant coverage above: rewire the 4 shallow consumers (`cli`/`agent`/`train`; `arle` bin has zero `infer::` refs) → **delete `infer/src`** → `cargo check --workspace` proves the new stack stands alone.
7. **CI + PRs** — fixing `main`'s red CI (legacy Clippy/Test) is subsumed by the cutover (deletes the offending code); then the 4 dependabot PRs become mergeable (rebase) and the rewrite PR opens + merges.

## PR state (2026-06-03)

`main` HEAD `cf7a2c6d` is **red** (CI + Metal CI fail — legacy CUDA/clippy). All 4 open dependabot PRs
(#49 hf-hub, #50 gha-all, #51 cargo-all, #52 sha2) inherit that red CI → **not mergeable** until `main`
is green. `main` has also diverged (active legacy-dsv4 work) from the rewrite branch. Handling: hold
until `main` greens (via the cutover or a CI fix), then rebase + merge. Do not merge red-CI PRs.
