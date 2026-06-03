# R6 clean CUDA Phase-0 hang: TileLang paged-attention num_pages/total_pages arg swap

**Status:** pending-remote (H20 clean greedy-parity re-run in flight on the pod)
**Track:** R6 clean-CUDA rewrite (`crates/infer-cuda`), Phase 0 (CUDA greedy parity)
**Commit:** `db85d56e` (fix) on `arch/ideal-inference-engine`

## Context

The clean `infer-cuda` BF16 Qwen3 forward (rewrite) had never run on a real GPU.
First H20 bring-up (Qwen3-0.6B, prompt ids `[785,6722,315,9625,374]`, MAX_NEW=16)
surfaced three bugs in sequence — two already fixed:

1. `SafetensorLoader` O(N²) re-read per tensor → read-once `RefCell` shard cache (`3f5f2ece`).
2. Wrong `hidden_size == heads*head_dim` config assertion (Qwen3 decouples head_dim) → removed (`fe841c62`).
3. **This entry:** the forward launched but never returned — GPU pinned at 100%
   util with no `clean_tokens`, dmesg `Xid 43` (GPU stopped processing) under
   `name=r6-qwen3-parity`.

## What Worked — localization

- Source inspection cleared the pre-GEMM kernels (embedding, rms_norm) and proved
  the GEMM layout matched the row-major `HiddenStates` convention. A non-LAUNCH_BLOCKING
  host backtrace pointed at `cublasGemmEx` (the o_proj GEMM) — but that was a red
  herring: it was just the **first device-sync after an async fault**.
- **`CUDA_LAUNCH_BLOCKING=1` was decisive.** Serializing every kernel pinned the true
  faulting op:
  ```
  cuLaunchKernel → tilelang_batch_prefill_paged_hd128_q16_kv8_run_cuda
    → infer_cuda::attention::run_tilelang_paged → paged_attention → forward_tokens
  ```

## Root Cause

`run_tilelang_paged` passed the two TileLang symbolic-shape args **swapped** vs the
legacy `infer/src/ops/attention.rs` contract (and its explicit comment), in all 8
prefill/decode HD128 arms:

| FFI arg | Correct (legacy) | Clean (buggy) |
|---|---|---|
| `num_pages` (arg 12) | `pool.max_total_pages` (K/V pool **capacity** = k/v_pool first-dim extent) | `meta.num_pages` |
| `total_pages` (arg 13) | page-table length (valid `kv_indices` entries = `meta.num_pages`) | `pool.max_total_pages` |

With the swap, the kernel computed K/V-pool strides as if only `meta.num_pages`
(=1 for a 5-token prompt) pages existed, then walked `max_total_pages` (thousands)
entries over a 1-entry `kv_indices` buffer → out-of-bounds read → illegal memory
access that launches but never returns (Xid 43).

## Fix

Swap the two args back to capacity-first / page-table-length-second in all 8 arms;
add a comment documenting the non-obvious naming. `page_size==16` is separately
enforced by the executor (`SUPPORTED_PAGE_SIZE`), so it is not a co-factor.
Typechecks under `cuda,no-cuda`.

## Rule

- A TileLang AOT kernel arg **named** `num_pages` is the pool capacity, not the
  current request's page count — the AOT wrapper promotes pool/tensor extents into
  kernel args. Always mirror the legacy call site arg-for-arg when porting a paged
  kernel; the param name is not the semantics.
- A host backtrace stuck in a CUDA API after an async launch names the **first sync
  point**, not the faulting kernel. `CUDA_LAUNCH_BLOCKING=1` is the cheap, decisive
  localizer (no rebuild) — reach for it before host-backtrace theorizing.

## Update — a SECOND bug behind the first (under investigation)

After the swap fix, `CUDA_LAUNCH_BLOCKING=1` shows the prefill kernel **still hangs**
— but now with **no Xid** (before: Xid 43). So the swap removed the OOB illegal
access, and a second, *control-flow* bug remains (a spin, not a memory fault). All
4 kernel calls (prefill/decode prep+run) match the legacy contract arg-for-arg.

### Empirical A/B (2026-06-04) — trip-count hypothesis FALSIFIED

Single-variable prompt-length sweep on the H20 (R6_ATTN_DEBUG=1 + CUDA_LAUNCH_BLOCKING=1),
all args confirmed correct each run:

| Prompt | qlen | grid bx | KV trip count | Q-tile | Result |
|---|---|---|---|---|---|
| 5 tokens  | 5  | `ceildiv(5,64)=1`  | `ceildiv(5,64)=1`  | partial | **hang** (no Xid) |
| 64 tokens | 64 | `ceildiv(64,64)=1` | `ceildiv(64,64)=1` | **FULL** | **hang** (no Xid) |
| 70 tokens | 70 | `ceildiv(70,64)=2` | `ceildiv(70,64)=2` | mixed   | **hang** (no Xid) |

This **falsifies** the short-prompt pipeline trip-count-deadlock hypothesis and the
partial-tile-NaN hypothesis:
- qlen=70 has trip count 2 / grid bx 2 yet hangs → not trip-count-1, not single-block.
- qlen=64 is a FULL tile (zero padding rows → zero `exp2(-inf - -inf)=NaN` injection)
  yet hangs → not partial-tile / not NaN-in-PV-operand.

The prefill cubin hangs at **every** geometry → a **fundamental prefill-cubin /
launch wedge on sm_90**, independent of prompt shape.

### Multi-lens root cause (3-agent read-only Workflow, 2026-06-04)

Verdict: a **prefill-specific FullRow-WGMMA wedge** in the TileLang HD128 prefill
cubin. The "decode survives trip-count-1" control does **not** transfer — legacy
routes `max_qlen==1` to the structurally different *decode* kernel
(`is_pure_decode` at `infer/src/ops/attention.rs:1118`), so the *prefill* cubin had
**never been exercised at this SKU before R6**. Top suspects that hang full tiles too:

1. **TileLang 0.1.10 FullRow codegen defect** — corroborated by two prior error
   entries on the identical SKU/symptom:
   `errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md` (0.1.10 FullRow
   miscompile; pin 0.1.9) and `errors/2026-05-30-gated-delta-short-seq-prefill-hang-h20.md`
   (sm_90 prefill 100% util / no Xid). Pod build dir is literally `tl010`.
2. **dyn-shmem mis-sizing** — prefill BLOCK_N=64 needs ~80 KB dynamic shared vs
   decode's ~32 KB; if `gen_tilelang_aot.py`'s heuristic baked decode's budget into
   the prefill launcher, the block live-waits on undersized shared.

Ruled out (high confidence): host stream/sync omission (symptom is 100%-util device
spin, not host idle) and generic `T.Pipelined` trip-1 deadlock (decode survives it).

### Probes (done, 2026-06-04)

- [x] TileLang version resolved by the pod build = **0.1.10** (system `/usr/bin/python3`, no venv).
- [x] `device_kernel.cu`: **no** `mbarrier`/`cp.async`/`wait_group` in either prefill or decode (only `__syncthreads`) → not an async-pipeline/mbarrier wedge.
- [x] prefill dyn-shmem = 49152 B = `q+k+v` tiles = correctly sized (decode 24576) → not mis-sized.

### TileLang 0.1.9 pin — FALSIFIED (2026-06-04)

Installed TileLang 0.1.9 in a dedicated venv, **forced AOT cubin regen** (confirmed
new prefill device-source sha `5ccc…` vs old 0.1.10 `ffea…`), rebuilt, re-ran
`seq_len=64`: **still hangs** (100 % util, no Xid, no `clean_tokens`). So the prior
"pin 0.1.9" verdict from `errors/2026-05-27` does **not** transfer to this case —
**the TileLang version is ruled out as the cause** of the sm_90 HD128 prefill hang.

### Narrowed direction — sm_90a / HD128-FullRow-WGMMA (probe in flight)

The 2026-05-30 H20 win ran the **HD256** prefill FullRow-WGMMA kernel correctly on
sm_90; the **HD128 q16_kv8** prefill (Qwen3-0.6B) may have **never** run on sm_90
before R6. `gen_tilelang_aot.py:538-563` is *supposed* to compile the AOT `.cu` with
`-gencode=arch=compute_90a,code=sm_90a` (the WGMMA-enabling target) when
`cuda_arch==90`. Open question (A vs B): (A) the actual build did **not** apply
sm_90a → build bug; (B) it did, and HD128 FullRow-WGMMA is miscompiled on sm_90a
regardless of version → kernel-level fix (GemmWarpPolicy / tile shape). Read-only
arch + WGMMA-instruction-count probe in flight to decide.

## Verification (pending-remote)

- [ ] Settle A vs B (prefill cubin actual arch + WGMMA instr count; HD256 cubin arch).
- [ ] Apply the indicated fix (sm_90a build flag, or HD128 prefill warp-policy/tile change), rebuild, re-run.
- [ ] clean `clean_tokens` == HF gold `[12095,13,576,6722,315,9625,374,1083,279,6722,315,279,5429,315,9625,13]` (Qwen3-0.6B, greedy, 16 new).
- [ ] Then: separate guarded-`exp2` patch (partial-tile NaN) + multi-shape greedy parity before declaring Phase 0 closed.
