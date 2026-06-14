# DSv4 MoE batching + EPLB — SGLang study & ARLE scope (understand-until-simple)

Commissioned by ckl after Phase B (batched FlashMLA attention): *"这是注意力 还有
moe呢 / 还有eplb呢 / 弄清楚了再做"* + *"以做好做成理想态为目标"*. Question: is the
decode MoE batched, and does ARLE need EPLB? Method: read SGLang's implementation
(`/Users/bytedance/code/sglang`), compare to ARLE, license-or-kill before building.

## TL;DR

- **MoE batching is already done — at SGLang parity.** The batched decode lane runs
  MoE grouped over the whole `[N]` batch via DeepEP low-latency dispatch → DeepGEMM
  **masked grouped GEMM** → EP-reducing combine (`dsv4.rs:1808`, `:2254`
  `dsv4_moe_forward_deepep_ll`). This is the same path as SGLang's `DeepEPMoE` LL
  branch. Unlike attention (per-row until Phase B), MoE was **never per-row** in the
  batched lane. There is no "batched MoE" gap analogous to the attention one.
- **EPLB is completely absent in ARLE** (static `ExpertSplit`, fixed contiguous
  `256/8=32` experts/rank; no load recorder / replication / physical↔logical map).
  SGLang has the full DeepSeek EPLB stack.
- **EPLB is a row-bound (prefill / high-concurrency) lever, not a decode-c=4 lever.**
  ARLE's decode MoE is explicitly **weight-read-bound** (`moe.rs:190,246`): per-expert
  row imbalance is nearly free there, and hot-expert *replication* can *increase* the
  total distinct-expert-tile count at low token volume. **Do not build EPLB blind.**
  License-or-kill it with a cheap per-rank MoE-GEMM wall-time skew measurement at the
  target shapes first (§4).

## 1. SGLang MoE decode path (the thing to match)

`ep_moe/layer.py::DeepEPMoE.forward_impl`: `dispatcher.dispatch` → `run_moe_core` →
`dispatcher.combine`. LL (low-latency, decode) branch:

1. **Pre-dispatch (fused, DSv4-specific)** — `deepseek_v4/mega_moe_pre_dispatch.cuh`:
   one CTA per token, fuses bf16→fp8_e4m3 **per-token-group quant** (UE8M0 scale, the
   DeepGEMM block-scale format) + topk_idx/weights copy + trailing pad to `padded_max`.
   Writes the contiguous `buf_x` / `buf_x_sf` / `buf_topk_*` that DeepEP dispatch reads.
2. **DeepEP LL dispatch** — masked all-to-all to expert-owner ranks; output is
   `[num_local_experts, world*max_tok, hidden]` with per-local-expert `masked_m` counts.
3. **DeepGEMM masked grouped GEMM** (gate/up → SwiGLU → down) over the masked bands.
4. **DeepEP LL combine** — reduces across EP back to original token rows (no separate
   all-reduce).

## 2. SGLang EPLB (the thing ARLE lacks)

`eplb/eplb_algorithms/deepseek.py` (copied verbatim from deepseek-ai/EPLB) +
`eplb_manager.py` + `expert_distribution.py` + `expert_location_updater.py`.

- **Algorithm** (`rebalance_experts`): input `weight:[layers, num_logical_experts]` =
  measured per-expert token load. Hierarchical policy: (1) `balanced_packing` of expert
  *groups* to nodes (keep a group's experts intra-node to cut inter-node traffic);
  (2) `replicate_experts` — greedily add a physical replica to whichever logical expert
  has the highest `load/replica_count`, so `num_physical > num_logical`; (3)
  `balanced_packing` of physical experts to GPUs so each GPU's summed load is even.
  Global policy = hierarchical with `num_groups=num_nodes=1`. Output:
  `physical_to_logical_map`, `logical_to_physical_map`, `logical_count`.
- **Trigger** (`EPLBManager`): a recorder counts tokens/expert/layer continuously;
  every `eplb_rebalance_num_iterations` forwards → `rebalance()`, which **skips** when
  `average_utilization_rate_over_window > eplb_min_rebalancing_utilization_threshold`
  (rebalancing only pays when imbalance is starving the GPU). Re-placement can be
  chunked across layers to bound the stall; `update_expert_location` physically **moves
  expert weights between ranks** — the expensive, dynamic part.
- **Two regimes**: *static* (record offline on a representative load → place once at
  load, no movement) vs *dynamic* (live re-measure + re-move). Static captures most of
  the win at a fraction of the cost.

## 3. ARLE current state

| Piece | ARLE | Verdict |
|---|---|---|
| Decode MoE grouping | DeepEP LL dispatch + DeepGEMM masked grouped GEMM + EP-reduce combine (`dsv4_moe_forward_deepep_ll`); grouped over `[N]`, no per-row MoE scratch (`dsv4.rs:1808`) | **Parity — done** |
| MoE transport default | `dsv4_use_deepep_transport()` defaults to **`allreduce`** (`dsv4.rs:4030`); DeepEP LL is **opt-in** (`ARLE_DSV4_MOE_TRANSPORT=deepep_ll`). Both transports run the grouped GEMM; allreduce path is `dsv4_moe_forward` (EP-sharded → TP all-reduce, `dsv4.rs:2297,2312`) | grouped either way; LL = SGLang-shaped a2a, opt-in |
| Fused pre-dispatch quant+pack | per-token FP8 quant + pack present in the LL path | parity (kernel-level fusion gap, if any, is a separate micro-lever) |
| Expert placement | static `ExpertSplit` (contiguous `256/world` per rank, `dsv4.rs:925`) | **EPLB absent** |
| Expert-load recorder | none | absent |
| Expert replication (`num_phy>num_log`) | none | absent |

## 4. EPLB license-or-kill for ARLE's decode target (§0 discipline)

**The skew EPLB targets vs the skew ARLE measured are not yet proven to be the same
variable.** The c-sweep mechanism研究
([wins](../experience/wins/2026-06-14-dsv4-batched-decode-csweep-threshold-n4.md))
measured "AllReduce per-rank skew 4–9× max/avg (lockstep-wait)" and attributed it to
the **TP Q-allgather** (`flashmla_q_allgather` = 10.4%, the biggest single collective),
naming **DP-attention** as the #2 lever — EPLB was *not* in that ranking. Whether any
of that skew is MoE-GEMM row imbalance (EPLB's target) vs Q-allgather/TP (DP-attn's
target) is a **confounded variable that must be isolated** before EPLB is licensed.

Why EPLB is likely a decode wash (hypothesis, to be measured, not asserted):
- Decode MoE is **weight-read-bound** (`moe.rs:190,246`). At c=4, ~`4×topk` token-expert
  pairs spread over 256 experts ⇒ most experts get 0 tokens, a few get 1–4. Per-rank
  cost ≈ (# distinct hit experts on that rank) × one weight-read tile — the # of *rows*
  inside a hit expert barely moves the tile cost. EPLB balances *rows*; at decode the
  imbalance is in *which experts are hit*, and **replicating** a hot expert into 2 ranks
  *adds* a tile (more total weight-reads), not removes one. EPLB's row-balancing win is
  real only in the **row-bound** regime: prefill / high concurrency / many tokens.

**Cheap experiment that licenses or kills EPLB (run before any code):**
1. On the pod (8×H20, TP8/EP8), instrument per-rank MoE grouped-GEMM wall time (CUDA
   events around the masked GEMM) at **c=4, c=8** (decode, ~2300-tok prod prompt) and at
   a **prefill chunk** (row-bound control). Report `max/avg` across the 8 ranks.
2. **Kill** for decode if `max/avg ≈ 1.0–1.3` at c=4/c=8 (balanced → nothing to win).
   **License** if `max/avg ≥ ~2×` *and* it's MoE-attributable (not the Q-allgather).
3. If the prefill control shows large skew but decode doesn't, EPLB is a
   **throughput/prefill lever**, orthogonal to ARLE's c=1–8 decode focus
   ([[feedback_metal_focus_c1_local]] is Metal; DSv4 target band is the pod's serving
   concurrency — pin it before building).

## 5. Recommendation (理想态)

1. **MoE batching: nothing to build.** The lane already groups MoE at SGLang parity.
   The only pending MoE-relevant action is the **N=4 batched-decode default flip**
   (approved) — that turns the whole grouped lane on at c≥4; it is the +58% @c=8 win.
2. **EPLB: measure first (§4), build second.** If licensed, ship **static EPLB**
   (offline load profile → placement at load, no live movement) before dynamic — it's
   the unified `infer-core`/`infer-seam` placement abstraction
   ([[feedback_unified_abstraction_not_per_model]]), not a DSv4 special-case: a load
   recorder + the `rebalance_experts` port (pure tensor ops) + a physical↔logical
   indirection in the EP dispatch.
3. **Lever ranking unchanged from the c-sweep研究:** Phase B (done) → **DP-attention**
   (targets the *measured* Q-allgather skew) → CUDA graph. EPLB enters this ranking
   only if §4 attributes real, decode-band MoE skew to it; otherwise it's a separate
   prefill-throughput track.

## Sources

- SGLang: `eplb/eplb_algorithms/deepseek.py`, `eplb/eplb_manager.py`,
  `layers/moe/ep_moe/layer.py`, `jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh`,
  `layers/moe/token_dispatcher/deepep.py`.
- ARLE: `infer-cuda/src/dsv4.rs` (`forward_decode_batch_stream_impl`, `:1808`, `:2254`),
  `infer-cuda/src/moe.rs` (`:190,246` weight-read-bound), `infer-cuda/src/deepep.rs`
  (LL dispatch/combine), `infer-cuda/src/moe_config.rs` (`ExpertSplit`).
- Prior measurement:
  [c-sweep N≈3 + mechanism](../experience/wins/2026-06-14-dsv4-batched-decode-csweep-threshold-n4.md).
