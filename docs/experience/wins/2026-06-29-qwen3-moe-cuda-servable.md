# Vanilla Qwen3-MoE on CUDA (Qwen3-30B-A3B) — 6 structural blockers cleared, numerical debug remaining

## Status: NOT shipped (forward mis-computes). Work saved as a patch + agent context.

gap-3 of the unified model-support plan. Goal: serve vanilla `Qwen3MoeForCausalLM`
(`model_type=qwen3_moe`, e.g. Qwen3-30B-A3B) on CUDA, which the backend rejects
("use --backend metal"). Verdict after the work: routing a vanilla **full-attn,
ungated, HD128, no-shared-expert** MoE through the gated-delta-**HD256 hybrid**
qwen35 executor is a deep undertaking — far more than a config adapter.

## What was cleared (each a real fix, all verified to compile + advance)

On the H20, vanilla Qwen3-30B-A3B went from "rejected" → **loads + serves + runs
the full forward** on CUDA. Blockers cleared, in the order they surfaced:
1. **Routing** — `loaded.rs` `classify_cuda_model`: `qwen3_moe` → new `Qwen3Moe`
   kind → `from_qwen35_safetensors` (was `Qwen3MoeUnsupported`).
2. **Config adapter** (`qwen35-spec`): `from_qwen3_moe_json` — flat schema, prefix
   `model.`, `full_attn_gated=false`, `shared_expert_size=0`, all-FullAttention,
   relaxed linear-attn validators. `plain_model_prefix` flag (default false).
3. **Per-head gate optional** (`qwen35.rs` + 4 prep/gate `.cu` kernels gain a
   `q_gated` param): ungated `q_proj` reads `head_dim` stride, gate kernels no-op.
4. **Shared-expert optional** (`moe.rs` + `loader.rs`): `add_shared_expert` early-
   returns when none; shared tensors `Option`, skipped at load.
5. **RoPE table** (`qwen35.rs` ~2505): size to `.max(max_seq_len)` (the 131072
   serve default exceeded the 40960 checkpoint context).
6. **Recurrent-state optional** (`qwen35.rs`): decouple "armed" from `gdr_states`;
   `acquire_recurrent` short-circuits when `num_linear==0` (vanilla has zero
   linear-attn layers). Audit confirmed gated-delta forward code is dead per-layer
   dispatch for a pure full-attn model.
7. **kv4 attention kernels**: HD128 paged kernels were kv8-only; Qwen3-30B is
   **q32/kv4**. Added `q32_kv4` HD128 prefill+decode configs to `kernels.toml`
   (templates are kv-parameterized → no template rewrite); FFI auto-resolves
   `(128,32,4,Prefill/Decode)`.

## The wall: numerically incorrect output

With all cleared, Qwen3-30B-A3B serves and the forward runs end-to-end (no crash),
but the completion is **garbage** ("UGC␣ trolls gariated…") — a remaining
**numerical bug**. Suspects (need decode-level isolation, A/B vs the Metal/HF
reference): (a) per-expert→stacked **MoE weight layout** in the loader (vanilla
ships `experts.{i}.gate_proj`, the qwen35 MoE may expect stacked `gate_up_proj`);
(b) the **ungated q-proj stride** in the prep kernel; (c) the new **kv4 GQA**
grouping. A focused numerical-correctness pass, not a structural fix.

## Decision

NOT committed (won't ship a garbage-output path; `qwen3_moe` stays rejected). The
full 2003-line diff is preserved as a patch (session scratch `gap3-wip.patch`);
the implementing subagent retains full context to resume the numerical debug.
gap-3 is the lowest-priority target (Qwen3-30B serves on Metal today). gap-1
(dense batched) and gap-4 (122B GQA-TP) — the tractable structural wins — shipped.

## Rule

Routing a non-hybrid model through the gated-delta hybrid executor is **whack-a-
mole**: every hybrid assumption (gate, shared-expert, linear-attn validators,
RoPE sizing, recurrent-state, kernel shape) must be conditionalized — feasible,
but a forward-path project, not an adapter. And **structural success ≠ numerical
correctness**: a model that loads + serves + runs without crashing can still emit
garbage; the correct-inference gate (decode the actual tokens) is the only proof.
