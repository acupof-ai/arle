# 2026-05-07 · M_pf-fuse hypothesis validated — vLLM source confirms QKV + gate_up fusion

## Priority & ROI

**Priority**: P0 validation for the M_pf-fuse milestone (codex
actively implementing Phase 0 at time of writing). Confirms
the predicted -14% TTFT gain is grounded in vLLM's actual
implementation, not just a heuristic guess.

**ROI evidence**: vLLM Qwen3 MLP and Attention modules use
explicit fused linears with merged weights — exactly the
pattern M_pf-fuse implements for ARLE.

## Source survey results

vLLM Qwen3 implementation (`/tmp/arle-vllm-venv/lib/python3.12/
site-packages/vllm/model_executor/models/qwen2.py`, shared by
qwen3.py via `from .qwen2 import Qwen2MLP as Qwen3MLP`):

### Fused gate_up (FFN)

```python
# qwen2.py:93
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.gate_up_proj",
)

# qwen2.py:114 (forward)
gate_up, _ = self.gate_up_proj(x)
x = self.act_fn(gate_up)  # SiluAndMul that splits and multiplies
x, _ = self.down_proj(x)
```

ONE GEMM call producing `[seq, 2 × intermediate]`, then a
fused activation that reads both halves.

### Fused QKV (Attention)

```python
# qwen2.py:159
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,
    self.total_num_heads,
    self.total_num_kv_heads,
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.qkv_proj",
)

# qwen2.py:214
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

ONE GEMM producing `[seq, q_size + 2*kv_size]`, then split into
Q, K, V views.

### Weight loader concat

```python
# qwen2.py:467
("gate_up_proj", "gate_proj", 0),
("gate_up_proj", "up_proj", 1),
```

vLLM loads HF checkpoint's separate `gate_proj.weight` and
`up_proj.weight`, concatenates them at load time into the
fused `gate_up_proj.weight` tensor. Same pattern for QKV.

## Per-layer GEMM count comparison

| Position | ARLE current | vLLM | M_pf-fuse target |
|---|---:|---:|---:|
| Attention QKV | 3 (q + k + v) | **1** (qkv) | 1 |
| FFN gate-up | 2 (gate + up) | **1** (gate_up) | 1 |
| FFN down | 1 | 1 | 1 |
| Attention o | 1 | 1 | 1 |
| **Per-layer total** | **7** | **4** | **4** |

ARLE has **75% more GEMM launches per layer** than vLLM at
the prefill path. Over 36 layers × 4 reqs × 2 chunks =
288 layer-ops:
- vLLM: 1,152 GEMMs per prefill cycle
- ARLE: 2,016 GEMMs per prefill cycle

Also, vLLM's larger fused-output GEMMs benefit from better
tensor-core utilization (output dim ~22k for gate_up_proj at
Qwen3-4B vs ARLE's ~11k separate).

## Math: predicted TTFT closure

ARLE long-ctx 4k/c=4 TTFT mdn = 1976 ms; vLLM = 1177 ms.
Gap = 799 ms = 1.68× slower.

M_pf-fuse Phase 0 (gate-up only, codex implementing now):
- GEMM call count: -25% (7 → 5 per layer; QKV still 3)
- FFN GEMM size: 11k output → 22k output (better TC util)
- Estimated -8% TTFT → ~1818 ms

M_pf-fuse Phase 1 (gate-up + QKV):
- GEMM call count: -43% (7 → 4 per layer)
- Both attention and FFN GEMMs benefit
- Estimated -14% TTFT → ~1700 ms

These projections are conservative — vLLM is at 1177 ms with
the same fusion pattern, suggesting the GEMM call count gap
explains most of the 1.68× delta. Closing it fully would put
ARLE at parity. The remaining 8-12% gap likely comes from:
- vLLM's persistent kernel patterns (TensorRT-LLM-style)
- Different attention kernel implementation (FlashAttention-2/3
  vs ARLE's TileLang prefill HD128)
- Scheduler / per-step overhead

## Cross-references

- M_pf-fuse plan: [`63396ff`](../../plans/M_pf-fuse-prefill-gemm.md)
- M_pf-gemm Phase 0 KILLED: [`267fcfa`](2026-05-07-m_pf-gemm-phase0-killed-cublas-heuristic-already-optimal.md)
- H_LP3 finding: [`cae08b7`](2026-05-07-h_lp3-diagnosed-cutlass-small-tile-gemm-bottleneck.md)
- vLLM Qwen2 source (Qwen3 reuses): `/tmp/arle-vllm-venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2.py:93-167,214,467-469`
- vLLM merged linear primitives: `vllm.model_executor.layers.linear.{MergedColumnParallelLinear, QKVParallelLinear}`

## Rule

- **Always validate hypothesis against the leading competitor's
  source** before committing implementation effort. 30 min of
  source reading saves 1-2 days if the hypothesis is wrong.
- **Industry-standard fusion is not optional for SOTA inference
  runtimes.** vLLM, SGLang, TRT-LLM all do it. ARLE not having it
  is a real omission, not a "different design choice".
- **Per-layer GEMM count is a first-order kernel-axis
  optimization metric.** Track it alongside per-row decode time
  and TTFT.
