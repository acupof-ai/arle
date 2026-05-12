# M_quant KV W4A8 — INT4 K/V cache + FP8 attention(orthogonal axis 跟 weight 量化叠加)

> 2026-05-08。Triggered by user directive "支持好量化算子 w4a8 可接受 fp4 是
> 未来的主流" + "KV 走 W4A8" + master strategy §0.1 主战场 axis 2(量化全套)。
>
> Task #33 from `M_quant-fp8-w4-magnitude-path.md`。**KV axis P0**(per
> `master-strategy.md` §1.2.1.B),orthogonal 跟 **Weight axis**(W4A8 Marlin
> 当前 codex 在 implement,task #25 W8A8 already KILL,W4A16 license fires
> `f6f3af3`)。
>
> 两 axis 设计为 **stack-able**:
> - Weight: BF16 / FP8 / W4A8 (Marlin) / W4A16 (Marlin)
> - KV: BF16 / FP8(production) / INT8(production) / **W4A8(this plan,P0)**
>
> 任何 weight × KV combination 都该 viable(weight axis 影响 prefill compute,
> KV axis 影响 decode + long-ctx memory bandwidth,orthogonal)。

## §0 硬件常数(per `M_quant-fp8-w4-magnitude-path.md`)

| Resource | Value |
|---|---|
| HBM bandwidth | 672 GB/s |
| FP8 tensor mma (sm_89 native) | 706 TFLOPS |
| L2 cache | 48 MB |
| GPU memory budget | 16 GB |

## §1 KV memory formula

per token KV bytes (Qwen3-4B,GQA 8 KV heads,head_dim=80,36 layers):

```
KV_bytes_per_token_BF16 = 36 layers × 2 (K+V) × 8 KV_heads × 80 dim × 2 byte = 92,160 bytes ≈ 92 KB / token
```

| KV format | bytes/tok | KV pool capacity (16GB - weight 8GB - other 4GB = 4GB KV budget) |
|---|---:|---:|
| BF16 (baseline) | 92 KB | 21,786 tokens |
| FP8 E4M3 (production) | 46 KB | 43,572 tokens |
| INT8 (production) | 46 KB | 43,572 tokens |
| **W4A8 INT4 K/V** ⭐ P0 | **23 KB** | **84,000+ tokens** |
| INT2 (extreme,accuracy 风险) | 11 KB | ~170,000 |

**W4A8 KV 实证 magnitude**:
- KV pool capacity 4× → 直接支持 32k+ ctx batched / multi-tenant 高 concurrency
- decode KV read bandwidth 4× saving → ITL 减(0.56 ms BF16 / 0.14 ms W4)
- 但 4k ctx 时 KV 占 ITL 仅 3%(0.56/19.27)→ **短 ctx 收益 noise level**
- 32k+ ctx KV 占 ITL > 50% → **真正 magnitude regime 在 long ctx**

## §2 ARLE substrate audit(代码为真理)

**已有可复用**:
- `crates/cuda-kernels/src/kv_quant.rs::decode_attention_int8`(INT8 KV decode attention,production)
- `crates/cuda-kernels/src/kv_quant.rs::decode_attention_fp8`(FP8 KV,production)
- `crates/cuda-kernels/src/kv_quant.rs::decode_attention_varlen_fp8`(FP8 KV varlen)
- `infer/src/quant.rs::Fp8Config`/`Int8Config`(KV format config)
- `infer/src/main.rs::VALID_KV_CACHE_MODES`("auto" / "bf16" / "fp8" / "int8")
- KV pool / paged write / scatter quant infra(`kv_quant.rs::quantize_paged_kv_fp8` 等)

**缺**:
- `decode_attention_w4_a_fp8.cu` 新 kernel(~300-400 LOC,based on `decode_attention_int8`/`fp8` 改 INT4 unpack + FP8 mma)
- `quantize_paged_kv_w4` quantizer(per-block FP8 scale + INT4 packing)
- `KVCacheDtype::W4A8` enum variant
- `--kv-cache-dtype w4a8` CLI flag
- `attention.rs` dispatch case

**TurboQuant W2/W3/W4 删除后**(codex Phase B current work)KV format enum 简化为 BF16/FP8/INT8 + 新加 W4A8。

## §3 Phase 0 license-or-kill(LOC budget 400)

### Phase 0a — cheap smoke(1h codex,SOLID first step)

新 file `/tmp/kv_w4a8_smoke.cu`(独立 binary,不入 build):
- 单 attention layer 测 INT4 K/V dequant + FP8 mma
- shape:c=4 batch / seq_len=4096 / head_dim=80 / 8 KV heads
- 对照 BF16 KV decode attention(`decode_attention_fp8` 同 shape FP8 KV)
- iterate 100 次 + cudaEvent,output mean/std

**License gate**(per §0 SOLID rule 6 wall-clock framing):
- speedup 跟 BF16 attention 比 ≥ **2×** → ✅ Phase 0b implement
- 1.5-2× → ⚠ borderline,看 KV bandwidth 占 ITL 比是否 > 50%
- < 1.5× → ❌ KILL,INT4 unpack overhead 反吃 bandwidth saving

**2026-05-12 Phase 0a-narrow result(sm_89 scan only): no runtime license.** Added
`scripts/kv_w4a8_smoke.cu` as a standalone CUDA probe for KV read + scale +
dequant scan (not full attention). Hot repeated runs on RTX 4070 Ti SUPER:

| path | hot time | vs BF16 time | vs FP8 time |
|---|---:|---:|---:|
| BF16 scan | ~732.4 us | 1.000x | n/a |
| FP8 E4M3 + f32 scale scan | ~694.4 us | 1.055x | 1.000x |
| FP4 E2M1 LUT + f32 scale scan | ~948.9 us | 0.772x | 0.732x |
| FP4 E2M1 bit-dequant + f32 scale scan | ~493.8 us | 1.483x | 1.407x |

Evidence: [`docs/experience/errors/2026-05-12-kv-w4a8-fp4-sm89-scan-kill.md`](../experience/errors/2026-05-12-kv-w4a8-fp4-sm89-scan-kill.md).

This does **not** kill every possible FP4 attention design, because the smoke
does not include register-resident FP4->FP8 unpack plus FP8 MMA. It does kill
the naive runtime direction for sm_89: the bit-dequant scan is only ~1.48x
faster than BF16 and ~1.40x faster than FP8, below this plan's 1.5x Phase 0a
license floor, so no `w4a8` KV enum/CLI/runtime dispatch until a
full-attention smoke passes in the long-context wall-clock regime.

### Phase 0b — implement(LOC budget 400)

| 文件 | LOC est | 内容 |
|---|---:|---|
| `crates/cuda-kernels/csrc/attention/decode_attention_w4_a_fp8.cu` | ~250 | INT4 K/V unpack 内联 + FP8 mma QK^T + V proj |
| `crates/cuda-kernels/csrc/quant/quantize_kv_w4.cu` | ~80 | per-block FP8 scale + INT4 packing |
| `crates/cuda-kernels/src/kv_quant.rs` | ~50 | Rust FFI `decode_attention_w4_a_fp8` / `quantize_paged_kv_w4` |
| `infer/src/quant.rs` | ~10 | `KVCacheDtype::W4A8` enum + `Int4KvConfig` |
| `infer/src/main.rs` | ~10 | `--kv-cache-dtype w4a8` |
| `infer/src/ops/attention.rs` | ~15 | dispatch case W4A8 |
| `infer/tests/e2e.rs` + `greedy_consistency.rs` | ~50 | W4A8 KV test |
| `crates/cuda-kernels/build.rs` | ~5 | new .cu files |
| **合计** | **470** | (10% over budget,acceptable) |

### Phase 0c — bench(2 shapes:short + long ctx)

```bash
# Setup:weight BF16 baseline + KV W4A8(只验 KV axis 单 axis effect)
nohup ./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --kv-cache-dtype w4a8 \
  --port 8000 --num-slots 8 --max-seq-len 32768 \
  > /tmp/infer-kv-w4a8.log & disown

# Short ctx 4k/c=4(预期 KV 占 ITL 3%,收益 noise level,作 sanity)
scripts/bench_guidellm.sh m_quant-kv-w4a8-c4-r1 \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,...,output_tokens=256,...'

# Long ctx 32k/c=2(KV bandwidth dominant regime,真 magnitude)
scripts/bench_guidellm.sh m_quant-kv-w4a8-32k-c2 \
  --concurrencies 2 --max-seconds 180 --warmup 10 \
  --data 'prompt_tokens=32768,...,output_tokens=256,...'
```

n=3,σ。

### Phase 0d — License decision per wall-clock framing(§0 SOLID rule 6)

**Short ctx 4k/c=4**:KV 占 ITL 仅 3%,W4A8 KV 实测 ITL 改善 ≤ 1% within noise → SOLID 不 license on 4k 数字。

**Long ctx 32k/c=2**(真 binding):

| KV W4A8 vs BF16 KV @ 32k | Action |
|---|---|
| ITL ≥ **1.5×** faster | ✅ License,production opt-in `--kv-cache-dtype w4a8` default-off → 验完 stability default-on |
| 1.2-1.5× | ⚠ Marginal,留 substrate 不 default,document trade-off |
| < 1.2× | ❌ KILL,KV bandwidth 不是 32k longctx ITL 主 binding,重审 |
| greedy diff > 5% | ❌ KILL,W4 KV accuracy 不达标(K/V 比 weight 更 sensitive) |
| KV pool capacity 实测 < 80k(理论 84k)| ⚠ packing 效率不足,debug |

## §4 Stack on weight axis(W4A8 weight + W4A8 KV combined)

完整 stack 三种 setup:

| Setup | Weight | KV | Predicted Effect |
|---|---|---|---|
| Baseline | BF16 | BF16 | TTFT 1976 / ITL 19.27 / out 153.83 (`786a20a`) |
| Weight only | W4A16 Marlin | BF16 (auto-FP8) | TTFT 2565 / ITL 11.76 / out 191 (`f6f3af3`,实测) |
| KV only | BF16 | **W4A8** | TTFT ~1976 / ITL ~19 short ctx,**32k 大幅改善**(待测) |
| Combined | **W4A8 Marlin + W4A8 KV** | | 短 ctx ITL ~10ms / 32k ctx 大幅 / TTFT 待 weight bench `f6f3af3+` 数据 |

**stack 优势**:
- Weight axis 主 decode bandwidth(weight read 占 ITL ~62%)
- KV axis 主 long-ctx bandwidth + KV pool capacity(short ctx 收益少,long ctx 主导)
- 两 axis orthogonal 不互相吃,可线性叠加 magnitude

## §5 Cross-references

- Master strategy §0.1 axis 2(量化全套)+ §1.2.1.B(KV axis P0)
- Weight axis plan:`M_quant-fp8-w4-magnitude-path.md`(Phase 0 W8A8 KILL,Phase 0 v2 cutlass FP8 KILL,W4A16 Marlin license `f6f3af3`,W4A8 Marlin codex 当前 implement)
- ARLE `kv_quant.rs` substrate(INT8/FP8 KV production)
- TurboQuant 删除(codex Phase B current work)— 删完后 KVFormat enum 简化为 BF16/FP8/INT8 + 加 W4A8

## §6 SOLID 检查

✅ Audit first(§2 ARLE substrate inventory before write LOC est)
✅ §0 rule 6 wall-clock framing(short ctx 占比公式 + long ctx magnitude regime)
✅ License threshold quantified(1.5× / 1.2× / <1.2× three-bucket)
✅ Cheap smoke first(Phase 0a 单 op test before Phase 0b 400 LOC)
✅ Greedy correctness gate(KV 比 weight 更 sensitive,5% diff threshold)
✅ Stack on weight axis license fires evidence(`f6f3af3`,not greenfield)
✅ Long ctx 32k/c=2 是 magnitude regime,short ctx 不 license
