# W4A8 canonical-pack greedy_consistency test — decision tree for outcome

> Codex currently Working(`6m 50s` per pane,1 background terminal):
> re-quantizing Qwen3-4B against canonical `scripts/quantize_qwen3_w4a8.py`
> (W4A8Layer 4-consecutive row + H3b + H3c + H4 + scale_perm comments)。
>
> Codex verbose diagnostic confirmed pack matches PR #31 W4A8Layer output
> exactly:s_group_real vs s_raw max diff 1.7e-05,1.71/1.33 scale
> amplification disappeared,qweight/scales identical to W4A8Layer。
>
> This entry documents what to look for in greedy_consistency outcome and
> next-step decision tree for each scenario。

## Scenario A — greedy_consistency PASSES ✅(probability ~70%)

**Signal**:`cargo test --release -p infer --features cuda --test greedy_consistency::test_w4a8_vs_bf16_token_diff` returns `ok`,token diff < threshold。

**Action**:
1. Mark `task #34` complete
2. Run `scripts/bench_guidellm.sh m_quant-w4a8-canonical` for production W4A8 numbers vs BF16 baseline
3. Decide on `default-on` flip(blocked by graph capture per `62e75ee` plan)
4. Promote `scripts/quantize_qwen3_w4a8.py` validation entry to `wins/`
5. **Strategic implication**:weight axis 1 quant 全套 unblocked,can bench W4A8 vs SGLang/vLLM at 4-shape canonical + W3 agent workload(once W3 503 deadlock fixed per `cb087c7`)
6. Methodology lesson commit:wrong-class identification(per `3cee2f0`)cost ~5 iterations;cheaper would have been "verify upstream class hierarchy first"

## Scenario B — greedy_consistency STILL 100% diff(probability ~30%)

**Signal**:test still fails,token diff at 100% from idx=0。

**Possible remaining bugs**(by audit completeness as of EOD+29):

### B.1 Activation quant array padding(P0)
PR #31 kernel `s1_sh_stride = 16 * thread_m_blocks`(line 367)+ predicate
`s1_sh_wr_pred = threadIdx.x < prob_m`(line 399)。Kernel reads s1 in
chunks possibly larger than `prob_m`,with predicate guarding the actual
write。**ARLE allocates `s_activation` as f32 array of length m** in
linear.rs。If kernel reads past m index without checking predicate
correctly,reads garbage memory。

Investigation:
- Read kernel lines 396-400 + 537-560 carefully for s1 access pattern
- Compare ARLE `s_activation` allocation:`ctx.stream.alloc_zeros(m)` —
  zero-padded so reads beyond m would return 0,not random garbage
- If alloc is correct,this isn't the bug

### B.2 INT8 sign convention(P1)
PR #31 quant clamps to `[-128, 127]` ASYMMETRIC range。ARLE
`w4a8_activation_quant.cu:40` clamps to `[-128, 127]` ASYMMETRIC ✓。
Kernel's mma `s8.s8` mma instruction handles full INT8 range。

Should be OK,but verify:
- `extern "C" int8_t output[...]` in ARLE — signed
- Kernel `(const int4*) A` cast — read as bytes,sign-aware via mma

### B.3 Storage int32 endian / packing direction(P1)
ARLE pack:`q |= res_np[:, i::8] << (4 * i)` — i ranges 0..7,packs into
uint32 with i=0 as least-significant nibble。Kernel unpacks with
`(q >> (4*i)) & 0xF`(per dequant code)。If kernel uses big-endian or
reverse stride,bug。

Check kernel's unpack:
```bash
grep -nE "q.*&.*0xF|>> [0-9]+.*&.*0xF" /tmp/marlin-w4a8/marlin/w4a8_marlin_cuda_kernel.cu
```

### B.4 Kernel autotune dispatch context(P2)
ARLE passes `thread_k=-1, thread_n=-1` → kernel auto-picks per
`prob_m <= 16 ? (128, 128) : (64, 256)`。For decode m=1 → small batch
config (128, 128)。

PR #31 testbench picks same config for small batch。Should be identical。
But `sms` differs:ARLE passes actual SM count(e.g. 84 on 4070Ti SUPER),
PR #31 may use -1 to auto-detect。Either should yield consistent kernel
behavior。

### B.5 Multi-layer model interaction(P2)
PR #31 reference is single Linear。ARLE runs through multi-layer Qwen3
where each layer's activation feeds next layer's input。If 1st layer
produces wrong activation,next layer's input is garbage → cascading
divergence。

Diagnostic:add a "first divergence layer" probe — instrument forward
pass to dump activation at each layer,compare to BF16 reference at
same layer,find which layer first diverges。

If divergence is at layer 0 → activation quantizer or layer 0 weight
unpacking。If layer N>0 → cascading error,fix layer 0 first。

## Decision tree

```
After re-quant + greedy_consistency:
├── PASS → Scenario A,proceed to bench + default-on
└── FAIL → diagnose path:
    ├── First-divergence at layer 0 → suspect B.1/B.2/B.3
    │   ├── Layer 0 attention?  → check QKV linear W4A8 forward
    │   └── Layer 0 MLP?        → check gate/up/down forward
    ├── First-divergence at layer N>0 → cascading bug,fix earliest
    └── Random divergence → instrument single-layer kernel directly:
        ├── Use `test_w4a8.py` reference test from /tmp/marlin-w4a8/
        ├── Build PR #31 marlin Python module locally
        └── Run on same hardware (sm_89) to baseline expected output
```

## Strategic note(if Scenario B)

5+ iterations on script side already exhausted。If canonical pack still
fails greedy,**escalate to known-good PR #31 reference**:
1. `cd /tmp/marlin-w4a8 && pip install -e .` to build PR #31 Python module
2. Run `python test_w4a8.py` to confirm kernel works on this hardware
3. If PR #31 test PASSES → kernel/hardware OK,bug in ARLE-specific
   integration(loader / linear.rs / multi-layer model)
4. If PR #31 test FAILS → kernel doesn't work on sm_89 with this build,
   revisit kernel compilation flags

This is the audit-Option-3 path(known-good reference checkpoint test)
prescribed in `01ace86`。Currently deferred pending Scenario A test
result。

## Cross-references

- Methodology retrospective: [`3cee2f0`](2026-05-08-w4a8-methodology-retrospective-wrong-class.md)
- Audit clean: [`01ace86`](2026-05-08-w4a8-kernel-and-wiring-audit-clean.md)
- Multi-shape diag: [`4aebcec`](2026-05-08-w4a8-pack-roundtrip-diag-confirms-broken.md)(actually confirmed prod-shape PASS)
- Diag tool: `scripts/diag_w4a8_pack_roundtrip.py`
- Canonical pack: `scripts/quantize_qwen3_w4a8.py`
- Activation quant: `crates/cuda-kernels/csrc/gemm/w4a8_activation_quant.cu`
- PR #31 reference: `/tmp/marlin-w4a8/marlin/__init__.py:160-261`(W4A8Layer)
- W4A8 garbage gate: `81b6481`
- Failing test: `infer/tests/greedy_consistency.rs::test_w4a8_vs_bf16_token_diff`

## Rule

When pack/kernel/wiring have all been audited 0-diff vs upstream,but
greedy still fails,the **next-most-likely bug is integration-level**:
multi-layer cascading,padding alignment,or dispatch context。Do NOT
re-iterate single-component audits — escalate to known-good reference
test(audit Option 3)to definitively rule out kernel/hardware。
