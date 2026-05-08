# M_quant Round 4 #6 — W4A16BatchGemv override LICENSE bench(pending-remote)

> Status:**pending-remote**(per CLAUDE.md MANDATORY-bench-per-change rule)。
> Implementation `02209f4` LANDED;bench requires GPU pickup to enable env
> var + run Round 1 protocol。

## Goal

Override `LinearKernelPlan::batched()` dispatch for batch>1 from MarlinW4Gemm
(3 kernel launches:bf16→fp16 + GEMM + fp16→bf16)to W4A16BatchGemv
(BF16-native,1 launch)。Hypothesis:eliminate 2× format-conversion launches
to reduce ITL。

## Hypothesis

Per `docs/experience/errors/2026-05-08-marlin-w4a16-bench-implementation-gap.md`:
- BF16-native W4A16BatchGemv = 1 launch
- MarlinW4Gemm = 3 launches(bf16→fp16 + GEMM + fp16→bf16)
- Predicted ITL 14.1-12.1 ms(1.37×-1.59× vs BF16 baseline 19.27 ms)
- Straddles license band 1.5×

## Implementation(LANDED)

`02209f4 feat(linear): R4#6 env-gated W4A16BatchGemv override at MarlinW4Gemm dispatch`:
- 12 LOC at `infer/src/ops/linear.rs:71-83`
- Env-gated:`INFER_R4_W4A16_GEMV_OVERRIDE=1` enables override
- Default OFF preserves production Marlin path
- Type-safe(both arms dispatch on `WeightFormat::W4A16`)

Audit chain:
- `6ade2d4` Claude Phase 0 audit(claims verification + bench protocol)
- `5bb99d7` Claude audit-of-audit(6/6 claims SOLID + 3-line pattern)
- `02209f4` Implementation LANDED(this entry's substrate)

## Command(pending GPU pickup)

Server:
```bash
INFER_R4_W4A16_GEMV_OVERRIDE=1 \
CUDA_HOME=/opt/cuda \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/infer \
  --model-path infer/models/Qwen3-4B-GPTQ-Int4-marlin \
  --port 8000 \
  --num-slots 8 \
  --max-seq-len 5120
```

Bench:
```bash
scripts/bench_guidellm.sh r4-6-w4a16-gemv-override \
  --concurrencies 4 --max-seconds 60 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

## Environment

- **Backend**:CUDA
- **Model**:Qwen3-4B-GPTQ-Int4-marlin(W4A16 Marlin format)
- **Hardware**:RTX 4070 Ti SUPER 16GiB,CUDA 13.2
- **Commit**:`02209f4`
- **Compare against**:Round 1 protocol baseline ITL p50 18.13 ms

## Results

**Pending GPU pickup**(this is a stub)。Numbers will land here when bench runs。

| metric | baseline(MarlinW4Gemm,Round 1)| override(W4A16BatchGemv) | Δ% |
|--------|-------------------------------|--------------------------|-----|
| ITL p50 | 18.13 ms | tbd | tbd |
| ITL p99 | tbd | tbd | tbd |
| TTFT p50 | tbd | tbd | tbd |
| out tok/s | tbd | tbd | tbd |

## License-or-kill criteria

Per `02209f4` commit body + `6ade2d4` Phase 0 audit:
- **License**:ITL p50 ≤ 12.85 ms(1.5× improvement vs BF16 baseline 19.27ms)→ ship default ON,update env var documentation
- **Kill**:ITL p50 ≥ 18.13 ms(no improvement vs Round 1 MarlinW4Gemm baseline)→ revert env var,eliminate W4A16BatchGemv hypothesis,update implementation-gap doc with KILL evidence
- **Mixed**(12.86-18.12 ms):accept as opt-in optimization,don't default ON

## §0 SOLID gates

- ✅ Single-variable A/B(only env var changes)
- ⏳ N≥3 paired(per `3c334ef` LICENSE protocol)— pending bench
- ⏳ σ/mean < 5%— pending bench
- ⏳ greedy_consistency verified in BOTH modes— pending bench
- ✅ Layer-8 num_slots gate(`--num-slots 8` constant per `bbedbc9`)

## Tradeoffs(8 axes per skill rule 7)

Per `6ade2d4` Phase 0 audit:
1. Memory waste(Marlin pack stays loaded even with override ON)— deferred follow-up cleanup
2. Numerical correctness(W4A16BatchGemv vs MarlinW4Gemm output equivalence)— gated by greedy_consistency in bench
3. Batch-range specificity(prediction at batch=4,verify at multiple sizes)
4. Backend isolation(CUDA-only;no Metal impact)
5. Backwards compat(env var default OFF preserves production)
6. Default flip risk(only if license + multiple confirmation runs)
7. Kernel availability(W4A16BatchGemv arm pre-existing,verified in 5bb99d7)
8. Production stability(env-gated rollout reduces blast radius)

## Cross-references

- `02209f4` implementation
- `6ade2d4` Phase 0 audit
- `5bb99d7` audit-of-audit verified
- `2026-05-08-marlin-w4a16-bench-implementation-gap.md` hypothesis source
- Round 1 baseline reference data
- CLAUDE.md MANDATORY-bench-per-change rule
- `bench_guidellm.sh` canonical bench tool

## Status

**pending-remote**:bench runs at codex GPU pickup or remote-machine pickup。
Once numbers land,this entry transitions to LICENSED / KILLED / partial verdict
with full Δ% table。

Per CLAUDE.md "If the bench can't run locally, the commit body MUST cite the
remote-machine ticket or plan entry that will execute it, and the entry is
opened as a stub under wins/ with status pending-remote. No silent skips."

This entry fulfills the pending-remote stub requirement。Next-tick pickup
runs the bench command above and updates Results + License verdict sections
in-place。
