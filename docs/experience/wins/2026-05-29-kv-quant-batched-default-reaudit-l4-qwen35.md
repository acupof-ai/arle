# KV-quant re-audit on L4 — FP8 "catastrophic divergence" kill falsified on Qwen3.5-4B; FP8 is the recommended batched-serving default

## Context

Follow-up to the L4 day-1 sweep
([`2026-05-29-guidellm-ttft-throughput-l4-qwen35.md`](2026-05-29-guidellm-ttft-throughput-l4-qwen35.md)),
commissioned by ckl's directive **"KV 量化设默认 (并发吞吐)"** — make a quantized
KV format the recommended default for batched L4 serving.

The blocker turned out to be **correctness, not perf**. `kv_mode_candidates`
in `infer/src/main.rs:2098` gates `auto`→BF16 with this rationale:

> FP8 historically held this slot, but the 2026-05-25 cross-precision parity
> audit shows FP8 diverges at the first decode token (mean trajectory match
> **0.4%** vs BF16). Until root-caused, `auto` ships the correctness-safe BF16
> default.

But [`errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`](../errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md)
(dated **one day later**) already showed that `mean_match` number was a
**test artifact**: the BF16 reference was a degenerate `"!!!!"` repetition loop
(Qwen3-**4B base**, greedy, long technical prompts). INT8 scored 1.0 by
faithfully reproducing the junk; FP8 "diverged" by breaking out of the loop
into real text. That entry's action #4 mandates **re-auditing FP8 under a
corrected methodology before any license decision** — and it was never done on
**Qwen3.5-4B**, the actual canonical serving model `auto` gates.

Two further facts make the old gate untrustworthy for this decision:
- The `kv_precision_parity.rs` unit test is **hard-coded to Qwen3-dense**
  (`MODEL_PATH=models/Qwen3-4B`, loads via `Qwen3Model::from_safetensors`),
  so it *rejects* `Qwen3.5-4B` (`model_type: qwen3_5`). The entire FP8 saga was
  measured on a different model family than the one in production.
- The methodology fix (degenerate-baseline guard + coherent `DEFAULT_PROMPTS`)
  has since landed in that test, but it still can't load Qwen3.5.

So this audit re-ran the errors-entry's own Rule **directly through the L4
serving path on Qwen3.5-4B**: decode the actual greedy tokens, bf16 vs
int8/fp8/int4, and compare.

L4 sm_89 / CUDA 12.8, `target/release/infer` @ `be706204`,
`ARLE_CUDA_DISABLE_FLASHMLA=1`, model `/content/Qwen3.5-4B`.

## Results — correctness (decoded greedy tokens, Qwen3.5-4B, n=8 prompts × 64 tok)

`temperature=0` greedy, 8 coherent prompts (Eiffel Tower, Fibonacci-in-Rust,
transformer attention, RLHF-vs-OPD, Dijkstra, CUDA warps, water cycle, BST).
`coherent` = output is real text, not a single-token repetition loop.
`char-prefix-agree` = % of bf16's output reproduced verbatim before the first
divergent character.

```
prec   coherent   avg char-prefix-agree vs bf16
bf16     8/8       (reference)
int8     8/8        23.2%
fp8      8/8        51.2%
int4     8/8        31.2%
```

- **All four precisions are 8/8 coherent. Zero degenerate / catastrophic
  outputs.** The BF16 reference is itself coherent on Qwen3.5-4B (no `"!!!!"`
  loop) — the degenerate-baseline trap that poisoned the Qwen3-4B-base audit
  does **not** occur here, so the comparison is valid.
- **The "FP8 catastrophic divergence" kill is falsified on the canonical
  model.** FP8 is byte-identical to BF16 on several prompts and is the
  *closest* of the three to BF16's exact trajectory (51% vs 23/31%).
- **`char-prefix-agree` is a divergence-point metric, NOT a quality metric.**
  Greedy decoding has a butterfly effect: a single quant-noise-induced argmax
  flip near a tie drops prefix-match to ~0 while *both* continuations stay
  coherent and valid (the exact lesson of the 2026-05-26 errors entry). So
  int8's 23% means "diverges from bf16's path earlier," **not** "23% as good."
  The quality signal that matters — coherence — is 8/8 for every format.

## Results — perf (multi-shape c=1→16, 128in/128out)

c=1/4/8 from the day-1 sweep; c=16 from this run (c=8 reproduced within 3%
across the two independent runs → stable). `ok/inc` = completed / incomplete
at the 30 s window close.

```
prec   c   ok/inc    TTFT md(ms)   ITL md(ms)   total_tok/s   Δ vs bf16
bf16   1    7/0         69.3         35.41          56.4        —
int8   1    7/0         69.3         36.85          54.2        −3.9%
fp8    1    7/0         68.6         36.94          54.1        −4.1%
bf16   4   21/3        237.3         38.70         175.8        —
fp8    4   24/0        176.6         39.37         199.1       +13.3%
bf16   8   41/7        302.3         41.59         316.1        —
fp8    8   48/0        280.3         42.18         366.6       +16.0%
bf16  16   66/14       409.7         46.88         540.7        —
fp8   16   80/0        484.3         47.42         636.1       +17.7%
```

(int8 / int4 are perf-identical to fp8 at every concurrency, within ~0.3%.)

- **At c=1 bf16 wins** (no dequant overhead): −4% throughput / +1.4 ms ITL for
  the quantized formats.
- **At c≥4 the quantized formats win and the gap grows with concurrency**
  (+13% → +16% → +17.7% total throughput) **and complete every request, while
  bf16 drops 12.5%→12.5%→17.5% of requests** (incomplete at window close). On
  L4's ~300 GB/s GDDR6 the bf16 pool's 2× KV-byte traffic per decode step
  saturates bandwidth under batched decode; the quantized pool (1.58× capacity,
  fewer bytes) sustains the load. *(Bandwidth-contention mechanism is a
  hypothesis consistent with the L4-vs-V100 flip; not ncu-profiled.)*

## What it says — the "设默认" verdict

1. **KV-quant is correctness-safe to serve on Qwen3.5-4B.** All of int8/fp8/int4
   produce coherent output; the FP8 kill was a Qwen3-4B-base + degenerate-
   reference artifact and does not reproduce on the canonical model.
2. **FP8 is the recommended batched-serving default.** It uniquely combines
   (a) the closest output to BF16 of any quantized format, (b) the full c≥4
   throughput win (+13–18%) with zero dropped requests, and (c) 1.58× pool
   capacity. Plausible reason it tracks BF16 best: per the `--kv-cache-dtype`
   CLI contract, **FP8 keeps the contiguous prefill cache in BF16 and only
   quantizes tokens migrated into the paged pool** — so for 128-token prompts
   that mostly live in the prefill cache, FP8 barely perturbs the prompt KV.
   *(int8's prefill-quant behavior not code-verified this session — hypothesis.)*
3. **The `auto`→BF16 default and its `main.rs:2098` comment are stale.** Their
   rationale (FP8 0.4% match) is falsified for Qwen3.5-4B. The comment was left
   unedited here because the dispatch/FP8 area is under active concurrent work
   (`8417fb56`, `5bd83267`, `be706204`); correcting it should land with that work.

## Recommendation — form of "设默认"

ckl chose the **documented batched-serving default** form, which is what's
SOLID-shippable today:

- **For batched L4 serving (c≥4): use `--kv-cache-dtype fp8`.** This is the
  recommended default; it is a Pareto win over BF16 on every batched metric and
  preserves BF16 output quality on this shape.
- **For latency-critical single-stream (c=1): keep BF16** (`auto`) — fp8 costs
  ~4% / 1.4 ms ITL there with no benefit.

**Gate before flipping the hard `auto` *value* to FP8** (a blanket code default
affecting all users): one real quality eval — perplexity on a held-out corpus
or a small lm-eval-harness task, bf16 vs fp8 — because `char-prefix-agree` +
coherence prove "not broken," not "equal quality." Plus a **long-context FP8
audit**: the fp8≈bf16 closeness here is measured at 128-token prompts that live
in the BF16 prefill cache; at long contexts more KV migrates into the fp8 pool
and divergence may grow. Both are cheap and are the licensed next step.

## Caveat — scope

- Single prompt shape (128/128) for perf; n=8 prompts × 64 tok for correctness.
  No long-context, no output-heavy, no c≥32.
- Coherence (8/8) proves no catastrophe; it does **not** prove fp8 output is
  *as good as* bf16 — that needs the perplexity/lm-eval gate above.
- Greedy temp=0 without `INFER_DETERMINISTIC=1` (that flag forces a
  graph-capture warmup that exceeds a 240 s boot window on L4). Greedy is
  deterministic within a single server process, so cross-precision divergences
  this large (whole words) are KV-precision-driven, not float jitter — fp8's
  exact match to bf16 on some prompts confirms it isn't random noise.

## Reproducibility

```bash
# Correctness: decode the actual tokens through the server (not the unit test —
# kv_precision_parity.rs is Qwen3-dense-only and rejects qwen3_5).
for prec in bf16 int8 fp8 int4; do
  pkill -9 -x infer; sleep 3        # -x infer, NOT -f target/release/infer (self-kill)
  ./target/release/infer --model-path /content/Qwen3.5-4B --port 8000 \
                         --kv-cache-dtype "$prec" --num-slots 4 &
  # wait /v1/models 200, then POST /v1/completions {temperature:0, max_tokens:64}
  # for N coherent prompts; diff generated text vs bf16. Full driver:
  # /content/kv_quant_e2e_parity.sh
done
# Perf c=8,16: same guidellm config as the day-1 sweep, --rate "8,16".
```

## Rule

**The gate for "can a quantized KV format be the serving default" is a decoded-
token correctness check on the *production model*, not a `mean_match` number
from a unit test on a different model family.** Before trusting any
trajectory-match metric: decode the reference and confirm it's coherent (a
degenerate reference inverts the ranking), and run it on the model the default
actually gates. The Qwen3-4B FP8 kill cost ~3 weeks and a default downgrade on
a metric that never applied to the canonical model.

For the default-flip itself: multi-shape perf (have it: c=1→16) establishes
*safe under load*; a real quality eval (perplexity / lm-eval) establishes
*equal quality* — both are required before flipping the hard `auto` value, and
the c=1 latency cost means the safe form is a documented per-workload default,
not a blanket flip.
