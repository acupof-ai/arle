# DSv4 EAGLE Acceptance Functional Gate PC10

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC9 made the SGLang best-practice startup contract fail closed unless
`ARLE_INTERNAL_MTP_ACCEPT_DRAFTS=1` is enabled for EAGLE/internal-MTP.
That prevented target-only effective output from being counted as an EAGLE
run, but it still needed a small functional gate before using accepted draft
tokens in any later benchmark.

## What Worked

A debug-fallback simple TP8 all-reduce run with explicit MTP loading and
accepted drafts served the raw-completion `137 + 269` gate:

- `ARLE_DSV4_PERFORMANCE_PROFILE=debug-fallback`
- `ARLE_TP_SIZE=8`
- `ARLE_DSV4_MOE_BACKEND=allreduce`
- `ARLE_DSV4_EXPERT_BACKEND=deepgemm`
- `ARLE_DSV4_LOAD_MTP_WEIGHTS=1`
- `ARLE_INTERNAL_MTP_ACCEPT_DRAFTS=1`
- `--spec-enabled --spec-draft-model eagle`

Artifact: `/tmp/dsv4_pc10_eagle_accept_func_mtp_1780455413`.

Validation:

- c=1 output contained `406`.
- c=4 outputs all contained `406`.
- c=8 returned no HTTP errors, 64 output tokens, and all rows contained `406`.
- Gate result: `ANSWER_PASS`.

Spec metrics:

- `infer_spec_draft_tokens_total`: 235
- `infer_spec_verified_tokens_total`: 235
- `infer_spec_accepted_tokens_total`: 30
- `infer_spec_acceptance_rate`: 0.127660
- debug spec rows: 512
- rows with nonzero effective acceptance: 208

Byte parity stayed diagnostic-only and failed, as expected for this prompt:
c=4 rows shared the answer token but diverged in trailing text. This is not a
correctness failure for the current gate.

## Verification

Remote DSv4 pod, `/data01/build/arle`, commit `d24b013c`:

- Server opened `/healthz` with simple TP8 debug-fallback all-reduce.
- `scripts/dsv4_batched_decode_validate.py 18536` returned status 0.
- Post-run process checks showed no lingering infer/timeout compute process
  output.

Two non-passing setup attempts were also useful:

- Advanced target axes under `debug-fallback` correctly failed env setup:
  `DeepSeek V4 advanced multi-axis layout is parsed but not wired into debug execution yet`.
- Simple TP8 without `ARLE_DSV4_LOAD_MTP_WEIGHTS=1` correctly failed startup:
  `ARLE loaded only 0 of 1 mtp.N layer(s)`.

## Rule

Accepted EAGLE drafts are now functionally licensed only for small debug
correctness gates. They are still not a DSv4-Flash TP8 + EAGLE performance
win. The target path still needs full-decode graph replay, token-owned
native DeepEP, and the 256K/1500 hot-cache benchmark metrics together.
