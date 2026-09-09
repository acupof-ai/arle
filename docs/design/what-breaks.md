# What breaks

Ten entries from the [experience corpus](experience/), selected for root
causes that generalize beyond this codebase. One paragraph each: Symptom,
Root cause, Rule. Dates are the entries' own; the linked entry carries the full
evidence. Design note 5 ([matched-measurement.md](matched-measurement.md))
states the method these ten broke.

## Full-prefix match feeds the wrong seed token

[wins 2026-07-08](experience/wins/2026-07-08-prefix-cache-wrong-seed-token-fix.md)
— **Symptom:** repeating the same prompt against a warm server corrupted the
output from the second call onward (`738291` → `738292`), 100% reproducible at
concurrency 1. **Root cause:** a full-prompt prefix match entered decode with
no forward pass and no generated token; the decode-row builder fell back to
the prompt's last token as the decode seed, duplicating it into KV and shifting
every later position by one. **Rule:** a cache or restore match must run one
genuine forward-and-sample step before decode; an empty-generation decode
state is a bug signal, never a valid state to cover with a fallback.

## Spec decode serializes above c=1

[errors 2026-07-26](experience/errors/2026-07-26-dspark-spec-decode-serializes-and-loses-above-c1.md)
— **Symptom:** DSpark spec decode ran 2.24× faster at c=1 and 39% slower at
c=16; latency scaled 1×/8×/16× with concurrency. **Root cause:** draft
generation ran per request, never batched across slots, so concurrency added
latency without adding throughput; the no-spec control scaled 2.9× over the
same range, isolating the serialization to the spec path. **Rule:** before
tuning a knob inside a feature, measure the feature against not using it at
the target operating point; a win at one concurrency says nothing about
another.

## A decode-graph flag logs ARMED and captures nothing

[errors 2026-08-01](experience/errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md)
— **Symptom:** `--qwen35-decode-graph` printed an ARMED log line, but a
four-arm serve A/B showed only noise and nsys counted zero `cuGraph*` calls
with the flag on. **Root cause:** the graph lane's single call site sat below
an unconditional early return for paged KV, the serving default; the ARMED
line came from the warmup path, which the dispatch path never reached. **Rule:**
a flag's log line is not evidence the feature ran; confirm engagement by
counting the API calls the treatment should produce.

## A zero-grid availability probe mistaken for a long-prompt crash

[errors 2026-08-13](experience/errors/2026-08-13-gdn-zero-token-chunk-kills-engine.md)
— **Symptom:** a 64k-token prompt reportedly killed the engine; an
`LD_PRELOAD` trace on `cuLaunchKernel` found a launch with grid=(0,1,1).
**Root cause:** the zero-grid launch was a deliberate one-shot capability
probe (a cumsum wrapper called with `seq_len = 0`), length-independent and
benign; the length-dependent engine death was never reproduced on a clean
build — the original boundary came from a stale build tree carrying DSpark
flags. The death's cause is unknown. **Rule:** a launch failure found by
tracing is not the failure under investigation; ask whether it correlates
with the input that crashes, and re-verify the premise on a clean binary
before building a chain on it.

## A repack that keeps its source stores the model twice

[wins 2026-08-20](experience/wins/2026-08-20-marlin-source-freed-18gb.md)
— **Symptom:** the smaller checkpoint had half the KV capacity (281,577 vs
593,995 tokens) and the 32K chain ran ~7× the wall clock, with 24 full
recomputes per run. **Root cause:** the Marlin repack kept its 18.7 GB
pre-repack source so one GEMM could stay 12–21% faster; the model was
resident twice, invisible to weights-only accounting, and the KV pool paid
for it. **Rule:** free the source after a repack, and make every dispatch
lane serve the layout that replaced it — the second lane here was found by a
crash, not by reading.

## Parity verified against the wrong oracle

[errors 2026-08-22](experience/errors/2026-08-22-marlin-fp4-parity-wrong-oracle.md)
— **Symptom:** a layout-parity gate passed twice on GPU, while an end-to-end
A/B showed a 0.3363 vs 0.6414 loss gap between the shared and private arms.
**Root cause:** the test's scales were powers of two, chosen so nothing was
lost; the repack's only lossy step (flushing lifted values below 2.0 to zero)
never fired, so the oracle was co-selected with the input that makes it hold.
**Rule:** when a path has a lossy step, a test whose inputs avoid that step
tests a regime the code never runs in; pick the oracle from what the feature
promises.

## Batched spec decode over quantized KV loses with parity intact

[errors 2026-08-22](experience/errors/2026-08-22-batched-dspark-quant-kv-verify-loses.md)
— **Symptom:** batched DSpark over FP8 KV lost 10% at c=8 and 16% at c=16,
while the needle ladder passed 12/12 exact and deterministic under a
concurrent 32K stream. **Root cause:** cause unknown. The verify-kernel
hypothesis was tested and refuted — an MMA verify kernel sharing one
attention path with plain rows left the loss unchanged — and smaller prefill
chunks were refuted as a fix. **Rule:** a correctness gate and a throughput
gate answer different questions; parity intact does not license a throughput
loss, and a rejected lever keeps its gate with the numbers in the comment.

## One NVFP4 checkpoint corrupts tool calls

[errors 2026-08-23](experience/errors/2026-08-23-nvfp4-tool-calls-corrupt.md)
— **Symptom:** ThinkingCap-Qwen3.6-27B-NVFP4 emitted token soup on tool-call
prompts where the FP8 build was correct, deterministically across runs; 68
agent-OPD rollouts returned `edited=false`. **Root cause:** unknown. The
static weight chain verified bit-exact (zero flushes on all 263 NVFP4
tensors), the Marlin kernel verified against CPU ground truth at all model
shapes, and a sibling NVFP4 checkpoint passes every probe; the suspect
narrows to a forward-path interaction with this checkpoint's weight
distribution. **Rule:** two quantizations of one checkpoint are two models
until a matched probe says otherwise; run the cheap matched probe before
reading any downstream number as a property of the workload.

## A slot budget without the prefill transient

[wins 2026-08-24](experience/wins/2026-08-24-dsv4-budget-prefill-reserve.md)
— **Symptom:** c=8 died mid-serve with all-rank CUDA OOM; the slot solve had
handed every budget byte to slot state and the FlashMLA pool. **Root cause:**
the first long prefill's chunk transients — 1352 MB itemized at TP=4 —
allocate outside the budget; the plan was valid only for an idle engine.
**Rule:** a budget planner that admits N units first reserves the transient
working set of the operation that fills them; reserve terms are itemized
from allocation sites, never a factor.

## Restored pages minted new logical ids

[wins 2026-09-02](experience/wins/2026-09-02-metal-prefix-restore-survives-turns.md)
— **Symptom:** turn 2 of a 12-turn conversation restored its prefix; turns 3
through 12 licensed 0 blocks and re-prefilled the whole prompt. **Root
cause:** republish minted a new logical id for every restored page, so
earlier boundary snapshots read as recycled and were pruned; the snapshots
that survived were keyed to a page chain the radix never hands out, because
the Metal implementation ignored the seam's `slot_pages` repair argument
(`_slot_pages`). **Rule:** when a contract passes a repair argument, grep
every implementation for the underscore-prefixed name; an ignored parameter
is a contract half implemented.
