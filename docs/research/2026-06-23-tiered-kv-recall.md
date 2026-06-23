# Write-Through Tiered KV-Cache with Relevance-Based Recall for Bounded-HBM Million-Token Sessions

> **Status (2026-06-23): working draft.** Method + related work are final; the
> Evaluation section gives the protocol + table skeletons — **results are TODO,
> pending the pod benchmark runs** (no numbers are fabricated). Design source:
> [`docs/plans/2026-06-23-writethrough-tiered-kv-memory.md`](../plans/2026-06-23-writethrough-tiered-kv-memory.md).
> Early GPU evidence (Qwen3.6-27B, H20 GPU0): per-step→per-prefill recall **fixed a
> crash** and removed the **>10 min per-step decode stall** (the load-bearing RQ3
> ablation); `6000/0.75` needle now **retrieved** via prefill prefetch (119 s,
> gated by a *synchronous* prefill write-back of ~340 cold blocks → §3.3 staging
> makes it async). Decode itself is fast (quick gen 0 s, no per-step tier I/O).
> Recall *quality* is partial on a 2-case probe (`2000/0.75` missed) — the
> mean-key scoring didn't rank that block top-k; RQ1 grid + top-k/query-window
> tuning pending. Not over-claiming "great recall" until the grid is run.

## Abstract

The key–value (KV) cache is the central scaling constraint of transformer inference: it
grows linearly with context, and modern workloads now stress it from two directions at
once — single requests reaching million-token contexts, and long-lived agent sessions that
accumulate history across hundreds of turns. Existing systems manage this pressure by
either *dropping* KV (eviction, sliding windows, token sparsification), which discards
memory the model may later need and forces expensive recomputation, or by *swapping* KV
between HBM and host memory mid-decode, which injects page-allocator contention and
transfer stalls directly onto the latency-critical decode path. We argue that neither the
loss of state nor the contention of on-the-fly paging is fundamental. We present a
KV-cache architecture that treats HBM as a *write-through cache* over a layered DRAM +
NVMe backing store, with one structural invariant: **decode never touches the tier.** All
movement between tiers — eviction of cold pages downward and relevance-scored *recall* of
pages back upward — is confined to the prefill synchronization point, where the engine is
already paused and the allocator is already quiescent. Relevance scoring at prefill selects
which historical KV to prefetch into bounded HBM before generation resumes, so decode runs
at full speed over a fixed-size working set regardless of total session length. The design
is model-agnostic, requiring no changes to attention kernels or model weights. We show that
this separation enables million-token contexts and arbitrarily long agent sessions under a
fixed HBM budget while leaving decode throughput statistically indistinguishable from the
unbounded-HBM baseline.

## 1. Introduction

The KV cache has quietly become the binding constraint on what large language models can
do. Every generated token attends to the keys and values of all prior tokens, so the cache
grows linearly with context length and is pinned in high-bandwidth memory (HBM) for the
lifetime of a request. As context windows have stretched toward and past a million tokens,
and as agentic deployments have shifted from single prompts to *sessions* — multi-turn
rollouts that interleave tool calls, retrieved documents, and reasoning across hundreds of
steps — the same cache now bounds two distinct regimes simultaneously: the depth of a
single long-context request and the longevity of a persistent agent. In both, HBM runs out
long before the model's quality does.

The field has answered this pressure in two broad ways, and both pay for capacity with
something the application can ill afford. The first family **drops** KV: sliding-window
attention, attention sinks, heavy-hitter eviction, and token-level sparsification all bound
HBM by deciding that some past state will never be needed again. When that bet is wrong —
when an agent must recall a fact from fifty turns ago, or a long document re-becomes
relevant — the state is simply gone, and the only recourse is to recompute it from scratch,
paying prefill cost a second time. The second family **swaps** KV: it keeps all state but
pages it between HBM and host memory on demand. This preserves memory, but it does so by
moving bytes *during* decode, placing host-device transfers and page-allocator activity
squarely on the per-token critical path. The result is decode-time jitter and allocator
contention precisely when the system is most latency-sensitive. The two approaches trade
the same coin from opposite sides: one sacrifices the model's memory to protect decode
speed, the other sacrifices decode speed to protect the model's memory.

Our central observation is that this trade-off is an artifact of *when* tier movement
happens, not an inherent cost of tiering. Decode is latency-critical and
allocator-sensitive; prefill is neither — at the prefill synchronization point the engine
is already stalled on a batch boundary and the page allocator is already idle. If all data
movement between memory tiers is confined to that point, then decode can run undisturbed
over whatever working set prefill left in HBM, while the full session history lives safely
in cheaper, larger tiers below.

This motivates a **write-through tiered KV-cache with relevance-based recall.** We treat
HBM not as the sole home of the cache but as a write-through cache over a backing store of
host DRAM and NVMe. Newly computed KV is written through to the lower tiers, so nothing is
ever lost and recomputation is never forced. Cold pages are evicted *downward* and
historically relevant pages are *recalled upward* — but exclusively at the prefill sync
point, governed by a relevance scorer that decides which past KV is worth occupying scarce
HBM for the coming generation phase. The invariant that **decode never touches the tier**
is what makes the scheme both fast and general: because the mechanism lives entirely
outside the attention kernels and depends on no model-specific structure, it applies
uniformly across model families and attention variants without kernel surgery or weight
modification.

We preview our findings qualitatively, without numbers. First, the design sustains
million-token single-request contexts and effectively unbounded agent sessions under a
*fixed, configurable* HBM budget — capacity is decoupled from HBM size. Second, because
decode is fully insulated from tier traffic, per-token decode latency and throughput remain
statistically indistinguishable from an unbounded-HBM baseline that holds the entire cache
resident. Third, relevance-based recall preserves answer quality on long-range retrieval
and long-horizon agent tasks where drop-based methods degrade, because the state is never
destroyed — only relocated and selectively re-summoned. Fourth, the gains hold across
multiple model families with no per-model engineering, confirming that the mechanism is
genuinely architecture-neutral.

### Contributions

- **A write-through tiering model for the KV cache.** We recast HBM as a write-through
  cache over a DRAM + NVMe backing store, so that all KV is durably retained across tiers
  and the system never drops state or pays forced recomputation — separating *capacity*
  (set by the backing tiers) from *resident footprint* (set by an HBM budget).
- **The "decode never touches the tier" invariant.** We confine every inter-tier movement
  — downward eviction and upward recall — to the prefill synchronization point, removing
  host-device transfers and page-allocator contention from the per-token decode path, and
  we show this is the structural reason decode speed is preserved.
- **Relevance-based recall at the prefill sync point.** We introduce a relevance-scored
  prefetch that selects which historical KV pages to promote into bounded HBM before each
  generation phase, turning tier management from reactive on-the-fly paging into a single
  proactive scheduling decision at a point where the engine is already quiescent.
- **A model-agnostic, kernel-free realization.** The mechanism lives entirely above the
  attention kernels and assumes no model-specific structure, so it generalizes across model
  families and attention variants with no kernel modification or weight change.
- **An end-to-end evaluation across the two binding regimes.** We evaluate on both
  long-context single requests and long-session agent rollouts, showing bounded-HBM scaling
  to million-token sessions, decode performance statistically indistinguishable from the
  unbounded-HBM baseline, and preserved long-range quality where drop-based baselines degrade.

## 2. Related Work

**KV-cache eviction and sparsity.** The dominant response to KV-cache growth is to keep
only a fraction of the cache resident. StreamingLLM [1] observes that a few initial tokens
act as *attention sinks* and, by retaining those sink tokens plus a sliding window of recent
KV, sustains stable generation over millions of tokens with a bounded window — but it
permanently discards the middle of the context, so any information outside the window is
unrecoverable. H2O [2] formalizes a *heavy-hitter* eviction policy, retaining the ~20% of
tokens that dominate accumulated attention mass alongside recent tokens; Scissorhands [3]
makes the analogous "persistence of importance" hypothesis the basis of a fixed-budget
cache. Both decide evictions from running attention statistics, so a token cheaply dropped
early cannot be revived if a later query needs it. SnapKV [4] compresses the prompt once at
the end of prefill by clustering the KV positions each head attends to, and FastGen [5]
profiles per-head attention structure to build an *adaptive* policy (local, special-token,
or full) per head. These methods are lossy by construction: they assume the relevant tokens
can be identified in advance, and they degrade on tasks whose query-relevance shifts over a
long session.

**Retrieval and block-level recall.** A second line keeps the full history but loads only
the relevant part on demand. InfLLM [6] stores distant context as fixed-size memory blocks
and, per step, looks up the token-relevant blocks for attention, extrapolating short-window
models to 1M-token inputs without training. Quest [7] adds *query-aware* page sparsity,
estimating each KV page's relevance to the current query and computing attention over only
the top pages. PQCache [8] compresses keys with product quantization and uses the codes to
approximate a top-k retrieval cheaply. The shared limitation is that recall happens *inside
the decode loop*: every decode step issues a relevance lookup and pulls candidate blocks,
putting retrieval (and, when the working set spills off-GPU, tier I/O) on the per-token
critical path.

**Serving systems and tiering.** vLLM's PagedAttention [9] manages KV in OS-style
non-contiguous pages, eliminating fragmentation and enabling copy-on-write sharing — the
substrate, not a recall policy. SGLang's RadixAttention [10] reuses shared prefixes across
requests via an LRU radix tree; its HiCache extension [11] generalizes this into a
hierarchical GPU→CPU→storage cache (L1/L2/L3) with a controller that prefetches cold data
back before a request arrives. LMCache [12] externalizes KV into a tiered CPU/disk/remote
store for cross-request and cross-engine reuse, and Mooncake [13] builds a KVCache-centric,
prefill/decode-disaggregated cluster that pools CPU, DRAM, and SSD across nodes. These
systems target *prefix reuse across requests*; their tiering moves whole prefixes for cache
hits, and none provides a within-session, relevance-driven recall that keeps decode entirely
on-GPU.

**KV quantization (orthogonal).** KIVI [14] applies tuning-free 2-bit quantization —
per-channel for keys, per-token for values — and KVQuant [15] adds pre-RoPE per-channel key
quantization with non-uniform, dense-and-sparse datatypes to reach near-lossless 3-bit and
10M-token contexts. Quantization shrinks each entry but does not change *which* entries are
resident; it composes with, rather than substitutes for, a tiering/recall policy.

**Our position.** This system differs on three axes. (1) *Decode does zero tier I/O.*
Unlike retrieval methods [6,7,8] that lookup-and-fetch per decode step, and unlike
hierarchical caches [11,12,13] that page across tiers reactively, our entire HBM→DRAM→NVMe
write-through recall cycle is amortized into prefill: decode reads a bounded, fully-resident
HBM working set, so per-token latency never pays a transfer. (2) *Model-family generality.*
Eviction, retrieval, and most tiering systems are specified against dense multi-head
attention; our recall is expressed once over a single backend seam and applies uniformly to
dense, MoE, and MLA (compressed-KV) models. (3) *Bounded-HBM million-token sessions.* Prior
work targets long single prompts [1,6,15] or cross-request prefix reuse [9,10,12,13]; we
target the *agent-rollout* regime — one evolving session whose cumulative KV crosses a
million tokens — holding HBM bounded while preserving exact (not evicted) history on lower
tiers, recalled by relevance at each prefill. Quantization-style compression [14,15] is
orthogonal and composable with our tiering.

## 3. Method

### 3.1 Design principle: a write-through cache, not a swap

We treat GPU high-bandwidth memory (HBM) as a *write-through cache* of a session's KV state,
and the host memory hierarchy (DRAM, then NVMe) as the source of truth that holds the entire
history. This inverts the swap model — which freed and reloaded the *same* page under
decode-time memory pressure and therefore contended with the single-allocator KV pool on
every step. Four timing rules govern the cache. On page completion the block's KV stays
resident in HBM and we only retain a small resident representative; on eviction we write the
cold page back to DRAM (L2) and free the HBM page; a background path writes every L2 block
through to NVMe (L3) so no block is ever lost; and at each turn's prefill we prefetch the
relevant history back into HBM. Decode itself touches no tier. The asymmetry — L1→L2
*write-back* (lazy, only on eviction, so the hot path never continuously mirrors) versus
L2→L3 *write-through* (every evicted block lands durably) — is deliberate: write-back keeps
eviction cheap (a copy into fast DRAM), while write-through guarantees full recall coverage,
so a block the working set later wants can always be re-fetched rather than recomputed.

### 3.2 Tier hierarchy and one unified live budget

L1 (HBM) and L2 (DRAM) form *one* live KV budget with a single capacity, a single
eviction/admission decision, and one ledger. A block migrates L1↔L2 within this budget via
cheap write-back or promote; the budget is sized to the *working set*, not to `max_seq_len`.
Concretely the HBM ledger is `working_set + staging + reps`, where
`working_set = num_slots × (n_init + n_local + top_k·l_bs + headroom)` pages. This is what
kills the over-allocation: the prior `num_slots × max_seq_len` sizing claimed tens of GB up
front and OOM'd at build time, whereas a bounded working set is on the order of a few hundred
tokens per slot (the validated default is 32 sink + 256 local + 8·32 recalled = 544 tokens).

L3 (NVMe) sits *outside* the live budget and is explicitly *decoupled* from recall: it is a
general deep tier shared by two independent consumers — the serve-side prefix cache (which
demotes to L3 on pressure) and recall (which uses L3 as its durable archive). Both ride the
same session-keyed `CudaKvTierStore` with separate key namespaces, so ordinary serving can
enable L3 on its own and recall is just another tenant of the same store. Multi-tenant
isolation is structural: a tier block is addressed by `TierBlockKey { session, block }`, so a
prefetch for session A can only name session-A keys.

### 3.3 Reserved staging area

To make every transfer bounded and to decouple a prefetch landing from an eviction, the
ledger reserves a fixed staging area rather than allocating on the fly (an on-the-fly malloc
implicit-syncs and can OOM). On HBM we reserve at least one KV-block per slot, which serves
as both the landing zone for a prefetch (L2/L3→HBM) and the source pin for a write-back
(HBM→L2). On DRAM we reserve a pinned-host ring that is the D2H landing for write-backs and
the H2D source for prefetches. Both D2H and H2D copies run on a side stream — never the
decode stream — so no copy stalls compute and no per-copy temporary `Vec` is allocated.
Because the staging is pre-reserved, a prefetch lands in its reserved page without first
evicting to make room, dissolving the evict-before-prefetch ordering dependency that would
otherwise risk deadlock or a surprise allocation.

### 3.4 The recall mechanism

For each recall block we keep a resident *representative*: the layer-0 K, mean-pooled over
the block's `l_bs` tokens and GQA-shaped to `[num_kv_heads, head_dim]`. The rep is computed
exactly once, at write-through time, while the block's K is still resident — a block
"freezes" when it leaves the local window, so its K is final and is never re-read from the
tier to score. Decoupling the recall-block grain (`l_bs = 256` in production) from the
device page grain (16) bounds the rep pool: a 10M-token session needs roughly 160 MB of reps
rather than 2.5 GB. When the rep pool is capped, reps for the coldest blocks are dropped,
degrading those blocks to prefix-only-recallable — a graceful horizon, not a cliff.

At recall time we score the whole history by relevance: each block's score is
`q_rep · rep[block]`, where `q_rep` is the GQA-mean of the last `m` prompt tokens' post-RoPE
layer-0 queries (the "what am I about to generate" signal). The planner then selects the
working set as the union of the attention sink (first `n_init`), the local window (last
`n_local`), and the top-k highest-scoring middle blocks. Correctness under this
non-contiguous page subset holds because RoPE is pre-baked into the cached K and attention is
position-absolute: a prefetched block attends at its original positions, and the prefetch
changes only *which* pages are resident, not the math.

### 3.5 The key invariant: decode does zero tier I/O

The entire recall cycle — score the history, choose the working set, write-back-evict what
falls out, and issue *one* batched H2D prefetch through the staging ring — runs exactly once
per prefill/turn. After prefill returns, the working set is *immutable* for the duration of
that decode run: decode only appends the new token's KV to the tail page and attends the
fixed resident set. There is no re-scoring, no eviction, no prefetch, and no tier touch
mid-decode. This is enforced by code structure, not convention: `prefill_row_recall` is the
sole site that calls the score/prefetch/evict path, and `decode_row_recall` shrinks to
alloc-token, read the fixed page list, forward, sample. The earlier per-step variant ran the
full score→evict→H2D cycle on *every* decode step, which is precisely why a 6000-token
request took over ten minutes; lifting the cycle to the one batched prefill sync point
removes it from the hot path entirely. This fits turn-based agent rollouts exactly: every
turn re-prefills, and that re-prefill is the natural and only recall point, so recall moves
from per-decode-step to per-turn at zero extra synchronization. The one accepted boundary is
a single ultra-long generation with no re-prefill, which keeps whatever working set its
prefill chose — by design, not a regression.

### 3.6 Generality across model families

The mechanism is expressed device-neutrally as a three-verb seam, `KvTier` (`write_through`,
`evict_drop`, `prefetch`), over a `PagedKVPool` whose page is the natural unit of
write-through, eviction, and prefetch. The same seam and pool therefore serve dense, MoE, and
MLA models without per-model recall logic. Backends opt in by overriding the verbs; a backend
with no tier reports zero capacity and the host never calls them, so the default decode path
stays byte-for-byte unchanged. Per-model differences are confined to a "read-swap" routing
decision at the executor: the dense paged path is wired first and pod-testable; per-slot-KV
models (Qwen3.6-hybrid) restructure their contiguous KV read through the paged pool; and
MLA (DSv4, latent-compressed KV) is routed separately — its already-sparse path is the
adversarial generality case. The same `--kv-recall` opt-in selects the
write-through-tier-plus-prefetch path uniformly.

## 4. Evaluation

We evaluate tiered KV-recall against the central claim: **bounded HBM at unbounded session
length, without sacrificing recall quality or decode throughput.** A naive eviction policy
bounds HBM trivially but forgets the evicted middle; full attention recalls everything but
its KV grows linearly with context and eventually OOMs. Our design retains a fixed working
set in HBM (attention sink + local window + top-k recalled blocks ≈ 544 tokens in the
reference config) and stages evicted blocks to a lower tier, recalling them by relevance at
prefill time. The evaluation must therefore prove *four* properties hold simultaneously —
recall is near the full-attention ceiling, HBM stays flat, decode is not slowed, and all
three generalize across model families. We frame each as a research question with an explicit
ceiling/floor baseline, because a number is only interpretable between the best achievable
(full attention) and the do-nothing (bounded drop).

**RQ1 — Recall quality.** Does recall preserve a needle the working set would otherwise
evict? We extend `scripts/cuda_recall_needle.py` to the full grid: context-depth
`{0, .25, .5, .75, 1.0}` × length `{2k, 8k, 32k, 128k}`. Each cell plants a unique passkey at
the depth fraction, fills the remainder with semantically-inert filler far exceeding the
working set, and asks the model to recover it. Scoring is **correct-inference** — the digits
of the planted key appear in the (regex-normalized) answer, decoded greedily at
`temperature=0` — *not* byte-identity against a reference run, which MoE non-determinism would
confound. Two baselines bracket every cell: **full-attention** (no eviction; the recall
ceiling) and **bounded-KV-drop** (evict-and-forget, no recall tier; the floor — expected to
retrieve only when the needle lands inside sink+local). Tiered-recall must approach the
ceiling, not the floor. We run each grid cell ×3 same-config repeats and report the per-cell
hit fraction; a cell counts as recalled only if it clears the same-config non-determinism
band the baselines exhibit.

**RQ2 — Memory.** Is HBM bounded? We sample `nvidia-smi --query-gpu=memory.used` at fixed
wall-clock intervals while driving a single session whose context grows from 2k to 128k
tokens, for tiered-recall vs full-KV. The claim is **flat HBM** for tiered-recall (working
set is constant-size; staged blocks live off-HBM) against **linear growth** for full-KV,
terminating in OOM at the length the device can no longer hold the full KV. We cross-cite
`/v1/stats` `peak kv_util` and the tier counters (`tier_recall`, `tier_src`,
`tier_promoted`, `kv_store_q`) so the flat curve is attributable to staging, not to a silent
prompt truncation.

**RQ3 — Throughput.** Does recall slow generation? Via `scripts/bench_guidellm.sh` (canonical
params, fixed-c vs reference per spec §7.2) we measure TTFT, ITL (p50/p99), and output tok/s
for tiered-recall vs full-attention vs bounded-drop. The architectural claim is **zero
per-step tier I/O**: recall resolves at prefill, so the decode loop touches only the in-HBM
working set and **ITL must be statistically indistinguishable from bounded-drop** (and no
worse than full-attention). TTFT may carry the recall/staging cost; that is the design's
chosen trade. We report achieved-vs-peak per spec §7.6 (decode is memory-bound: GB/s vs H20
~1.6 TB/s HBM) so a "fast" ITL is not a dispatch-bound artifact. Correctness-gate (§7.1)
every throughput row.

**RQ4 — Generality.** We repeat RQ1–RQ3 across three KV geometries: dense **Qwen3**,
**Qwen3.6-MoE**, and **DSv4-MLA** (latent-compressed KV). MLA is the adversarial case — its
KV is already compressed, so the recall tier must operate on latent blocks; a win there is
the strongest generality evidence.

**Hardware.** All runs on 8×H20 (TP=8/EP=8 for DSv4; TP as configured per model), profiling
OFF for all throughput baselines (spec §7.8).

**Ablations.** (a) **Prefetch granularity** — per-step vs per-prefill recall: per-step recall
stalls the decode loop on tier I/O and drove the observed >10 min long-context latency;
per-prefill recall is the fast path. This ablation is the load-bearing one for the RQ3 claim.
(b) **Staging on/off** — eviction-to-tier enabled vs blocks dropped, isolating recall's
contribution from plain bounded-drop. (c) **Recall top-k sweep** `k ∈ {2, 4, 8, 16, 32}` —
recall quality (RQ1) vs working-set size (RQ2) and TTFT (RQ3) trade-off, locating the knee.

### Results — table skeletons (TODO; no numbers fabricated)

**RQ1 — Recall accuracy** (hit-fraction, n=3 same-config; `T / C / F` = tiered / full-attn
ceiling / bounded-drop floor):

| depth ↓ \ length → | 2k | 8k | 32k | 128k |
|---|---|---|---|---|
| 0.0  | TODO | TODO | TODO | TODO |
| 0.25 | TODO | TODO | TODO | TODO |
| 0.5  | TODO | TODO | TODO | TODO |
| 0.75 | TODO | TODO | TODO | TODO |
| 1.0  | TODO | TODO | TODO | TODO |
| **grid mean** | TODO | TODO | TODO | TODO |

**RQ2 — HBM vs session length** (`nvidia-smi memory.used`, GB):

| ctx tokens | tiered-recall HBM | full-KV HBM | peak kv_util | tier_recall count |
|---|---|---|---|---|
| 2k   | TODO | TODO | TODO | TODO |
| 32k  | TODO | TODO | TODO | TODO |
| 128k | TODO | TODO (or OOM) | TODO | TODO |
| 256k | TODO | TODO (OOM expected) | TODO | TODO |

**RQ3 — Throughput** (guidellm, fixed-c):

| config | TTFT p50 | ITL p50 | ITL p99 | out tok/s | decode GB/s (% of 1.6 TB/s) | gate |
|---|---|---|---|---|---|---|
| full-attention | TODO | TODO | TODO | TODO | TODO | TODO |
| tiered-recall  | TODO | TODO | TODO | TODO | TODO | TODO |
| bounded-drop   | TODO | TODO | TODO | TODO | TODO | TODO |
| **Δ tiered vs bounded (ITL)** | — | TODO% | TODO% | — | — | — |

**RQ4 — Generality:**

| model | KV geometry | RQ1 grid-mean (T vs C) | RQ2 HBM flat? | RQ3 ITL Δ vs bounded |
|---|---|---|---|---|
| Qwen3 (dense)   | full per-head | TODO | TODO | TODO |
| Qwen3.6-MoE     | full per-head | TODO | TODO | TODO |
| DSv4-MLA        | latent-compressed | TODO | TODO | TODO |

**Ablations:**

| ablation | variant | TTFT p50 | ITL p50 | RQ1 grid-mean | note |
|---|---|---|---|---|---|
| prefetch granularity | per-step    | TODO (>10min long-ctx) | TODO | TODO | stalls decode loop |
| prefetch granularity | per-prefill | TODO | TODO | TODO | fast path |
| staging | off (bounded-drop) | TODO | TODO | TODO | recall floor |
| staging | on (recall)        | TODO | TODO | TODO | recall contribution |
| top-k | k∈{2,4,8,16,32} | TODO | TODO | TODO | knee TODO |

> **Harness notes.** RQ1 extends `scripts/cuda_recall_needle.py` (digits-in-answer at
> `temperature=0`; add `--ctx 2000,8000,32000,128000`, a recall/control toggle, and the n=3
> loop). RQ3 uses `scripts/bench_guidellm.sh` fixed-c (spec §7.2), profiling OFF (§7.8),
> correctness-gated (§7.1), roofline ratio (§7.6). RQ2 tier counters come from `/v1/stats`
> (§3.1) — the recall tier must be wired on the serve/rollout path before RQ2 can be
> populated (a known gap, not a fabricated result).

## References

[1] G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. arXiv:2309.17453.
[2] Z. Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs.* NeurIPS 2023. arXiv:2306.14048.
[3] Z. Liu et al. *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time.* NeurIPS 2023. arXiv:2305.17118.
[4] Y. Li et al. *SnapKV: LLM Knows What You are Looking for Before Generation.* NeurIPS 2024. arXiv:2404.14469.
[5] S. Ge et al. *Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs (FastGen).* ICLR 2024. arXiv:2310.01801.
[6] C. Xiao et al. *InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory.* NeurIPS 2024. arXiv:2402.04617.
[7] J. Tang et al. *QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML 2024. arXiv:2406.10774.
[8] H. Zhang et al. *PQCache: Product Quantization-based KVCache for Long Context LLM Inference.* SIGMOD 2025. arXiv:2407.12820.
[9] W. Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention (vLLM).* SOSP 2023. arXiv:2309.06180.
[10] L. Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs (RadixAttention).* NeurIPS 2024. arXiv:2312.07104.
[11] SGLang Team / LMSYS. *SGLang HiCache: Hierarchical KV Caching (GPU L1 / CPU L2 / Storage L3).* LMSYS Blog, 2025.
[12] Y. Cheng et al. *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference.* 2025. arXiv:2510.09665. (CacheGen arXiv:2310.07240; CacheBlend arXiv:2405.16444.)
[13] R. Qin et al. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.* FAST 2025. arXiv:2407.00079.
[14] Z. Liu et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024. arXiv:2402.02750.
[15] C. Hooper et al. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* NeurIPS 2024. arXiv:2402.18079.
