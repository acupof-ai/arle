# The KV cache is the memory: recalling a session's past instead of forgetting it

A transformer's KV cache is usually treated as scaffolding — a per-request decode
accelerator you allocate, fill, and throw away. ARLE now treats it as the thing
itself: **long-term memory**. A session's old KV doesn't get truncated when it
outgrows the GPU; it gets offloaded to a cheap tier and the *relevant* slices are
recalled back, so the model attends a fixed-size working set yet effectively sees an
unbounded history. Ten million tokens, served through a 256K-token window.

## The shape of the trick

At every decode step the model attends three regions, not the whole past:

- **sink** — the first few tokens (the attention anchors every long-context model
  leans on),
- **local** — the most recent window (what you're saying right now),
- **recall** — the top-k older *blocks*, scored by relevance to the current query.

The first two are StreamingLLM, and StreamingLLM alone forgets: anything that
scrolled out of the local window is gone. The third region is what turns a sliding
window into a memory. Each block keeps a tiny resident **mean-key rep** — one vector
— so even a block whose full KV has been pushed to NVMe is still *scorable*. Score
every block by `query · mean-key`, promote the winners' KV back, attend
`sink ∪ recalled ∪ local`. That structure — anchors that never leave, recent context
that never leaves, relevant history pulled back on demand — is also exactly the
shape of an agent's context: the task and the last few turns are always live; the
rest is recalled when it matters.

## What it buys, measured

On the real Qwen3.6-35B, a passkey buried mid-context:

| attend | KV used | answer |
| --- | --- | --- |
| everything (ceiling) | 5691 (100%) | correct |
| sink + local (StreamingLLM) | 288 (5.1%) | **wrong — forgot it** |
| sink + recall + local | 544 (**9.6%**) | **correct — same as full** |

Recall reproduces the full-attention answer at a tenth of the KV; the same budget
*without* recall misses entirely. And the working set is constant: a one-thousand-
and a one-million-token session both resolve to the same 56-token attention budget.
History grows; what the GPU holds does not.

## The unfashionable part that made it work

The mechanism is the fashionable bit. The discipline behind it is not, and it's the
reason the numbers are real: **license-or-kill every simplification on the actual
model before building it.** A source survey will happily tell you a design is fine.
This one had three forks that a survey would have shipped wrong:

- *Per-layer vs per-slot* — the validated baseline picks blocks independently per
  layer; the cheap design picks once per slot. We didn't assume it transferred. We
  ran it: per-slot retrieves at every depth. Licensed.
- *Stale query* — orchestrating recall from the host means scoring this step's
  blocks with last step's query. Plausibly fine; unproven. Ran it: still retrieves.
  Licensed.
- *The scoring-needs-full-K trap* — reading the kernel revealed that offloading a
  block (the whole point) would also make it unscorable, collapsing recall to a
  frozen subset. Caught *before* writing it, fixed by the resident mean-key rep.

Three cheap experiments, three traps that inference would have missed. The fast way
to ship a long-context feature that's quietly broken is to trust the diagram; the
fast way to ship one that works is to make every load-bearing assumption earn its
place against the model. Evidence over inference — even when, especially when, the
diagram already looks right.
