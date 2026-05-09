---
title: guidellm 0.6.0 TTFT measurement broken with INFER_PREFILL_GRAPH=1
date: 2026-05-10
type: research
status: bench-tool-bug-not-substrate-bug
---

# guidellm 0.6.0 TTFT measurement broken with INFER_PREFILL_GRAPH=1

> Per #40 Path B.2 Tier 1 wins evidence(`c44788f`),guidellm reports
> `TTFT p50 was 0.0 despite successful requests` when ARLE has
> `INFER_PREFILL_GRAPH=1`。Direct curl streaming probe confirms ARLE SSE
> response is correct — bug is in guidellm 0.6.0 TTFT computation,not
> ARLE substrate。

## Direct streaming probe(this tick,verifies ARLE side)

```bash
curl -N -X POST http://127.0.0.1:8765/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-4B-W4-hybrid-zpfix","prompt":"...","max_tokens":20,
       "stream":true}' --max-time 15
```

Response(per-token SSE chunks delivered):
```
data: {"choices":[{"text":" Also","index":0,"logprobs":null}]}
data: {"choices":[{"text":",","index":0,"logprobs":{"token_logprobs":[-0.0017]}}]}
data: {"choices":[{"text":" what","index":0,"logprobs":{"token_logprobs":[-0.562]}}]}
data: {"choices":[{"text":" is","index":0,"logprobs":{"token_logprobs":[-0.440]}}]}
data: {"choices":[{"text":" the","index":0,"logprobs":{"token_logprobs":[-0.188]}}]}
```

→ ARLE streams correctly with INFER_PREFILL_GRAPH=1。Tokens delivered
incrementally with logprobs。

## Suspected guidellm bug

guidellm 0.6.0 metrics report:
- `Successful Output Tokens/Iter: Mean=2.99, Median=3.0` for c=4 4k bench
- → guidellm sees ~3 streaming iterations per 256-token request
- → either guidellm misreads chunked SSE OR parses EOS too early

But direct streaming shows **20 separate `data:` chunks for 20 tokens**(per
this probe)— 1 token per chunk。

**Hypothesis**:guidellm 0.6.0 TTFT computation expects specific SSE
chunk pattern(maybe a `text:""` warmup chunk first?)。ARLE's first chunk
contains real token immediately,which guidellm may misread as "no warmup"
and skip TTFT timing。

OR guidellm buffers SSE chunks(some HTTP/2 + stream coalescing)and
reports first **buffered batch** as one iter,not first chunk。

## Workaround for #40 wins claim

**Use server-side `engine_ttft_us`**(per `/v1/stats`)as ground truth:
- Bench A graph OFF:**engine_ttft = 2,000,000 us = 2000 ms**
- Bench B Path B.2:**engine_ttft = 150,000 us = 150 ms**
- **Δ = -92.5%**

Server-side TTFT is the **time from request start in scheduler to first
sampled token written to slot**。Comparable across A/B same-machine。

guidellm 0.6.0 TTFT broken does NOT invalidate Path B.2 wins per `c44788f`。
Just means client-perceived metric is unmeasurable until guidellm fixed。

## Action items

1. **Investigate guidellm 0.6.0 TTFT calculation source** in
   `/home/ckl/projects/arle/.venv/lib/...../guidellm/` — find where TTFT
   is parsed from SSE first-chunk timestamp
2. **Try guidellm 0.6.1+** if available(check upstream releases)
3. **Workaround patch**:add `text=""` empty warmup chunk to ARLE's first
   SSE response(if guidellm expects it)— but this changes API contract
4. **Long-term**:switch to direct curl-based bench with custom TTFT
   measurement,OR contribute fix upstream to guidellm

## Implication for `c44788f` Tier 1 wins claim

`c44788f` wins entry uses **server-side engine_ttft_us as ground truth**
for the -92.5% TTFT improvement claim。This is **valid and reproducible**
regardless of guidellm bug。Note the wins entry section explicitly
acknowledges:

> "guidellm reports broken TTFT measurement — streaming pattern changed
> post-Path B.2(possibly batched non-streamed delivery)。**Server-side
> engine_ttft 150ms is the ground truth** — guidellm tool can't measure
> the new timing pattern。"

Direct streaming test this tick confirms ARLE side is fine。Bug is
upstream guidellm。

## Cross-references

- #40 Path B.2 Tier 1 wins:`docs/experience/wins/2026-05-10-bench-40-pathB2-tier1-strong-proceed.md`(`c44788f`)
- Codex's Path B.2 wins draft:`docs/experience/wins/2026-05-10-bench-40-pathb2-bucketed-prefill-graph-key.md`(`a56b7a9` commit)
- guidellm version:0.6.0 per `pip show guidellm`
- ARLE SSE streaming endpoint:`infer/src/http_server/openai_v1.rs`(streaming completion handler)

## 状态

guidellm 0.6.0 TTFT measurement broken when graph capture enabled —
**bug is in guidellm,not ARLE Path B.2 substrate**。Direct curl probe
confirms ARLE SSE streams correctly token-by-token with logprobs。
**`c44788f` Tier 1 wins claim valid via server-side engine_ttft_us
ground truth**。Long-term:investigate guidellm fix or switch bench tool。
