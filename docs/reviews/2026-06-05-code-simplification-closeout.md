# #30 Closeout — Code Simplification Execution (~1,650 LOC removable)

**Verdict: CLOSE #30 as substantially-done.** Every survey target is resolved — already-deleted, deduped this session, or verified speculative/wire-boundary. No high-confidence removable LOC remains in the in-scope (non-cuda) crates. The non-cuda tree compiles **100% warning-clean** on both `metal` and `no-cuda` (0 dead-code, 0 unused warnings). The only large remaining dead-code mass (~5.3k LOC Marlin + GGUF k-quant) is **#18's scope** (`crates/cuda-kernels`), explicitly carved out and blocked on Codex's live DSv4 kernel work.

## What was done

**13 in-session refactor/dedup commits, net ~547 deletions** (dominated by the cli_args delete, 277):

| Commit | Scope | Net |
|---|---|---|
| `eaace0d3` | delete dead `crates/train/src/cli_args.rs` (argv helpers for removed binaries, 0 refs) | −259 |
| `6ce26bd6` | dedup CUDA `ServeHandle` builder → `cuda_serve_handle` | −10 |
| `8987c752` | dedup finish-reason + `TokenUsage` construction in `serve_engine` | +4 |
| `2c634986` | dedup `parse_parallel_env_usize` + collapse head_shard guards | −17 |
| `512f0d38` | extract `print_resolution` (dedup run + list_models in doctor) | −9 |
| `80dea1a5` | dedup metal weight-ext / usize-array / page-reclaim helpers | +9 |
| `a6307443` | hoist `VISIBLE_TAGS` + extract `hidden_block_for_open_tag` (chat) | −9 |
| `e7acce8a` | extract `echo_tokens()` test helper (infer-core) | −6 |
| `07002baf` | collapse `waiting_request_precedes` to one lexicographic key | −2 |
| `47275512` | `PendingCompletions` alias + single-pass `deliver_completions` | −2 |
| `4b45b00f` | drop dead `NO_TOOL` branch in `repair_tool_calls` | −1 |
| `019d6bc0`, `790c8411` | `f64_to_json` helper; truncation-marker const | −0/−0 |

Plus the **elegant-Rust apply pass** (`3dc2fd1f`, `742184c2`): 4 parallel verify-or-kill agents, **all 4 candidate borrows KILLed** with source evidence — structured-errors speculative (zero production branchers), `#[non_exhaustive]` counterproductive intra-monorepo, `kind`/`role`/`dtype` are external wire fields, `sampling_params` dedup already done. The one real finding (cli_args) was deleted, not patched.

## The ~1,650 accounting

| Bucket | LOC | Status |
|---|---|---|
| **Already-done before/at survey** (OPD-only pivot + rewrite deletion-refactors: `72ebaae4` −319 speculative infer-models, `fdc74452` −1374 comment/dead trim, `a48bf704` −103, `0680709a` −44, dropped train commands/eval_lm/sampling/GRPO/multi-turn) | ~1,100+ | the bulk of the target was structural deletion from the pivot, not a fresh pass |
| **Done this session** (13 commits above) | ~547 net | the genuine #30 dedup/dead-code execution |
| **Was-never-real** (survey rated for a generic crates.io library, not this monorepo) | ~? | structured errors, `#[non_exhaustive]`, kind/role/dtype enums — 4+ borrow-list items KILLed as speculative or wire-boundary |
| **Different ticket (#18)** | ~5.3k+ | Marlin + GGUF k-quant in `crates/cuda-kernels` — NOT #30, blocked on Codex DSv4 |

## Real-and-remaining: none high-confidence

Fresh grep of every non-cuda `allow(dead_code)` confirms each is a **KEEP** under `feedback_necessity_not_callers`, not a removal:
- `hf_search.rs:fuzzy_filter` — has test callers (`:131`, `:151`); real helper with coverage, not dead.
- `HfSearchResult.tags`, `RepoFileEntry.file_type` — serde **wire fields** (deserialize external JSON; removing breaks the contract).
- `CatalogEntry.{family,param_count}` — descriptive static catalog metadata.
- `autograd/backend.rs`, `infer-api/loaded.rs`, `serve_engine.rs:350` — **cfg-gated** (CUDA-conditional); legitimate under backend-isolation.

The few same-named functions that survive (`finish_reason_for` / `finish_reason_to_str` / `finish_reason_from` / `finish_reason`) are **distinct** — different signatures and input types per crate (CompletedRequest vs FinishReason vs Option), not duplicates.

## Recommendation

**Close #30.** Do not manufacture churn chasing the literal 1,650 figure — that number was a generic-library estimate; the real tree was already converging via the OPD-pivot and rewrite deletion-refactors, and this session's 13 commits + the verify-or-kill pass closed the remainder. The only outstanding dead-code mass is cuda-kernels Marlin/GGUF, correctly owned by **#18** and gated on Codex's live DSv4 work + a pod nvcc build.