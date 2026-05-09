# #24 substrate cleanup audit prep

> 2026-05-09 EOD+151 — concrete catalogue of dead substrates accumulated
> through the 2026-05-08 → 2026-05-09 KILL cycle。Prep doc for #24 task
> "3 KILLED substrate half-state cleanup(1 周观察期)" deferred to
> **2026-05-14**。
>
> §0 SOLID rule:**no half-states** — accumulated dead code violates this。
> Cleanup is mandatory once observation period closes。This audit gives
> concrete file:line references + KILL evidence + recommended action so
> 2026-05-14 cleanup is mechanical not exploratory。

## Audit method

Each candidate verified via:
1. `grep -rnE "<symbol>" infer/src/` — verify zero callers in serving runtime
2. File size 量化 substrate footprint
3. KILL evidence link(commit hash + errors entry / research entry)
4. Recommended action(delete / deprecate / preserve as reference)

## Candidates(grep-verified)

### Tier 1 — Pure dead code(zero callers,no env-gated reactivation)

#### 1. `crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu`
- **Size**:23,752 bytes
- **Callers**:0(grep `decode_attention_quantized_cuda` infer/src/ → empty)
- **Origin**:Earlier quant decode hypothesis,never wired
- **Cross-ref**:Flagged in `2e21da1` ops-layer roadmap §"Substrate gap audit"
- **Action**:**DELETE**(plus its build entry in `crates/cuda-kernels/build.rs` if any)

#### 2. `crates/cuda-kernels/csrc/attention/decode_attention_turboquant.cu`
- **Size**:14,128 bytes
- **Callers**:0
- **Origin**:Turboquant 实验,未产生 production wire
- **Cross-ref**:`2e21da1` ops-layer roadmap
- **Action**:**DELETE**

#### 3. `crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu`
- **Size**:14,530 bytes
- **Callers**:0(grep `decode_attention_varlen_fp8_cuda` infer/src/ → empty)
- **Origin**:Variable-length FP8 attention 实验,未 wire
- **Cross-ref**:`2e21da1` ops-layer roadmap;P1.4 KILL `51dd5b2` 进一步证明 FP8 decode 路线 substrate 语义不对齐(此文件并非 P1.4 wire 目标但 belongs 同 FP8 dead family)
- **Action**:**DELETE**

**Tier 1 total**:~52 KB CUDA C dead code,3 files

### Tier 2 — Env-gated dead paths(behavioral dead code,not file-level)

#### 4. `infer/src/ops/linear.rs:99-112` R4 #6 env-gated override
- **Source**:`3b9cc06` KILL commit + EOD+106 bench evidence
- **Code shape**:
  ```rust
  if batch > 1
      && marlin_prefill_aligned(weight).is_ok()
      && !(batch <= 8
          && std::env::var("INFER_R4_W4A16_GEMV_OVERRIDE")
              .as_deref()
              .ok()
              == Some("1"))
  {
      return Self::MarlinW4Gemm;
  }
  ```
- **Behavior**:Env-var `INFER_R4_W4A16_GEMV_OVERRIDE` defaults unset → evaluates to `Some("1") != None` → false → outer condition behaves as `if batch > 1 && marlin_prefill_aligned(weight).is_ok()`,即 always Marlin。Env-on path is the KILLED hypothesis(W4A16BatchGemv override,+37% ITL regression)。
- **Cross-ref**:`3b9cc06` empirical KILL + comment block lines 96-102 + `docs/research/2026-05-09-eod106-r4-6-bench-preliminary-solid-gap.md`
- **Action**:**SIMPLIFY** — remove env-var branch + comment block,leave `if batch > 1 && marlin_prefill_aligned(weight).is_ok() { return MarlinW4Gemm; }`
- **Risk**:Low — env-var path was empirical KILL,no production user

### Tier 3 — Built but unwired AOT cubins

#### 5. TileLang HD64 paged decode/prefill cubins
- **Symbols**(per `crates/cuda-kernels/src/ffi/attention.rs`):
  - `tilelang_batch_decode_paged_hd64_*_run_cuda`
  - `tilelang_batch_prefill_paged_hd64_*_run_cuda`
- **Callers**:0(grep `tilelang_batch_decode_paged_hd64` / `tilelang_batch_prefill_paged_hd64` infer/src/ → empty)
- **Origin**:Historical small-model substrate(HD64 = some Qwen-tiny variants);Qwen3/3.5 全是 HD128
- **Cross-ref**:`2e21da1` ops-layer roadmap "0 callers"
- **Action**:**DELETE** AOT macros + corresponding TileLang DSL source(reduces cubin build time)

#### 6. TileLang HD128 FP8 paged decode cubin(`tilelang_batch_decode_paged_hd128_fp8_q32_kv8_run_cuda`)
- **Symbols**:Macro defined `crates/cuda-kernels/src/ffi/attention.rs:735` `tilelang_decode_hd128_fp8_decl!`
- **Callers**:0(P1.4 wire reverted per `51dd5b2`)
- **Origin**:M_quant Phase A0 substrate(2026-05-06)
- **KILL evidence**:`51dd5b2` P1.4 KILL — substrate 语义不对齐 with ARLE existing FP8 KV cache(scale layout / FP8 cast / dequant 不一致)
- **Action**:**EITHER**
  - (a) **DELETE** — accept that TileLang FP8 path is dead until substrate semantic alignment work happens
  - (b) **PRESERVE + DOCSTRING** — leave for future P1.4 v2 diagnostic(custom FP8 vs TileLang FP8 attention output diff,per codex's own proposal)
- **Recommended**:(b) preserve until P1.4 v2 verdict;if v2 KILL → delete with v2 KILL commit;if v2 LAND → keep + wire

### Tier 4 — Documented dead-end paths(comments / research entries)

#### 7. `infer/src/ops/linear.rs:749-773` quantized fused_mlp 4-launch fallback
- **Status**:NOT dead — production-default path for quantized weights
- **Why mentioned**:P1.3 KILL `edacfe7` proved this path is **already-optimized**(autotune found good algo,fusion 反 saturate)
- **Action**:**PRESERVE** + add inline comment ref to `edacfe7` errors entry explaining "P1.3 KILL evidence:do not attempt launch reduction here"。Prevents future re-attempt at saturated hypothesis。
- **Effort**:1 LOC comment

## Out-of-scope candidates(do NOT cleanup)

- **TileLang HD128 BF16 paged decode/prefill** — production hot path
- **TileLang HD256** — Qwen3.5-32B+ future support
- **TileLang HD128 BF16 split-KV** — recently LANDED long-ctx decode path
- **MLA decode kernel** — DeepSeek V4 readiness substrate(intentional)
- **Any csrc/quant/** kernel — all production-wired

## Cleanup execution recipe(2026-05-14)

```bash
# Tier 1: file deletion
git rm crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu
git rm crates/cuda-kernels/csrc/attention/decode_attention_turboquant.cu
git rm crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu
# Plus remove their build.rs entries / FFI declarations

# Tier 2: surgical Edit on linear.rs:99-112
# Replace with simplified if-block(see Tier 2 above)

# Tier 3: TileLang HD64 AOT macro removal in cuda-kernels build.rs / DSL
# (FFI macro at attention.rs around hd64 decl)

# Tier 4: comment add at linear.rs:749 referencing edacfe7
```

## Verification gates

After cleanup:
- `cargo check --release -p infer --features cuda` PASS
- `cargo build --release -p infer --features cuda` PASS
- `cargo test --release -p infer --features cuda --test e2e` PASS
- Bench:`scripts/bench_guidellm.sh post-substrate-cleanup` σ<5%,no regression vs current main

## Cross-references

- `2e21da1` ops-layer roadmap §"Substrate gap audit"(original flagging)
- `3b9cc06` R4 #6 KILL commit
- `edacfe7` P1.3 KILL commit + errors entry
- `51dd5b2` P1.4 KILL commit + errors entry
- `2778dc8` anti-pattern #26 candidate research(EOD+149)
- Task #24:"3 KILLED substrate half-state cleanup(1 周观察期)" — observation period ends 2026-05-14
- §0 SOLID rule "no half-states"

## Status

Audit prep COMPLETE。**Mechanical cleanup ready to execute 2026-05-14**(or
sooner if user direction green-lights early)。

Estimated cleanup effort:
- Tier 1:~10 min(rm + build.rs edits)
- Tier 2:~5 min(surgical Edit + lint)
- Tier 3:~30 min(AOT macro + TileLang DSL source removal,verify cubin
  build still passes)
- Tier 4:~2 min(1 LOC comment)
- **Total:~50 min cleanup + verification gates**

Gross substrate reduction:~52 KB CUDA C(Tier 1)+ ~14 LOC env-gated dead
path(Tier 2)+ HD64 AOT entries(Tier 3 LOC TBD)+ FP8 HD128 cubin(Tier
3 if (a) accepted)。

Net codebase health benefit:eliminates "what is this for" confusion for
future contributors,reduces cubin build time,enforces §0 SOLID
no-half-states rule。
