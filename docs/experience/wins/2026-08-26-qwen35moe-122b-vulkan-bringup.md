# Qwen3.5-122B-A10B on Vulkan: five blockers, each found by loading the model and reading the error

## Context / Goal
A 122B-A10B MoE checkpoint (`qwen35moe`, 63.65 GiB, 256 experts / 8 active)
sat on the box unused. Goal: make the Vulkan lane serve it, and find out what
was actually missing rather than what looked missing.

## Hypothesis
Going in, the two worries were **capacity** ("63.65 GiB will not fit") and
**architecture** ("the MoE path is built for the 35B-A3B, a 122B is a different
model"). Both were wrong. The heap is 74.43 GiB and the arch string is literally
the same one `model_qwen36.rs` already binds. Every real blocker was in the
plumbing between them.

## Params
- Model: `Qwen3.5-122B-A10B-UD-Q4_K_XL-0000{1,2,3}-of-00003.gguf` (Unsloth
  dynamic quant, 3-part split, 879 tensors)
- Backend: Vulkan, `arle serve --backend vulkan`
- Competitor: llama.cpp `726704a16`, same box, same file

## Env
- Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, RDNA 3.5, 40 CU), 128 GB LPDDR5X
- Windows 11, AMD proprietary 26.7.1 (LLPC)
- **Armoury Crate Performance mode, on AC** — see
  [the power-mode rule](2026-08-20-vulkan-coopmat-prefill-warptile.md)
- Date: 2026-08-26

## Results

It loads and generates. Three sequential requests, no crash:

| prompt | output | tok/s |
| --- | --- | ---: |
| `Q: What is the capital of France? A:` | `Paris.` | 12.84 |
| `Compute 17 * 23...` | `391` | 12.27 |
| `用一句话解释什么是内存带宽。` | answers in Chinese, echoes `内存带宽` | 18.5 |

llama.cpp on the same file: **tg128 = 23.41 t/s**, so ARLE is at ~0.53×.

### The checkpoint, measured
| dtype | tensors | elements | % |
| --- | ---: | ---: | ---: |
| **MXFP4** | 336 | 109,527,957,504 | **89.70%** |
| Q6_K | 59 | 10,791,419,904 | 8.84% |
| Q5_K | 36 | 887,095,296 | 0.73% |
| Q4_K | 1 | 762,839,040 | 0.62% |
| Q8_0 | 86 | 102,236,160 | 0.08% |
| F32 | 361 | 39,979,008 | 0.03% |

### The split, measured
| part | tensors | KVs |
| --- | ---: | ---: |
| 00001 | **0** | 55 (all arch + tokenizer metadata) |
| 00002 | 651 | 3 (`split.*` only) |
| 00003 | 228 | 3 (`split.*` only) |

## The five blockers, in the order the loader hit them

**1. No sidecar tokenizer.** `OpenAiTokenizer::from_model_dir` requires
`tokenizer.json`; a GGUF downloaded on its own has none, even though it carries
the whole vocab (248320 tokens, 247587 merges) *and* its chat template in
metadata. Fixed with `scripts/gguf_extract_tokenizer.py`, which mirrors
`GgufTokenizer::from_gguf` exactly — if one changes, so must the other, or a
model tokenizes differently depending on which path loaded it.

**2. Split GGUF unsupported.** `GgufFile` held one mmap and read one path. Part
1 has `tensor_count == 0`, so pointing at it gave full metadata and no weights,
surfacing far downstream as *"GGUF has neither qwen35moe.vocab_size nor
token_embd.weight/output.weight dims"*. Pointing at part 2 or 3 was worse:
those carry no `general.architecture`, so it defaulted to `"qwen3"` and a 122B
MoE loaded as a dense model. Now a `Vec<Shard>` + per-tensor shard index.
**Offsets restart at 0 in each part**, so `tensor_data` must resolve against
its own shard's `data_start` — concatenating the blobs would be silently wrong.

**3. MXFP4 had no pinned block size.** `type_size()` had no arm → `byte_len()`
returned `None` → upload aborted on the first tensor. It is 17 B per 32 values.

**4. MXFP4 fell through to `DequantF16`.** This is the one that would have
looked like "the box is too small". 17 packed bytes per 32 values expand to 64
as F16, and MXFP4 is 89.7% of this checkpoint, so the plan came to **213 GiB
against a 74.43 GiB heap**. Adding `KeepQuant` brings it to ~63 GiB. Writing a
CPU MXFP4 dequant instead would have *moved* the failure, not removed it:
109.53e9 elements × 2 B ≈ 219 GB.

**5. The MoE lane never reset carried state.** The dense lane resets when
`start_pos == 0`; `model_qwen36.rs` never got that arm. So the **second**
request died with *"start_pos 0 != materialized seq_len 50"* and took the
server down — one request into a fresh serve. The first request always worked,
which is exactly why a single smoke test would have called this shipped.

## Verification
| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --no-deps -- -D warnings` | PASS |
| `cargo test -p infer-gguf --lib` | PASS |
| vulkan suites, `--test-threads=1` | PASS, 38 tests / 15 suites |
| 122B loads, 3 sequential requests | PASS, no crash |
| output quality: English, arithmetic, Chinese | PASS (`Paris.`, `391`, CJK) |

The `????` a PowerShell client showed for Chinese was **PowerShell 5.1's
`Invoke-RestMethod` UTF-8 decoding bug**, not the model — a raw `urllib` fetch
of the identical request returns proper CJK. Verify a suspected encoding bug
through a second client before filing it against the server.

## Learnings

**Capacity was the fear and plumbing was the problem.** An older entry recorded
this box's unified memory as 63.6 GB, which would have made a 63.65 GiB
checkpoint hopeless. `vulkaninfo` says heap 1 is **74.43 GiB DEVICE_LOCAL**
(70.71 GiB budget-free). Nothing in ARLE reads
`VkPhysicalDeviceMemoryProperties` at startup — `memory_heaps()` exists in
`vulkan-sys` with **zero call sites** — so the number nobody logs was the number
everyone guessed at.

**"Unsupported dtype" was 90% of the model, and it was a define.** MXFP4 read
as an exotic format needing a new kernel. The vendored `mul_mat_vecq.comp`
already had `DATA_A_MXFP4` arms; the work was two build.rs variants, two enum
entries, two dispatch arms. Grep the vendored shaders before estimating a
kernel.

**Loading the model beat analyzing it, and the two agreed.** A static survey and
an empirical load ran concurrently. The survey produced the same blocker list
with file:line and the exact byte layouts; the load produced the same list as a
sequence of error messages, in the order they bite. The survey was worth it for
*why* and *how big*; the load was worth it for *which one is next*. Neither
alone would have sequenced the work correctly.

**A second request is a different test from a first one.** Blocker 5 is
invisible to any smoke test that sends one prompt, and it is fatal — the server
exits for watchdog restart.

## Rule
When a new checkpoint will not load, drive it by the error message, not by the
architecture diagram: fix what the loader actually complains about and re-run.
The blockers arrive in a fixed order (tokenizer → container → dtype size →
residency → kernel → state lifecycle) and each one hides the next, so any
up-front estimate of "what's missing" is a guess at everything past the first.

And before concluding a model is too big, print the heap. A remembered capacity
number is not a measurement, and this one was off by 11 GiB in the direction
that would have cancelled the work.
