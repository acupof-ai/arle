# W4A8 bug — H4 + H5 RULED OUT,H1 algebra OK,H3 last suspect

> Continues hypothesis-elimination chain:
> [`e20f24c`](2026-05-08-w4a8-bug-claude-investigation.md) ranked 5 hypotheses,
> [`b65c8c6`](2026-05-08-w4a8-bug-h2-ruled-out.md) ruled out H2 (FP16/BF16
> mismatch), this entry rules out H4 + H5 by direct kernel read.
>
> Hypothesis space now reduced from 5 → 1.5 (H3 prime + H1 algebra-checked
> but unit-test pending).

## H4 RULED OUT — kernel correctly subtracts 8 (INT4 zero point)

`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu::dequant_per_group`
(line 174-216):

```cpp
static constexpr uint32_t SUB = 0x64086408;  // FP16 half2: each lane = 1032
// ...
*reinterpret_cast<half2*>(&t0) = __hsub2(
    *reinterpret_cast<half2*>(&t0),
    *reinterpret_cast<const half2*>(&SUB)
);
```

Decoding `0x6408` as FP16:
- `0x6408` = sign=0, exp=11001=25 (biased: 25-15=10), mantissa=0000001000=8/1024
- Value = (1 + 8/1024) × 2^10 = 1024 + 8 = **1032**

Earlier in `dequant_per_group`:
```cpp
static constexpr uint32_t LO = 0x000f000f;
static constexpr uint32_t EX = 0x64006400;  // half2: each lane = 1024
uint32_t t0 = lop3<...>(q, LO, EX);  // packs (q & 0xF) into FP16 mantissa, exp fixed
```

`t0` interpreted as FP16 has value `1024 + (q & 0xF)` per lane. Then `t0 - SUB = 1024 + (q & 0xF) - 1032 = (q & 0xF) - 8`. **Symmetric zero point [-7, 7] correctly recovered.**

H4 (kernel forgets `int4 - 8`) is RULED OUT.

## H5 RULED OUT — activation INT8 quant correct

`crates/cuda-kernels/csrc/gemm/w4a8_activation_quant.cu` (59 LOC):

```cpp
float scale = smem[0] > 0.0f ? smem[0] / 127.0f : 1.0f;  // line 33: INT8 range ✅
// ...
float qf = nearbyintf(__bfloat162float(in_row[col]) / scale);  // line 39: divide ✅
qf = fminf(127.0f, fmaxf(-128.0f, qf));                         // line 40: clamp ✅
out_row[col] = static_cast<int8_t>(qf);                         // line 41: cast ✅
```

Per-row `s_act = max_abs / 127` is the standard INT8 dynamic-range scale.
Reduce-max + scale + clamp + cast — all correct.

H5 (activation INT8 wrong scale) is RULED OUT.

## H1 status — algebra checked, needs unit-test confirmation

`/tmp/quantize_qwen3_w4a8.py` (read at `e20f24c`):

```python
s_channel = max_per_channel(weight) / 127.0  # f32 [1, n]
s_per_group = max_per_group(weight) / 7.0    # f16
s_group_stored = s_per_group / s_channel     # f16  ← stored as `marlin_w4a8_s_group`
w_int4 = round(weight / s_per_group) + 8     # ∈ [0, 15]
```

Kernel reconstruction (assumed):
```
weight_recovered = (int4 - 8) × s_group_stored × s_channel × s_act × INT32_acc
                 = ... × (s_per_group / s_channel) × s_channel × ...
                 = ... × s_per_group × ...
```

**Algebra is correct** if kernel multiplies BOTH `s_group_stored` AND `s_channel`.
This was assumed earlier; codex confirmed at `b65c8c6` that kernel reads s2 (channel) and s3 (group) consistently from the FFI args.

The remaining H1 risk is **off-by-one or transpose error** in the script's
indexing. The permutation `scale_perm` at line 56-60 of the script is the
load-bearing part:

```python
scale_perm = []
for i in range(8):
    scale_perm.extend([i + 8 * j for j in range(8)])
```

This generates `[0,8,16,...,56, 1,9,17,...,57, ..., 7,15,...,63]` — a column-major
re-ordering of 64 elements. The kernel's expected load order must match.

**H1 status: algebra OK, but `scale_perm` ordering is unit-test-required to
verify**. If kernel expects `[0..63]` row-major and script writes column-major
(or vice versa), all per-group scales are misordered → garbage output.

This is closely related to H3 (the broader perm layout question).

## H3 PRIME SUSPECT — `get_perms()` vs kernel mma fragment

The script's `get_perms()` at line 33-61 generates a permutation that
re-arranges 8192 weight bytes per tile to match Marlin's tensor-core
fragment loading convention.

Marlin permutations are **kernel-version specific**: original Marlin
W4A16 (FP16 mma) uses one layout; W4A8 (INT8 input + W4 weight, smem
LDMATRIX) may need a different layout; Hopper Marlin (TMA-based) uses
yet another.

If codex's W4A8 PR #31 cherry-pick uses the **original W4A16 permutations**
in the script but the kernel internally expects **W4A8-specific layout**,
all packed weights are byte-misordered → kernel reads wrong INT4 values
per tile → wrong output.

### Diagnostic for codex (medium-cost)

1. Read `marlin_w4a8_kernel.cu::cp_async4_stream` (line 491) to see how
   tile bytes are arranged in shared memory.
2. Read `frag_b_quant` reads (line 463 + nearby `dequant_per_group` calls)
   to see expected fragment layout.
3. Compare to `get_perms()` script line 33-54 — confirm permutation
   indices match the kernel's tile fragment shape (likely 16×8×16 for INT8
   mma vs 16×8×8 for FP16 mma).
4. If mismatch → this is the bug.

### Reference

`get_perms()` script line 33-54:
```python
def get_perms(groupsize: int, k: int):
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [4*(i%4), 4*(i%4)+1, 4*(i%4)+2, 4*(i%4)+3]:
                perm1.append(16*row + col + 8*block)
        for j in range(4):
            perm.extend([p + 256*j for p in perm1])
    perm = np.array(perm)
    if groupsize == k:
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
    else:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
```

The `[4, 0, 5, 1, 6, 2, 7, 3]` interleave is recognizable as the **W4A16 Marlin**
column re-shuffle pattern. The W4A8 kernel may need a **different interleave**
(e.g., `[0, 1, 2, 3, 4, 5, 6, 7]` or `[2, 0, 3, 1, 6, 4, 7, 5]`).

## Hypothesis space summary

| H | Description | Status | Confidence |
|---|---|---|---|
| 1 | quantize script scale-perm ordering | algebra OK; perm needs unit-test | low |
| 2 | s3 dtype FP16/BF16 mismatch | RULED OUT (b65c8c6) | — |
| 3 | get_perms vs kernel mma fragment | **PRIME SUSPECT** | high |
| 4 | int4 - 8 offset | RULED OUT (this entry) | — |
| 5 | activation INT8 scale | RULED OUT (this entry) | — |

**Conclusion**: codex investigation focuses on H3 (matrix-fragment-permutation
match) with H1 unit-test as parallel sanity check.

## Skill methodology applied

This 3-tick chain (`e20f24c` → `b65c8c6` → this entry):
- Tick 1: read 3 source files, rank 5 hypothesis (Claude)
- Tick 2: kernel s3 dtype check (codex)
- Tick 3: kernel dequant_per_group + activation_quant.cu read (Claude)

Cumulative cost: ~2 hours wall-clock (Claude reading + codex 5-min diag).
Hypothesis space narrowed 5 → 1.5 with grounded code evidence at each step.
Per skill anti-pattern #13 (NULL is real elimination) — ruled-out hypotheses
are real progress.

## Cross-references

- H2 ruled out (codex): [`2026-05-08-w4a8-bug-h2-ruled-out.md`](2026-05-08-w4a8-bug-h2-ruled-out.md) (`b65c8c6`)
- H1-5 initial ranking (Claude): [`2026-05-08-w4a8-bug-claude-investigation.md`](2026-05-08-w4a8-bug-claude-investigation.md) (`e20f24c`)
- W4A8 garbage output errors entry: [`../experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md`](../experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md) (`81b6481`)
- Skill v1.3.0: [`.claude/skills/kernel-optimization/SKILL.md`](../../.claude/skills/kernel-optimization/SKILL.md) (`d09480b`) — anti-pattern #13 NULL elimination
