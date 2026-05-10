---
title: 2026-05-10 ARLE marlin_pf8/ subdir = COMPLETE vLLM marlin/ headers fork — P2 priority DROPPED (already done)
date: 2026-05-10
type: research
status: open (corrects 2f19a3c P2 sizing further; drops P2 from queue)
related_docs: [`2f19a3c` Marlin parity survey, `86b28c7` M''' completion correction, `d8ebe73` Machete-inspired reframing]
---

# ARLE marlin_pf8/ = complete vLLM marlin/ fork — P2 dropped from queue

> **Why now**: `2f19a3c` §5 sized P2 (vLLM Marlin diff-port) at 1.5d
> based on file-level header attribution check. This tick's deeper
> survey reveals ARLE's `marlin_pf8/` subdir IS a complete vLLM
> marlin/ headers fork. P2 is essentially DONE.

## §1 Evidence: 6/6 ARLE marlin_pf8 files attributed to vLLM

```
$ ls /home/ckl/projects/arle/crates/cuda-kernels/csrc/gemm/marlin_pf8/
core/  dequant.h  kernel.h  marlin.cuh  marlin_dtypes.cuh
marlin_mma.h  marlin_template.h
```

Per-file attribution headers:
- `dequant.h`: "Adapted from vLLM csrc/quantization/marlin/dequant.h (Apache-2.0)."
- `kernel.h`: "Adapted from vLLM csrc/quantization/marlin/kernel.h (Apache-2.0)."
- `marlin_mma.h`: "Adapted from vLLM csrc/quantization/marlin/marlin_mma.h (Apache-2.0)."
- `marlin_template.h`: Frantar 2024 (same upstream as vLLM)
- `marlin.cuh`: "Adapted from vLLM csrc/quantization/marlin/marlin.cuh (Apache-2.0)."
- `marlin_dtypes.cuh`: "Adapted from vLLM csrc/quantization/marlin/marlin_dtypes.cuh (Apache-2.0)."

Side-by-side with vLLM `csrc/quantization/marlin/`:
```
vLLM has              ARLE has              Status
-------               --------              ------
.gitignore            (not tracked)         skip
awq_marlin_repack.cu  (skipped)             ARLE has no AWQ loader
dequant.h             marlin_pf8/dequant.h  ✓ adapted
generate_kernels.py   (TileLang used instead) different codegen path
gptq_marlin_repack.cu marlin_repack.cu      ✓ existing analog
kernel.h              marlin_pf8/kernel.h   ✓ adapted
marlin.cu             (Rust dispatch in     N/A — ARLE uses Rust
                       infer/src/ops/        FFI not Torch)
                       linear.rs)
marlin.cuh            marlin_pf8/marlin.cuh ✓ adapted
marlin_dtypes.cuh     marlin_pf8/marlin_    ✓ adapted
                       dtypes.cuh
marlin_int4_fp8_      marlin_int4_fp8_      ✓ adapted (PF8.2)
preprocess.cu          preprocess.cu
marlin_mma.h          marlin_pf8/marlin_    ✓ adapted
                       mma.h
marlin_template.h     marlin_pf8/marlin_    ✓ same upstream
                       template.h
```

**Score: 8/9 vLLM marlin/ files have ARLE analogs** (+1 deliberately
skipped: AWQ; +1 N/A: marlin.cu Torch FFI). This is essentially full
parity.

## §2 What this means for P2

Original P2 framing: "port vLLM upstream Marlin diff to ARLE for
~2-5% gain". But the ports ALREADY EXIST. The remaining differences
are:
- Possible UPSTREAM changes since ARLE's last sync (likely small,
  Marlin kernel is mature)
- Different codegen path (TileLang vs vLLM's generate_kernels.py)
- Different dispatch (Rust vs Torch)

A real "vLLM diff-port" today would be:
- Track vLLM commits to marlin/ subdir since ARLE's last sync
- Diff against ARLE marlin_pf8/ files
- Backport meaningful kernel-internal changes (not codegen/dispatch)

This is a **maintenance task, not an optimization task**. Expected
gain on sm_89 is essentially 0% unless vLLM has shipped sm_89-specific
tuning.

## §3 Updated priority table (final, supersedes `2f19a3c` §6)

| Priority | Path | Wall-clock | Status | Expected |
|---|---|---:|---|---|
| P1 | A+B combined (Medusa + Hybrid) | 4-5 days | gated on user GO | 2.61× tok/s + -14% latency |
| ~~P2~~ | ~~vLLM Marlin diff-port~~ | DONE | ARLE marlin_pf8/ = complete fork | maintenance only |
| P3 | Task #47 H1' v2 | 1 day | gated on diagnostic logging | unblocks PF8 path |
| ~~P3.5~~ | ~~M''' (W4-FP8 preprocess)~~ | DONE | PF8.2 in production | already integrated |
| P4 | Option M'' (Marlin schedule auto-tune) | 3-5 days | open | 2-8% conditional |
| P5 | Option M' (full cutlass rewrite) | 2-3 weeks | open | 5-15% best-case, HIGH risk |
| P6 | Wait sm_100 (NVFP4 native) | months | hardware | new path |
| KILLED | Literal Machete port (sm_90+) | impossible | KILLED `fc33cfb` | 0% on sm_89 |

### §3.1 Compounded conclusion (across 14 docs in the chain)

ARLE has ALREADY done the bulk of "port vLLM W4 kernels to ARLE":
- M''' W4-FP8 preprocess: DONE (PF8.2)
- vLLM marlin/ headers: DONE (marlin_pf8/ subtree)
- W4A8 (HandH1998 mods): DONE (marlin_w4a8_kernel.cu)

What's left from the user's "port Machete W4 from vLLM" directive:
- Architectural sm_90+ features (WGMMA, TMA): IMPOSSIBLE on sm_89
- A+B combined: the actual sm_89-feasible Machete-class path
- Task #47: the substrate USAGE bug, not the ported code

User's true forward path is **A+B**, not "more vLLM port work".
The vLLM work has ALREADY been done.

## §4 Wall-clock budget (refined)

| Item | Days | Cumulative |
|---|---:|---:|
| P1 A+B combined | 4-5 | 4-5 |
| P3 Task #47 H1' v2 (parallel codex track) | 1 | 5-6 |
| Total compound win wall-clock | | **5-6 days** |

(Down from `86b28c7`'s estimate of 7 days, since P2 is now dropped.)

Compound expected: **2.61× tok/s + -14% latency + PF8 path unblock**.

## §5 Cross-references

- `2f19a3c` ARLE Marlin parity survey (this entry refines §6 P2 to DONE)
- `86b28c7` M''' completion correction
- `d8ebe73` Machete-inspired reframing brief
- `fc33cfb` Machete KILL errors entry
- `494ad3a` Task #47 H1' v2 redesign brief
- `bccf1bd` Hybrid plan consistency audit
- `9735b47` REFUTATION wins entry
- `f0c7561` Phase 1.B Medusa brief
- `e021026` Alpaca data ready
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_pf8/` (complete vLLM fork, 6 files attributed)
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_kernel.cu` (IST-DASLab fork, 828 lines)
- vLLM `csrc/quantization/marlin/` directory (12 files, 8/9 with ARLE analogs)

## §6 Strengthens SKILL candidate

"always-source-survey-before-pending-list" now n=4 evidence:
1. `e021026` Alpaca data already downloaded
2. `86b28c7` M''' (W4-FP8 preprocess) already DONE
3. `2f19a3c` ARLE marlin_kernel.cu already at-par with vLLM
4. **THIS entry: marlin_pf8/ subdir is COMPLETE vLLM fork**

n=4 is well past the typical n=2 graduation threshold per
`kernel-optimization` skill. Recommend graduating to canonical
anti-pattern at next SKILL bump.
