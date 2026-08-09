# DFlash draft norms used the target model convention

> Status: Fixed

## Context

The corrected Qwen3.6-27B target produced a 0.334% DSpark acceptance rate on
the canonical workload. The archived pre-fix binary measured 27.59%, but its
target final norm was wrong and therefore cannot be an accuracy reference.

## Root Cause

Commit `694245eec` correctly changed the Qwen3Next target final norm to its
`(1+w)` convention and also changed seven DFlash draft norms. The DFlash model
is Qwen3 and uses plain-weight RMSNorm. SGLang's DFlash implementation uses its
plain RMSNorm, and the checkpoint norm weights are centered below one; adding
one scales the measured vectors by 1.4-1.9x.

The draft q/k norms remain stored as `w-1` because their shared CUDA prep kernel
applies `(1+w)`. The target verify final norm remains offset.

## Fix

Restore plain `rms_norm_batch` for the draft feature norm, two layer norms, and
final norm in both single-slot and batched draft paths. This is seven call
sites and one existing import; no new path or configuration is added.

Remote CUDA release build passed. Concurrent needle c=2/8/16 x3 passed 78/78
exact with zero misses. The canonical sweep completed 128/128 at every point;
acceptance was 26.90-27.81% at c=2-16, recovering from 0.334%. The accepted
baseline is recorded in
[`2026-08-10-qwen36-27b-corrected-baseline.md`](../wins/2026-08-10-qwen36-27b-corrected-baseline.md).

## Rule

Norm convention follows the checkpoint architecture at each model boundary;
shared target and draft dimensions do not imply shared norm semantics.
