# Marlin's fp32-reduce buffer was sized for one block per SM — the prefill crash, 2026-08-20

> Status: Confirmed and fixed. Root cause of the crash
> [`2026-08-19-blocks-per-sm-search-two-latent-bugs.md`](2026-08-19-blocks-per-sm-search-two-latent-bugs.md)
> left open, and the reason pinning `MARLIN_MAX_BLOCKS_PER_SM` to 1 worked.

## Phenomenon

`arle serve` on Qwen3.8-27B-NVFP4 answered a 4-token probe, then died on the
next request — a 531-token prompt — with

```
ERROR infer-server engine step failed:
  DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")
```

Every synthetic-prompt bench passed at the same time (`SERVER_ERRORS=0` across a
c=1..16 grid), and the 33K long-agent run had been crashing for two days.

## Root cause

`marlin_template.h` picks the fp32 reduce slot as

```cpp
int c_cur_offset = locks_off * c_size;   // locks_off <= gridDim.x - 1
```

and `gridDim.x` is `sms * blocks_per_sm`. `marlin_gemm.cu` sized the buffer for
`sms` slots:

```cpp
return sms * max_m_block * device::marlin::max_thread_n;   // no blocks_per_sm
```

Every block above the first on each SM wrote past the end. Same defect class as
the lock buffer, which was grown to `sms * MARLIN_MAX_BLOCKS_PER_SM * 4` in the
earlier fix; `c_tmp` was missed in the same pass.

Two conditions had to coincide, which is why it looked like a long-context bug:

- **`blocks_per_sm >= 2`.** Reachable only after the blocks-per-SM search
  landed. Pinning to 1 makes `locks_off < sms` and the overflow disappears —
  that is why the pin "worked", not the partial-prefix-restore theory it was
  filed under.
- **`thread_m_blocks == 4`,** i.e. prefill. At decode `thread_m_blocks == 1`
  makes `c_size` a quarter of the prefill slot, so the `sms`-sized buffer
  happens to cover `4 * sms` of them and nothing overflows.

A prompt only has to be long enough to leave the decode routes. 531 tokens is
enough; 8-token synthetic prompts never are, so every decode bench passed.

## Fix

One line, `marlin_gemm.cu`:

```cpp
return sms * MARLIN_MAX_BLOCKS_PER_SM * max_m_block * device::marlin::max_thread_n;
```

20 MB more scratch. After it: needle ladder 3/3 exact at 512 / 4096 / 16384 /
32768, `SERVER_ERRORS=0`, and the 32K long-agent workload completes 32/32 on
both arms.

## Rule

When a kernel's grid grows, every buffer indexed by `blockIdx` grows with it.
The lock buffer and the reduce buffer are indexed by the same `locks_off`; one
was fixed and the other was not, and the survivor hid for two days behind a
decode tile that happened to be a quarter of the prefill one.

Corollary for the earlier entry: "pinning is not established as necessary" was
right that the mechanism was unproven, and wrong to conclude the pin was
therefore unnecessary. An unexplained workaround that works is evidence about
where to look, not evidence that there is nothing to find.
