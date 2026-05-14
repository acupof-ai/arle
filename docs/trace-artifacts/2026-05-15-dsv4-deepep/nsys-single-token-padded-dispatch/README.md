# DSv4 Padded Dispatch Negative Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run enables fixed-top-k B=1 padded dispatch with
`ARLE_DSV4_PADDED_DISPATCH=1` but still leaves the old send-rank zero/count
kernel in the decode path.

```text
prompt: 用两个字形容彩虹。
output: 霓彩
```

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 136.908 ms |
| Per-rank decode range p50 | 136.676 ms |
| Request wall time | 1.259 s |
| Returned text | `霓彩` |

Compared with [`../nsys-single-token-allgather-counts/`](../nsys-single-token-allgather-counts/),
the 256-byte all-rank count D2H is gone, but the path regresses because every
rank receives padded route rows and the old send-count kernel still runs.
Decode-only D2H becomes 344 128-byte local-count reads, and launch/memset work
increases.

This trace is intentionally retained as a negative optimization record. The
fixed version is
[`../nsys-single-token-padded-dispatch-skip-count/`](../nsys-single-token-padded-dispatch-skip-count/).

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
