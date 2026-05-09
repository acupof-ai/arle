# B3-bf16-c4-decode — n=3 metrics (统一目标 N≈20)

| metric | mean | σ | σ/mean | r1 / r2 / r3 |
|---|---:|---:|---:|---|
| time_to_first_token_ms | 204.9994 | 0.0290 | 0.01% | 205.03 / 205.00 / 204.97 |
| inter_token_latency_ms | 18.3010 | 0.0040 | 0.02% | 18.30 / 18.31 / 18.30 |
| time_per_output_token_ms | 18.3960 | 0.0036 | 0.02% | 18.39 / 18.40 / 18.39 |
| output_tokens_per_second | 113.7558 | 0.3105 | 0.27% | 113.40 / 113.97 / 113.90 |
| tokens_per_second | 113.7578 | 0.3096 | 0.27% | 113.40 / 113.97 / 113.90 |

| run | successful | errored | total | success% |
|---|---|---|---|---:|
| r1 | 17 | 0 | 20 | 85.0% |
| r2 | 17 | 0 | 20 | 85.0% |
| r3 | 17 | 0 | 20 | 85.0% |
