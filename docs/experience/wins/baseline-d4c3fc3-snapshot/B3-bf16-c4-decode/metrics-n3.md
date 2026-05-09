# B3-bf16-c4-decode — n=3 aggregate metrics

| metric | mean | σ | σ/mean | r1 / r2 / r3 |
|---|---:|---:|---:|---|
| time_to_first_token_ms | 205.0695 | 0.4734 | 0.23% | 205.25 / 205.42 / 204.53 |
| inter_token_latency_ms | 18.3108 | 0.0140 | 0.08% | 18.33 / 18.30 / 18.30 |
| time_per_output_token_ms | 18.4049 | 0.0150 | 0.08% | 18.42 / 18.39 / 18.40 |
| output_tokens_per_second | 113.8299 | 0.7801 | 0.69% | 113.23 / 113.55 / 114.71 |
| tokens_per_second | 113.8299 | 0.7801 | 0.69% | 113.23 / 113.55 / 114.71 |

| run | successful | total | success% |
|---|---|---|---:|
| r1 | 13 | 16 | 81.2% |
| r2 | 13 | 16 | 81.2% |
| r3 | 13 | 16 | 81.2% |
