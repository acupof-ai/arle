# B2-bf16-c4-4k — n=3 aggregate metrics

| metric | mean | σ | σ/mean | r1 / r2 / r3 |
|---|---:|---:|---:|---|
| time_to_first_token_ms | 2012.1843 | 0.6031 | 0.03% | 2012.37 / 2012.67 / 2011.51 |
| inter_token_latency_ms | 25.4551 | 0.0176 | 0.07% | 25.43 / 25.46 / 25.47 |
| time_per_output_token_ms | 33.2206 | 0.0168 | 0.05% | 33.21 / 33.22 / 33.24 |
| output_tokens_per_second | 79.0394 | 0.0830 | 0.10% | 78.96 / 79.13 / 79.02 |
| tokens_per_second | 97.5532 | 24.3411 | 24.95% | 81.68 / 125.58 / 85.40 |

| run | successful | total | success% |
|---|---|---|---:|
| r1 | 52 | 56 | 92.9% |
| r2 | 52 | 56 | 92.9% |
| r3 | 51 | 55 | 92.7% |
