# B1-bf16-c1-4k — n=3 metrics (统一目标 N≈20)

| metric | mean | σ | σ/mean | r1 / r2 / r3 |
|---|---:|---:|---:|---|
| time_to_first_token_ms | 522.9926 | 3.9698 | 0.76% | 521.89 / 527.40 / 519.69 |
| inter_token_latency_ms | 22.8055 | 0.0053 | 0.02% | 22.81 / 22.81 / 22.80 |
| time_per_output_token_ms | 24.7570 | 0.0214 | 0.09% | 24.76 / 24.78 / 24.73 |
| output_tokens_per_second | 43.8482 | 0.0105 | 0.02% | 43.85 / 43.84 / 43.86 |
| tokens_per_second | 43.8490 | 0.0101 | 0.02% | 43.85 / 43.84 / 43.86 |

| run | successful | errored | total | success% |
|---|---|---|---|---:|
| r1 | 19 | 0 | 19 | 100.0% |
| r2 | 19 | 0 | 19 | 100.0% |
| r3 | 19 | 0 | 19 | 100.0% |
