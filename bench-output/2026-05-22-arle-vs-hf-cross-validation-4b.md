| Label                                | Backend | Model        | gsm8k               | mmlu                    |
| ---                                  | ---     | ---          | ---                 | ---                     |
| ARLE serve 4B                        | ?       | Qwen3___5-4B | 2.5% (5/198, inv 2) | 77.3% (116/150, inv 21) |
| HF transformers 4B                   | hf      | Qwen3___5-4B | (missing)           | 78.2% (129/165, inv 6)  |
| Δ HF transformers 4B − ARLE serve 4B |         |              | —                   | +0.85pp                 |
