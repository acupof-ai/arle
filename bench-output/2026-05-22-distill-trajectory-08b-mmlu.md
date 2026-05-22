| Label                                  | Backend | Model                 | gsm8k                | mmlu                    |
| ---                                    | ---     | ---                   | ---                  | ---                     |
| Base 0.8B                              | ?       | Qwen3___5-0___8B-Base | 1.5% (3/194, inv 6)  | 51.4% (73/142, inv 29)  |
| After 1k distill                       | arle    | Qwen3___5-0___8B-Base | (missing)            | 47.9% (81/169, inv 2)   |
| After 2k distill (final)               | arle    | Qwen3___5-0___8B-Base | 1.6% (3/188, inv 12) | 50.0% (83/166, inv 5)   |
| Teacher 4B                             | ?       | Qwen3___5-4B          | 2.5% (5/198, inv 2)  | 77.3% (116/150, inv 21) |
| Δ After 1k distill − Base 0.8B         |         |                       | —                    | -3.48pp                 |
| Δ After 2k distill (final) − Base 0.8B |         |                       | +0.05pp              | -1.41pp                 |
| Δ Teacher 4B − Base 0.8B               |         |                       | +0.98pp              | +25.92pp                |
