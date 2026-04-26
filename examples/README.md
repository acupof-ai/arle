# Examples

Copy-paste paths for the public surfaces. Set `ARLE_MODEL` (and `ARLE_BASE_URL`) to avoid repeating flags.

| File | What it is |
|---|---|
| [`curl_chat.sh`](curl_chat.sh) | Minimal OpenAI-compatible chat request via `curl`. |
| [`openai_chat.py`](openai_chat.py) | OpenAI Python SDK client against `/v1/chat/completions`. |
| [`docker-compose.cuda.yml`](docker-compose.cuda.yml) | CUDA serving via `docker compose up`. |
| [`metal_local.sh`](metal_local.sh) | Apple Silicon local serving smoke. |
| [`train_tiny_fixture.sh`](train_tiny_fixture.sh) | Build and run the canonical tiny train fixture. |

Server-side prerequisite for all of the above: a running `arle serve …` (or `infer …`) instance — see [Quick Start](../README.md#quick-start).
