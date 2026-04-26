# ARLE Examples

These examples are copyable entry points for the public surfaces.

- `curl_chat.sh`: minimal OpenAI-compatible chat request.
- `openai_chat.py`: stdlib Python client for `/v1/chat/completions`.
- `docker-compose.cuda.yml`: CUDA container serving through `arle serve`.
- `metal_local.sh`: Apple Silicon local serving smoke.
- `train_tiny_fixture.sh`: build the canonical tiny train fixture and run it.

Set `ARLE_MODEL` to avoid repeating `--model-path`.
