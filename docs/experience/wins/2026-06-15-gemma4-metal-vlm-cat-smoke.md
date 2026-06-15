# Gemma4 Metal VLM Cat Smoke

## SLO-shape probed?

N. This was a functional VLM correctness smoke, not a guidellm throughput run.
The route under test is `/v1/chat/completions` with image content parts;
guidellm only exercises text `/v1/completions` in the current wrapper.
No default/performance claim is made from this entry.

## Roofline check

Deferred. This change wires the Gemma4 image soft-token path; no Metal capture
or MLX Instruments run was taken. Perf/SLO validation remains a separate task.

## Goal

Functional correctness: prove a Gemma4 4bit Metal VLM checkpoint can consume a
real image and return an image-grounded answer instead of the previous 400.

## Hypothesis

If the HTTP layer expands `<|image|>` into BOI + soft-token image ids + EOI and
the Metal backend replaces those rows with vision-tower embeddings, the
Wikimedia cat image request should identify a cat.

## Command

```bash
./target/release/arle serve \
  --backend metal \
  --model-path /tmp/arle-gemma4-e2b-it-4bit \
  --port 8018 \
  --low-impact \
  --max-total-tokens 4096 \
  --max-prompt-tokens 2048 \
  --total-pages 512

curl -L --fail --silent --show-error \
  https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg \
  -o /tmp/arle-gemma4-test-cat.jpg

IMG_B64=$(base64 -i /tmp/arle-gemma4-test-cat.jpg | tr -d '\n')
jq -n --arg img "data:image/jpeg;base64,$IMG_B64" \
  '{model:"arle-gemma4-e2b-it-4bit", max_tokens:32, messages:[{role:"user", content:[{type:"text", text:"What animal is in this image? Answer in one short sentence."},{type:"image_url", image_url:{url:$img}}]}]}' \
  > /tmp/arle-gemma4-cat-request.json

curl -sS --max-time 180 \
  http://127.0.0.1:8018/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/arle-gemma4-cat-request.json
```

## Environment

- Backend: Metal
- Model: `/tmp/arle-gemma4-e2b-it-4bit`
- Hardware: Apple M4 Pro MacBook Pro, 20-core GPU, 48 GB unified memory, Metal 4
- OS: macOS 26.3.1 build 25D771280a
- Commit before diff: `857e7ff5`
- Feature set: `cargo build --release --no-default-features --features metal,no-cuda`
- Non-default flags: `--low-impact --max-total-tokens 4096 --max-prompt-tokens 2048 --total-pages 512`

## Results

Image input:

```text
/tmp/arle-gemma4-test-cat.jpg: JPEG image data, 1600x1598, components 3
279603 /tmp/arle-gemma4-test-cat.jpg
```

VLM response:

```json
{
  "model": "arle-gemma4-e2b-it-4bit",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The image shows a **cat**."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 280,
    "completion_tokens": 8,
    "total_tokens": 288
  }
}
```

Text regression smoke:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hello"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 2,
    "total_tokens": 17
  }
}
```

## Problems

- No canonical guidellm number: current guidellm wrapper cannot send image
  content parts, so this entry is correctness-only.
- Multi-image requests are intentionally not licensed by this smoke; the
  landed backend path supports one image per request and errors clearly for
  more.

## Learnings

- Gemma4 VLM needs prompt expansion plus embedding replacement: accepting
  image parts at schema level is insufficient unless the backend receives the
  exact soft-token count and scatters vision embeddings into those token rows.
- PLE inputs must mask image token positions back to token id 0; otherwise the
  per-layer embedding stream treats image placeholders as text tokens.

## Delta vs baseline

- Baseline behavior: image content returned HTTP 400 with
  `VLM soft-token embeddings are not wired yet`.
- Now: same image route returns HTTP 200 and identifies the image as a cat.

## Artefacts

- Image: `/tmp/arle-gemma4-test-cat.jpg`
- Request JSON: `/tmp/arle-gemma4-cat-request.json`
