# V32/GLM batched FlashMLA decode: the concurrency lever was MODEL1-only

## Context

The batched (b=N) sparse-FP8 decode lane (#60, the #1 concurrency lever — one
`sparse_decode_fwd(b=N)` per layer instead of N per-row launches) was hard-gated
to MODEL1: `config.head_dim == 512`. V32/GLM (head_dim=576 = 512 latent NoPE +
64 RoPE, d_v=512, 656 B/tok) fell through to the per-row single-decode path, so
every concurrent GLM-5.2 decode step paid N launches where MODEL1 paid one.

The single-row decode path already handled V32 correctly (`try_flashmla_decode_
attention` matches `(head_dim, rope, kv_lora)` → `(model_type_int, bytes)` and
sets `d_v = 512 if is_v32`). The batched lane just never mirrored that mapping —
it baked `DSV4_FLASHMLA_MODEL1`, `584 B/tok`, and `d_v == head_dim == 512`.

## What Worked

Mirror the single-row dim mapping in `Dsv4FlashMlaDecodeBatchScratch`:
- `match (head_dim, rope, kv_lora)` → `(model_type_int, bytes_per_token)`, same
  table as the single-row path (512→MODEL1/584, 576→V32/656).
- `d_qk = head_dim`; `d_v = 512` for V32 else `head_dim`.
- `stride_o = h_q * d_v` (was `h_q * head_dim`). For MODEL1 they coincide; for
  V32 the output latent (512) is narrower than d_qk (576), so the row pitch of
  `out_batched` / `o_accum` and the fwd's `stride_o_*` args must use d_v.
- Size `out_batched` / `o_accum` by `h_q * d_v` (new `h_q_d_v()` helper + `d_v`
  field); `q_batched` stays `h_q * head_dim` (Q is full d_qk).
- Pass `model_type_int` to both `get_meta` and `sparse_decode_fwd`.
- `kv_layout` allocates the batched scratch for both families (was `if
  head_dim == 512`).

The split-KV accum stays shared across the batch (`[num_sm_parts + b, s_q,
h_q*d_v]`, b folds into the split index via `num_splits`), exactly as MODEL1 —
only the d_v term in the h_q(*d_v) stride changes.

## Rule

When a batched/concurrency lane is added after the single-row path, port the
single-row's model-type dim mapping wholesale — don't bake the model you tested
on. The MODEL1 gate was a "works on my model" restriction, not a real kernel
limit: the same `arle_flashmla_sm90_sparse_decode_fwd` shim takes `d_qk`/`d_v`
per call (SGLang's `sparse_decode_fwd` takes `head_dim_v` for the same reason).

## Verification

pending-remote: needs a GLM-5.2 (V32) serve on the 8×H20 pod to exercise the
batched V32 path at c=8/c=16 and confirm parity vs the single-row V32 lane +
the concurrency win. MODEL1 batched decode is byte-identical (the 512 branch is
unchanged — `d_v == head_dim`, `stride_o` and buffer sizes reduce to the old
values). Build verified: BUILD_EXIT=0, 3m53s on the pod (`--release --features
cuda --bin arle`). ThinkingCap-27B (24 heads) can't reach either FlashMLA
decode lane (h_q must be 64/128), so a local A/B on it is a no-op.
