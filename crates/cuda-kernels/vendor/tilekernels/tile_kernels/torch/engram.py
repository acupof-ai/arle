import torch


def make_offsets(vocab_sizes: torch.Tensor) -> torch.Tensor:
    """Compute exclusive prefix-sum offsets from vocab_sizes.

    Args:
        vocab_sizes: Per-layer per-ngram embedding table sizes of shape
            (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.

    Returns:
        Offsets of shape (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_ngram_layers = vocab_sizes.shape[0]
    offsets_list = []
    for layer_idx in range(num_ngram_layers):
        flat = vocab_sizes[layer_idx].view(-1)
        prefix = torch.cat([torch.zeros(1, dtype=torch.int32, device=flat.device), flat[:-1].cumsum(0, dtype=torch.int32)])
        offsets_list.append(prefix)
    return torch.stack(offsets_list, dim=0)


def engram_hash_ref(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of engram hash.

    Args:
        ngram_token_ids: N-gram token IDs of shape (num_tokens, max_ngram_size), int32.
        multipliers: Per-layer hash multipliers of shape (num_ngram_layers, max_ngram_size), int64.
        vocab_sizes: Per-layer per-ngram embedding table sizes of shape
            (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.
        offsets: Per-layer embedding table offsets of shape
            (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.

    Returns:
        Embedding indices of shape (num_ngram_layers, num_tokens, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_ngram_layers = multipliers.shape[0]
    max_ngram_size = multipliers.shape[1]

    prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)

    ans = [[] for _ in range(num_ngram_layers)]
    hashes = prod[:, :, 0].clone()
    for i in range(1, max_ngram_size):
        hashes.bitwise_xor_(prod[:, :, i])
        for layer_idx in range(num_ngram_layers):
            ans[layer_idx].append((hashes[layer_idx].unsqueeze(-1) % vocab_sizes[layer_idx, i - 1].to(torch.int64).unsqueeze(0)).to(torch.int32))

    for layer_idx in range(num_ngram_layers):
        ans[layer_idx] = torch.cat(ans[layer_idx], dim=-1)

    output = torch.stack(ans, dim=0)
    return output + offsets.unsqueeze(1)


def engram_gate_ref(
    hidden_states: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    clamp_value: float,
    eps: float,
    save_for_backward: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation of engram gate (vectorized, supports autograd).

    Computes: output = x + sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar)) * v

    Args:
        hidden_states: Input of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        k: Key embeddings of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        v: Value embeddings of shape (num_tokens, hidden_size), bfloat16.
        weight_hidden: RMSNorm weight for hidden states, shape (hc_mult, hidden_size), bfloat16.
        weight_embed: RMSNorm weight for key embeddings, shape (hc_mult, hidden_size), bfloat16.
        clamp_value: Clamp threshold for signed-sqrt gate activation.
        eps: Epsilon for RMSNorm numerical stability.
        save_for_backward: If True, also return (dot, gate_score, rstd_x, rstd_k).

    Returns:
        If save_for_backward is False: output tensor of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        If save_for_backward is True: tuple of (output, dot, gate_score, rstd_x, rstd_k).
    """
    hidden_size = hidden_states.shape[-1]
    scalar = hidden_size**-0.5

    x = hidden_states.float()
    k_f = k.float()
    wh = weight_hidden.float().unsqueeze(0)
    we = weight_embed.float().unsqueeze(0)

    # RMSNorm
    rstd_x = torch.rsqrt(x.pow(2).mean(-1) + eps)
    rstd_k = torch.rsqrt(k_f.pow(2).mean(-1) + eps)

    # Dot -> sqrt-gate -> sigmoid
    # raw_dot is the unnormalized sum(x * wh * k * we), matching the kernel's dot_out
    raw_dot = torch.einsum('...d,...d->...', x * wh, k_f * we)
    dot = raw_dot * rstd_x * rstd_k * scalar
    signed_sqrt = dot.abs().clamp_min(clamp_value).sqrt() * dot.sign()
    gate_score = signed_sqrt.sigmoid()

    output = x + gate_score.unsqueeze(-1) * v.unsqueeze(-2)
    output = output.bfloat16()

    if save_for_backward:
        return output, raw_dot, gate_score, rstd_x, rstd_k
    return output
