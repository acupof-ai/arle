#!/usr/bin/env python3
"""PyTorch CUDA baseline for one ARLE OPD moderate-shape step.

This intentionally mirrors `crates/train/examples/opd_step_cpu_moderate_bench.rs`
and `crates/train/src/opd.rs::opd_step` rather than using HuggingFace
Transformers. The model topology is Qwen3.5-style full attention:
RMSNorm, gated GQA, RoPE, causal attention, SwiGLU MLP, final RMSNorm,
and an untied lm_head.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


OUT_DIR = Path(__file__).resolve().parent
ARLE_CURRENT_STEP_SECONDS = 0.83

WARMUP_RUNS = 1
MEASURED_RUNS = 3
STEPS_PER_RUN = 10

HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 1536
NUM_HIDDEN_LAYERS = 12
VOCAB_SIZE = 32_768
NUM_ATTENTION_HEADS = 8
NUM_KEY_VALUE_HEADS = 4
HEAD_DIM = 64
ROPE_CACHE_LEN = 64
RMS_NORM_EPS = 1.0e-6
PROMPT_IDS = [1, 3, 8]
ROLLOUT_LEN = 2
LR = 1.0e-3
GRAD_CLIP = 1.0
SEED = 0xB300_0D15_71A0_2026


@dataclass
class RunResult:
    run: int
    wall_seconds: float
    per_step_seconds: float
    steps_per_sec: float
    first_loss: float
    last_loss: float
    peak_memory_bytes: int


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
        return x * inv_rms * self.weight


def build_rope_cache(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    half_dim = HEAD_DIM // 2
    idx = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10_000.0 ** ((2.0 * idx) / HEAD_DIM))
    positions = torch.arange(ROPE_CACHE_LEN, device=device, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_half = cos.shape[-1]
    x0 = x[..., :rotary_half]
    x1 = x[..., rotary_half : 2 * rotary_half]
    rotated = torch.cat(((x0 * cos) - (x1 * sin), (x1 * cos) + (x0 * sin)), dim=-1)
    if 2 * rotary_half == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., 2 * rotary_half :]), dim=-1)


class Qwen35Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(HIDDEN_SIZE)
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM * 2, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KEY_VALUE_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KEY_VALUE_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)
        self.post_attention_layernorm = RMSNorm(HIDDEN_SIZE)
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = self.input_layernorm(x)
        q_full = self.q_proj(h).view(batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM * 2)
        q = q_full[..., :HEAD_DIM].transpose(1, 2)
        gate = q_full[..., HEAD_DIM:].transpose(1, 2)

        k = self.k_proj(h).view(batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(h).view(batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        kv_repeat = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
        k = k.repeat_interleave(kv_repeat, dim=1)
        v = v.repeat_interleave(kv_repeat, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn * torch.sigmoid(gate)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, HIDDEN_SIZE)
        x = x + self.o_proj(attn)

        h = self.post_attention_layernorm(x)
        mlp = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + mlp


class Qwen35Moderate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = nn.ModuleList(Qwen35Block() for _ in range(NUM_HIDDEN_LAYERS))
        self.final_norm = RMSNorm(HIDDEN_SIZE)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, input_ids: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[-1]
        cos = cos_cache[:seq_len].view(1, 1, seq_len, HEAD_DIM // 2)
        sin = sin_cache[:seq_len].view(1, 1, seq_len, HEAD_DIM // 2)
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.lm_head(self.final_norm(h))


def init_like_arle(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


def perturb_student(model: nn.Module, generator: torch.Generator) -> None:
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.rand(param.shape, device=param.device, generator=generator, dtype=param.dtype)
            param.add_((noise - 0.5) * 1.0e-3)


def build_pair(device: torch.device) -> tuple[Qwen35Moderate, Qwen35Moderate, torch.optim.AdamW]:
    torch.manual_seed(SEED)
    teacher = Qwen35Moderate().to(device)
    init_like_arle(teacher)
    student = Qwen35Moderate().to(device)
    student.load_state_dict(teacher.state_dict())
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED ^ 0xA11CE5EED)
    perturb_student(student, generator)
    for param in teacher.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0)
    return teacher, student, optimizer


def kl_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    return -(teacher_probs * student_log_probs).mean()


def opd_step(
    teacher: Qwen35Moderate,
    student: Qwen35Moderate,
    optimizer: torch.optim.AdamW,
    prompt: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
) -> float:
    rollout = prompt.clone()
    with torch.no_grad():
        for _ in range(ROLLOUT_LEN):
            logits = student(rollout, cos_cache, sin_cache)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            rollout = torch.cat((rollout, next_token), dim=-1)
        teacher_logits = teacher(rollout, cos_cache, sin_cache)

    student_logits = student(rollout, cos_cache, sin_cache)
    loss = kl_distill_loss(student_logits, teacher_logits)
    optimizer.zero_grad(set_to_none=False)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
    optimizer.step()
    return float(loss.detach().cpu())


def run_once(run: int, device: torch.device, measured: bool) -> RunResult:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    teacher, student, optimizer = build_pair(device)
    cos_cache, sin_cache = build_rope_cache(device)
    prompt = torch.tensor([PROMPT_IDS], device=device, dtype=torch.long)
    torch.cuda.synchronize(device)

    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(STEPS_PER_RUN):
        losses.append(opd_step(teacher, student, optimizer, prompt, cos_cache, sin_cache))
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    if not measured:
        del teacher, student, optimizer, cos_cache, sin_cache, prompt
        torch.cuda.empty_cache()
    return RunResult(
        run=run,
        wall_seconds=wall,
        per_step_seconds=wall / STEPS_PER_RUN,
        steps_per_sec=STEPS_PER_RUN / wall,
        first_loss=losses[0],
        last_loss=losses[-1],
        peak_memory_bytes=peak,
    )


def sigma_pct(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if mean == 0.0:
        return 0.0
    return statistics.pstdev(values) / mean * 100.0


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    # Like-for-like FP32 baseline. Do not silently route matmul through TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(
        f"env torch={torch.__version__} torch_cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(device)} total_bytes={total_bytes} free_bytes={free_bytes}"
    )
    print(
        f"config hidden={HIDDEN_SIZE} intermediate={INTERMEDIATE_SIZE} layers={NUM_HIDDEN_LAYERS} "
        f"vocab={VOCAB_SIZE} heads={NUM_ATTENTION_HEADS} kv_heads={NUM_KEY_VALUE_HEADS} "
        f"head_dim={HEAD_DIM} prompt={PROMPT_IDS} rollout_len={ROLLOUT_LEN} lr={LR} "
        f"warmup_runs={WARMUP_RUNS} measured_runs={MEASURED_RUNS} steps_per_run={STEPS_PER_RUN}"
    )

    for warmup in range(1, WARMUP_RUNS + 1):
        result = run_once(warmup, device, measured=False)
        print(
            f"warmup={warmup} wall_seconds={result.wall_seconds:.6f} "
            f"per_step_seconds={result.per_step_seconds:.6f} first_loss={result.first_loss:.9f} "
            f"last_loss={result.last_loss:.9f} peak_memory_bytes={result.peak_memory_bytes}"
        )

    measured = [run_once(run, device, measured=True) for run in range(1, MEASURED_RUNS + 1)]
    per_step = [run.per_step_seconds for run in measured]
    mean_step = statistics.fmean(per_step)
    median_step = statistics.median(per_step)
    sigma = sigma_pct(per_step)
    ratio_vs_arle = mean_step / ARLE_CURRENT_STEP_SECONDS
    speedup_vs_arle = ARLE_CURRENT_STEP_SECONDS / mean_step

    for result in measured:
        print(
            f"run={result.run} wall_seconds={result.wall_seconds:.6f} "
            f"per_step_seconds={result.per_step_seconds:.6f} steps_per_sec={result.steps_per_sec:.6f} "
            f"first_loss={result.first_loss:.9f} last_loss={result.last_loss:.9f} "
            f"peak_memory_bytes={result.peak_memory_bytes}"
        )
    print(
        f"summary mean_step_seconds={mean_step:.6f} median_step_seconds={median_step:.6f} "
        f"sigma_pct={sigma:.3f} ratio_vs_arle_0p83={ratio_vs_arle:.4f} "
        f"speedup_vs_arle_0p83={speedup_vs_arle:.4f}"
    )

    report = {
        "env": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "total_bytes": total_bytes,
            "free_bytes": free_bytes,
            "tf32": False,
        },
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "vocab_size": VOCAB_SIZE,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_key_value_heads": NUM_KEY_VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "prompt_ids": PROMPT_IDS,
            "rollout_len": ROLLOUT_LEN,
            "lr": LR,
            "grad_clip": GRAD_CLIP,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "steps_per_run": STEPS_PER_RUN,
        },
        "runs": [asdict(run) for run in measured],
        "summary": {
            "mean_step_seconds": mean_step,
            "median_step_seconds": median_step,
            "sigma_pct": sigma,
            "arle_current_step_seconds": ARLE_CURRENT_STEP_SECONDS,
            "ratio_vs_arle_0p83": ratio_vs_arle,
            "speedup_vs_arle_0p83": speedup_vs_arle,
        },
    }
    (OUT_DIR / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
