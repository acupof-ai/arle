# Progress Snapshot — 2026-05-01 evening

Single-day status of the world-#1 mission and parallel product-line prep.
Frozen at commit `dc45660d` (Phase 2.B.2 landed) with `P2.B.3 forward sparse path` in flight.

## Mission grid (per leadership doc §2.4)

| | W1 max-throughput (32k×c=4) | W2 long-decode (32k+2048×c=4) |
|---|---|---|
| **H1 (L4)** | ✅ **SECURED 1.609× SGLang** (range 1.469-1.678×, mean 26.169 out tok/s, 3-run reproducibility) cite `2026-05-01-phase1-close-evictable.md` | ⏳ pending Phase 2.B sparse-KV real lift |
| **H2 (H20/H100)** | ⏳ deploy bundle ready (`f4a5bc1c`), waiting on user-provided H20 SSH/path | ⏳ waiting on multi-GPU + spec lift |

**1/4 grid SECURED**. Phase 1 entrance gate (≥0.95×) AND world-#1 margin (≥1.30×) both passed on (W1, H1).

## Today's commit chain (chronological, all on `main` branch)

### Phase 1 close path

- `5aff05f7` feat(scheduler): plumb spec decode counters and no-op config
- `21bd44d2` feat(scheduler): wire spec decode verifier micro-batch path
- `5eddaab8` feat(scheduler): adaptive spec acceptance rate threshold
- `0cc41f6f` fix(scheduler): correct verifier bit-identity (CUDA Graph + force_eager)
- `b2de936c` ★ **docs(longctx): close phase 1 on c4 mission criterion + c1 parallel track**

### Phase 2 multi-token spec (plumbing landed, -63% regression honest)

- `a1106f15` fix(scheduler): reject fake multi-token spec canary
- `1d7e8187` fix(scheduler): reject unwired external draft path
- `2309dad5` docs(scheduler): warn against plain SelfSpec without MagicDec sparse-KV
- `77946bf8` feat(scheduler): real multi-token speculative decode with external draft model
  + errors entry `2026-05-01-phase2-real-spec-regression.md` records `9.73 tok/s -62.8%` and `12% acceptance` root causes (Qwen3-0.6B draft not cheap, KV pool 136K→121K, sequential verifier overhead)

### Multi-GPU F0-F4 scaffold (TP/PP/EP axes)

- `fa624daa` feat(cuda): add nccl group coordinator smoke behind nccl feature
- `f0a61418` docs(projects): multi-gpu f0 readiness assessment
- `90ceaa22` feat(distributed): F1 parallel state + tp weight loading
- `e8e1cdd6` feat(distributed): LayerCommunicator skeleton
- `f4a5bc1c` chore(scripts): h20 single-node deploy bundle (sm_90 cubin + smoke_h20_phase1.sh)
- `047b3b08` feat(scheduler): F0.7 ForwardBatch + IntermediateTensors type
- `b0826108` docs(environment): F0.11 multi-rank env vars
- `aa692748` ★ **feat(model): qwen3 + qwen35 TP forward sharding**
- `57ae6def` feat(distributed): F3 pipeline parallel scaffold
- `01fe70c1` feat(distributed): F4 expert parallel scaffold

### DeepSeek V4 product line prep

- `4e849b02` docs(projects): deepseek v4 readiness assessment (DS0-DS8 gap matrix)
- `1e53b20e` feat(deepseek-spec): DS0 scaffold crate (config + tensor names + Shard annotations)
- `fc14ee52` feat(deepseek-spec): DS2 MoE forward type scaffold
- `ddbbc5ad` docs(projects): MLA kernel design for DeepSeek path
- `c29dad08` feat(cuda-kernels): DS3 MLA decode kernel skeleton

### DevOps / infra

- `39d760cc` feat(docker): dev image with CUDA + Rust + Zig + Python deps
- `297af460` docs(meta): sync canonical docs with multi-GPU + DeepSeek V4 progress
- `2cd30891` docs(readme): fix version references and improve clarity

### Phase 2.B MagicDec sparse-KV (W2 grid real lever, in flight)

- `8dfa25f6` docs(projects): phase 2.B magicdec sparse-KV design (B.1-B.6 work slices)
- `59a30dcb` feat(scheduler): P2.B.1 sparse-KV view interface for spec decode draft (3-round codex review + Explore cross-check clean)
- `dc45660d` feat(scheduler): P2.B.2 RadixCache page drop for sparse-KV draft view (3-round + Beauvoir cross-check clean)
- ⏳ **P2.B.3 forward sparse path** (in flight — sparse view真接入 forward；Phase 2.B 真生效 hot path)
- (next) P2.B.4 spec_path 集成 / P2.B.5 单测 / P2.B.6 longctx-32k c=4 bench 验 ≥26.169 + acceptance 0.4-0.7

## Outstanding blockers

| Blocker | Resolution path |
|---|---|
| H20 SSH/path 用户未提供 | 等用户给 host/repo path/model path/GPU 数量；deploy bundle (`f4a5bc1c`) 一拿就跑 |
| Phase 2 real spec lift unmet (-63% regression) | P2.B.3-B.6 sparse-KV path 真生效 — 当前进行中 |
| Phase 1 c=1 0.85× gap (parallel track) | 不阻 mission；单流 decode kernel 优化作 follow-up |

## Process learnings logged this round

- `feedback_codex_tmux_double_enter.md` — 多行 directive 必须 Enter Enter 才提交
- AGENTS.md scheduler 加: Plain SelfSpec K>1 = K+1 forwards 同模型 = no speedup（防未来 agent 再踩）
- AGENTS.md scheduler 加: acceptance_rate=100% canary 假象不算 win，看 effective throughput vs Phase 1 baseline
- Cron prompt 加: 代码质量审查 (整洁/统一/优雅) + git diff 抽查抓违规
- Cron prompt 加: ≥3 轮 codex review + Explore agent cross-check（mission-critical patches）

## Next actions (if mission goes uninterrupted)

1. P2.B.3 forward sparse path commit + cross-check
2. P2.B.4 spec_path 接 sparse draft (replace dead `_sparse_views` placeholder)
3. P2.B.5 unit tests (verifier full-KV bit-ident with sparse draft)
4. P2.B.6 longctx-32k c=4 300s bench — 真 lift 验证或证伪
5. Once H20 SSH lands: scp `arle-h20-phase1-<sha>.tar.gz` + run `smoke_h20_phase1.sh` → success(W1, H2) candidate data
6. Phase 3 disaggregated prefill/decode (Mooncake-aligned) on multi-GPU H20 / 2× L4

Mission progress: from "Phase 1 catch-up failing 0.61×" at start of day to "Phase 1 SECURED 1.609× + Phase 2.B real-lift path on track + multi-GPU F0-F4 scaffold + DeepSeek V4 readiness" by evening. ~28 commits in single session, all under 3-round review + cross-check rigor.
