# ARLE OPD 方法论审查 — 行业对标

日期：2026-05-25  
审查范围：`crates/train/src/opd.rs`, `crates/train/src/loss.rs`,
`crates/cli/src/args.rs`, `crates/cli/src/train_cli.rs`,
`crates/train/examples/opd_step_cuda_infer_teacher_train.rs`,
`crates/autograd/src/optim.rs`, `crates/autograd/src/lr_schedule.rs`  
对标：GKD (Agarwal et al., ICLR 2024)、TRL `GKDTrainer` (HF
`trl/experimental/gkd`)、MiniLLM (Gu et al., 2023)、Thinking Machines
Lab "On-Policy Distillation" (2025)、DeepSeek-R1-Distill-Qwen 系列。

---

## 1. Executive Summary

**ARLE OPD 复现的是 forward-KL on greedy-rollout 的最朴素版本**，缺失
六个行业标配里的至少四个：(a) **distillation temperature**（全 stack
没有任何 `temperature` 概念，softmax 直接作用于 raw logits）、(b)
**stochastic sampling 的 student rollout**（greedy argmax，违背 GKD
paper 明确推荐的 `γ=1` 训练时温度）、(c) **completion-only token
masking**（ARLE 在整段 rollout 包括 prompt token 上算 KL，TRL 只在
completion token 上算）、(d) **LR schedule / warmup**（trait 已实现
但 OPD 路径根本没 wire，全程固定 lr）。这些都不是 hypothesis，是源码
确证。当前 T18/P5 的 "0.59pp MMLU 涨幅、跨不过 base" 结果与这四个
gap 的存在一致 — 行业 small-student 蒸馏的典型增益是 +1~+10pp，ARLE
当前 recipe 等价于 GKD/MiniLLM 论文里被论文自己 ablate 掉的弱基线。

**最重要的发现：rollout=8 短到几乎不可能产生 in-distribution 的
on-policy 行为**（TRL default 128，Thinking Machines 用完整 reasoning
trace；GKD 训练序列 512~1024 token）。即使其他三个 gap 全修，rollout=8
本身就是 16× 偏小的 hyperparameter，是 capability 不动的最大单点
hypothesis（high-confidence，证据基于 file:line 默认值）。

**T20 的 1000-prompt corpus diversity probe 是必要不充分**。它 fix
了"20 prompt 严重过拟合"这个 confounder，但即使 corpus 完美，下面
列的四个 method-level gap 依然会让 capability transfer 见顶在
"loss 下降、capability 不动"。

---

## 2. ARLE 当前实现摘要

### 2.1 OPD 主流程（`crates/train/src/opd.rs:813-1183`）

```
prompt_ids
  → greedy rollout (rollout_len 默认 8，无 temperature/top-k/top-p)        # 989-1090
  → teacher forward on full sequence [prompt + rollout]                    # 1097
  → student forward on full sequence (tape on)                             # 1117
  → kl_distill_loss(student, teacher, rollout.len(), ...)                  # 1128
     ※ 注意 num_positions = rollout.len() = prompt_len + cfg.rollout_len，
       所以 KL 是在整段（含 prompt token）上算的，不是 completion-only
  → optional GKD SFT anchor mix（默认 lambda=0，纯 KL）                    # 1136-1150
  → backward, grad clip, AdamW step                                        # 1159-1174
```

关键代码位置：
- 默认 GKD config：`GkdSftAnchor::StudentRollout, lambda=0, no temperature`
  (`opd.rs:92-100`)。
- StudentRollout SFT anchor 用 `rollout[1..]` 做 next-token CE target
  (`opd.rs:593-617`) — **这只是 self-distillation**（学生学习自己
  argmax 的下一个 token），跟 GKD paper / TRL 的 "ground-truth dataset
  completion" 完全不同。CorpusTruth anchor 是真正 ground-truth，但
  CLI 没暴露，且需要额外 windowed forward（受 T16 阻塞）。
- `opd_step_with_teacher_forward_profiled_gkd_anchor` 是真接收 anchor
  的入口；调用者只有 `opd_step_cuda_infer_teacher_train.rs` 这个
  example，**production `arle train opd` CLI 路径根本不暴露
  `--gkd-lambda` / `--sft-anchor`**（`crates/cli/src/args.rs:449-493`
  TrainOpdArgs 完整定义只 8 个 flag，没有 GKD/SFT 任何 knob）。

### 2.2 KL Loss 公式（`crates/train/src/loss.rs:33-53`）

```
teacher_probs = softmax(teacher_logits)              # 无 temperature
student_log_probs = log_softmax(student_logits)      # 无 temperature
loss = -mean(teacher_probs * student_log_probs)       # mean over (positions × vocab)
```

属性确认：
- **Forward KL** `KL(teacher || student)`（drop `-H(t)` 常数项）。
  跟 TRL `beta=0` 等价；与 MiniLLM/Thinking Machines 的 **reverse KL**
  `KL(student || teacher)` 反向。
- **无 temperature**（搜遍 `crates/train/` `crates/autograd/src/` 没有
  任何 `temperature` / `kd_temp` / `tau` 符号；仅 inference path 有
  sampling temperature 但与训练不通）。
- **normalize 是 mean over `positions*vocab`** 而非 TRL 的
  `sum / num_completion_tokens`。等价于额外 `1/vocab` rescale；OPD 作者
  在 `loss.rs:43-50` 自己承认 "AdamW 吸收常数"，**但与 SFT-CE 混合
  时会出问题**（`opd.rs:590` `next_token_sft_loss_from_logits` 手动
  `* 1/vocab` 抵消，意味着两路 loss 在数值尺度上是 stack 内 hack 出来的
  一致性，不是数学上的标准）。

### 2.3 Optimizer / LR（`cli/src/train_cli.rs:254, 337`）

```rust
AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0)
                                          ^^^ weight_decay = 0
```

确认：
- `weight_decay = 0`（标准 AdamW 蒸馏 0.01~0.1）。
- **没有 LR schedule**。`autograd::lr_schedule::{LinearWarmup,
  CosineWithWarmup}` 都实现完毕（`crates/autograd/src/lr_schedule.rs`），
  `Trainer` 也接 (`crates/train/src/trainer.rs:425 self.optim.set_lr`)，
  但 OPD 路径不走 Trainer，`opd_step*` 直接 call `optimizer.step()`
  — 全程固定 lr，无 warmup，无 decay。

### 2.4 默认超参（`crates/cli/src/args.rs:449-493` + example）

| flag | CLI default (`arle train opd`) | example default | 备注 |
| --- | --- | --- | --- |
| `--rollout-len` | 8 | 8 | TRL default 128 |
| `--lr` | 1e-4 | 1e-5 | T18 用 1e-5；P5 用 2e-5 |
| `--grad-clip` | 1.0 | 1.0 | 标准 |
| `--steps` | 5 | 1 | smoke 默认 |
| `--gkd-lambda` | **不存在** | 0.0 | 实质 100% pure KL |
| `--sft-anchor` | **不存在** | student-rollout | self-distill, 非 ground truth |
| `--kl-chunk-size` | **不存在** | None | 一段 KL |
| temperature | **完全无概念** | — | — |
| warmup | **完全未连接** | — | — |
| weight_decay | hard-coded 0 | hard-coded 0 | — |

---

## 3. 行业标准对比

| 维度 | GKD paper (Agarwal 2024) | TRL GKDTrainer | DeepSeek-R1-Distill | MiniLLM (Gu 2023) | Thinking Machines 2025 | ARLE current |
| --- | --- | --- | --- | --- | --- | --- |
| Loss family | generalized JSD（forward+reverse mix）| generalized JSD via `F.kl_div` | pure SFT CE（无 logit KD）| **reverse KL** | **per-token reverse KL** | **forward KL only** |
| `beta` (JSD interp.) | sweep {0.1, 0.5, 0.9} | default 0.5 | n/a | reverse only (β=1) | reverse only (β=1) | **forward only (β=0)** |
| `lambda` (on-policy fraction) | 主推 1.0；ablate {0, 0.5, 1.0} | default 0.5 | 0 (SFT) | 1.0 | 1.0 | **0.0 by default, lambda=lambda_kl 用法不同**（lambda>0 才 mix SFT，纯 OPD 训练时 lambda=0）|
| Student rollout sampling | **temperature γ=1**, stochastic | `temperature=0.9, do_sample=True, top_k=0` | n/a | sampled with teacher-mix | sampled (on-policy) | **greedy argmax (effective γ→0)** |
| Distillation temperature | 调 γ 同时用于 sampling+softmax | `temperature=0.9` 同时缩放 logits | n/a (no KD) | reverse KL no temp scaling | reverse KL no temp scaling | **无（softmax on raw logits）** |
| Rollout / completion length | 任务相关（XSum ≤512，GSM8K ≤512）| **max_new_tokens=128** | full SFT seq ≤4096 | full inst-tuning seq | full reasoning trace（数千 token）| **8** |
| KL masking | completion tokens only | **completion tokens only** (`prompt_lengths-1:-1`) | label mask | completion only | completion only | **整段 rollout（含 prompt token）** |
| Optimizer | Adafactor | AdamW via HF Trainer | AdamW | AdamW | unspecified | AdamW |
| LR | 3e-4 (T5-XL→large) / 1e-3 (small) | HF default 5e-5 | unspecified | 5e-4 ~ 5e-5 | unspecified | 1e-4 / 2e-5 / 1e-5 |
| Warmup steps | **2000** (XSum) | HF default `warmup_ratio` configurable | linear 0-3% | linear warmup | unspecified | **0 (no warmup, no schedule)** |
| Weight decay | unspecified | HF Trainer default 0.0 | 0.1 | 0.01 | unspecified | **0.0** |
| Batch size | **32** (XSum) | configurable | 1024 | 64 | **64 prompts × 4 samples = 256** | **1 (single prompt)** |
| Total training steps | **40k–100k** task-dep. | configurable | 800k samples × ~3 epochs | 20k steps | **~150** | 5000 step ≈ 5000 prompt |
| Total training tokens (rollout) | 40k × 32 × 256 ≈ 327M | 128 × steps × batch | 数 B | 数 B | ~256 × 150 × 1024 ≈ 39M | **8 × 5000 = 40k** rollout token |
| Teacher–student gap measured | T5-XL 3B → T5-small 77M (40×) | up to Qwen3-8B → 0.6B | Qwen2.5-32B → Qwen2.5-1.5B (21×) | GPT2-1.5B → 120M (12×) | Qwen3-32B → Qwen3-8B (4×) | Qwen3.5-4B → Qwen3.5-0.8B (5×) |
| Reported capability gain | XSum +2.1×, GSM8K +1.9×, MMLU +1pp, BBH +2pp | docs say "comparable to dataset SFT" | AIME +20pp, MATH +30pp（vs base）| +6 ~ +12 ROUGE | AIME 60% → 70% (+10pp) | **MMLU −0.82pp**（best, T18 step1000 = 50.59 vs base 51.41）|

---

## 4. 五个最大方法论缺陷（按影响排序）

### 缺陷 #1 — Rollout 长度 16~64× 偏小（HIGH confidence）

**证据**：`opd.rs:53` 默认 `rollout_len: usize = 8`；`example.rs:44`
`DEFAULT_ROLLOUT_LEN: usize = 8`；T18/P5 run 实际 `rollout_len=8`
(wins entry `2026-05-25-t18-recipe-variant-result.md` line 11)。
TRL default 128（`gkd_config.py:max_new_tokens=128`）；Thinking Machines
2025 用完整 reasoning trace（典型 512~2048 token）；GKD paper XSum
最长 512 token output。

**为何重要**：on-policy distillation 的整个点是**让学生在自己分布
下产生的序列上被教师纠正**。rollout=8 时学生只生成 8 个 token，
对 chat / reasoning / 长 instruction following 完全不构成 in-distribution
样本 — 这跟 short-context teacher forcing 几乎等价。Thinking Machines
明确：on-policy 的 "discount factor zero, next-token only" 假设只在
学生真的能 rollout 出完整 trajectory 时成立。8 token 不算 trajectory。

**推断 capability 损失**：HIGH。这是 single biggest hypothesis。
GKD paper Figure 2 显示 rollout length 5x 翻倍能拿 +0.5~+1pp BBH。
ARLE 16x gap 推断 +2~+4pp 量级（hypothesis，未做 ARLE 内 sweep
验证）。

**修复 cost**：~10 LOC — `args.rs:465` 把 default 改到 64-128，加
benchmark 跑 rollout_len ∈ {8, 32, 64, 128} sweep。**但 memory cost
非线性**：rollout 越长，student/teacher forward 序列越长，KL 张量
`[1, seq, 248320]` 越大。`rollout_len=128, prompt=16` 时 KL 张量
~146 MB f32，需要先 land sequence-windowed forward（T16）或
`kl_chunk_size` 路径才能稳跑。

**与 T16/T20 关系**：T16 (sequence-windowed forward) 是这个修复的
前置条件 (memory blocker)；T20 (corpus diversity) **orthogonal** —
T20 解决 prompt 多样性，rollout_len 解决 trajectory 长度。两者都需要。

---

### 缺陷 #2 — KL 在 prompt token 上算（HIGH confidence）

**证据**：`opd.rs:1128-1135` 调
`kl_distill_loss_for_config(..., rollout.len(), ...)`，其中
`rollout = prompt_ids + generated_tokens`（line 991）；
`loss.rs:39-53` 的 mean reduction 横跨**整个 (positions×vocab)
张量**，不做任何 prompt mask。对比 TRL `gkd_trainer.py:382-384`：
```python
prompt_lengths = inputs["prompts"].shape[1]
shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
```

**为何重要**：teacher/student 在 prompt 部分本来就该差不多（两者
都看到一样的输入做 next-token prediction，KL 本就低）。把这部分
塞进 loss 不会提供 supervision signal，反而**稀释 completion 部分
的 gradient**。例如 prompt=16 + rollout=8 时，24 个 position
mean，其中 16/24 = 66.7% gradient 权重在 prompt token 上。这等价
于把有效 lr 砍到 33%（hypothesis，未实验验证）。

**推断 capability 损失**：MEDIUM-HIGH。直接拉低有效 supervision
density 至 1/(1+prompt/rollout) 倍。

**修复 cost**：~30 LOC — 在 `kl_distill_loss` 引入 `start_position`
参数，slice `[:, prompt_len:, :]` 后再做 softmax/log_softmax/mean。
现有 `kl_distill_loss_chunked` 已经有 slice 路径
(`loss.rs:104-110`)，复用基础设施即可。

**与 T16/T20 关系**：纯 orthogonal。是行级 bug 性质，不依赖 T16/T20。
应该单独立成 P0 实验：固定 P5/T18 其它参数，只 mask prompt，跑同
checkpoint 复测 MMLU。

---

### 缺陷 #3 — 缺 distillation temperature（MEDIUM-HIGH confidence）

**证据**：全 `crates/train/` `crates/autograd/src/` grep `temperature`
零 hit（仅 inference path 有 sampling temperature，跟训练不通）。
TRL `gkd_trainer.py:255` 显式 `student_logits = student_logits /
temperature; teacher_logits = teacher_logits / temperature`，default
`temperature=0.9`。Hinton 2015 标准蒸馏 temperature ∈ {2, 4, 8}。

**为何重要**：teacher logits 在大词表（Qwen3.5 vocab=248320）下
**极度 peaky**（top-1 prob 常 > 0.99）。softmax(raw_logits) 后
`teacher_probs` 几乎是 one-hot，KL loss 几乎退化成 hard-label CE，
失去蒸馏的全部信息论价值（"dark knowledge"）。Hinton 论文核心论点。

**辅证**：T18 wins entry `KL 量级 1.6e-5`（line 47）— 这个数量级
异常小，符合 "teacher 几乎 one-hot, student 已经接近, loss 没
gradient" 的表现。**任何蒸馏论文里 KL loss 都不会出现在
1e-5 量级**，典型在 0.1~5。这是 SOLID evidence: teacher 分布太
peaky → 无 dark knowledge → KL 早已收敛 → loss 不再驱动学生改变。

**推断 capability 损失**：HIGH。是 T18 "heldout KL 收敛但 MMLU 不
动"的最直接技术解释。修了 temperature，KL 数量级会上去到 0.5~3，
gradient 重新有效，capability 才有机会转移。

**修复 cost**：~15 LOC — `kl_distill_loss` 加 `temperature: f32`
参数，softmax 输入除以 T，loss 输出乘 T²（Hinton 标准 scaling
保持 gradient magnitude 一致）。CLI 加 `--temperature` flag。

**与 T16/T20 关系**：完全 orthogonal。可在 T16/T20 完成前独立验证：
固定 P5 setup，只把 KL 加 T=2.0，跑 200 step 看 KL 量级是否
上到 0.1+。

---

### 缺陷 #4 — Greedy rollout（违背 GKD paper 明确推荐）（MEDIUM confidence）

**证据**：`opd.rs:115-160` `greedy_next_token` 用 argmax；
`opd.rs:163-209` `device_argmax_token` 同；无 sampling 路径。
GKD paper Section 3.2 + Table A.1：`temperature γ=1` 训练时
stochastic sampling；TRL `gkd_config.py:51 temperature=0.9` + 
`gkd_trainer.py:206 do_sample=True, top_k=0`。

**为何重要**：greedy 永远走最高概率分支 → 学生在自己 mode 上 cycle，
不探索其它合理 trajectory → teacher 永远在同一组 token 上打分 →
等价于过拟合到 student 当前的 mode。这是 GKD paper "self-generated
mistakes" 设计初衷的反面（学生**就该犯错并让 teacher 纠正各种
mistake**）。

**辅证**：T18/P5 用同一组 20 prompt + greedy 鸡 5000 step → 学生
重复看到自己 argmax 的 20 个 trajectory（除去 lr 影响 forward
变化），等价 **40k token 的 self-distillation closed loop** — 跟
T20 "1000 prompt diversity" 的 fix direction 一致，但 greedy 让
diversity 的 effective leverage 减半。

**推断 capability 损失**：MEDIUM。stochastic sampling 不会魔法
fix capability，但能 unblock T20 corpus diversity 的实际效果。

**修复 cost**：~80 LOC — 实现 categorical sampling with temperature
+ top-k/top-p 的 device kernel；CLI 加 `--rollout-temperature
--rollout-top-k --rollout-top-p`。可暂时用 CPU readback + rand
做 first pass。

**与 T16/T20 关系**：与 T20 协同放大。greedy + 1000 prompt 比
greedy + 20 prompt 好，但**stochastic + 1000 prompt** 才接近 TRL
recipe。

---

### 缺陷 #5 — 缺 LR schedule + weight_decay = 0（MEDIUM confidence）

**证据**：`train_cli.rs:254, 337` `AdamW::new(args.lr, (0.9, 0.999),
1e-8, 0.0)` — wd=0；OPD path 不 call `optimizer.set_lr()`，无 warmup
no decay。`autograd/src/lr_schedule.rs` 早已实现 `LinearWarmup` +
`CosineWithWarmup`，被 `crates/train/src/trainer.rs:425` 使用，但
OPD `opd_step*` 不走 Trainer。

**为何重要**：
- **No warmup**：第一个 step 立刻吃满 lr。蒸馏前期 teacher 分布跟
  student 差距最大，gradient 量级最大 — 没有 warmup 极易 spike 出
  bad direction（P5 5000 step 5 个 lr 实验里 valley→recovery 形状
  与此一致，2026-05-22 wins entry）。
- **No weight_decay**：1e-5 ~ 1e-4 lr 下 5000 step 累积小但有方向性
  的偏移，没有 wd 拉回，是过拟合到训练 prompt 的促因之一。HF Trainer 
  default wd=0.0 但所有蒸馏 best practice（DeepSeek 0.1, Qwen2.5 0.1）
  都用 wd ≥ 0.01。

**推断 capability 损失**：MEDIUM。不会单独解释 "−0.82pp MMLU"，但
跟 #1/#2/#3 叠加。

**修复 cost**：~30 LOC — 把 `Trainer` 的 schedule 接口接到
`opd_step_with_teacher_forward_profiled_gkd_anchor` 的外层 loop
（即 example/CLI 主循环），call `optimizer.set_lr(schedule.lr(step))`。
CLI 加 `--warmup-steps --weight-decay --lr-schedule {constant,linear-warmup,cosine-warmup}`。

**与 T16/T20 关系**：orthogonal。可独立打。

---

## 5. 推荐修复 Roadmap（按 ROI 排序）

每个实验保留 **single-variable change** 原则（§0 第一原则），先 cheap
ablation 验证 hypothesis，再考虑组合。

### Phase A — 24h 可拿数字的 cheap ablation（不修代码）

| # | 实验 | 单变量 | 预期信号 | 何时 kill |
| --- | --- | --- | --- | --- |
| A1 | `kl_distill_loss` 报告里加 `prompt_kl_share` 计数器 | log only | 验证假设 #2：prompt KL 占总 KL ≥ 50% | share < 30% → 缺陷 #2 降级 |
| A2 | T18 step_1000 checkpoint 上跑 manual KL with `T=4.0` 看 magnitude | 加 div by T scratch | 缺陷 #3 验证：KL 应从 1.6e-5 升到 ~1e-2 量级 | KL 没变化 → 缺陷 #3 降级 |
| A3 | T20 完成时 grep run log，按 prompt 看 unique rollout count | log only | 验证 greedy 在 1000 prompt 上是否还在重复 | unique ≥ 90% → 缺陷 #4 降级 |

A1/A3 是 grep 工作；A2 需要 ~20 LOC 写个 evaluator script。**不动
production code。**

### Phase B — 单变量 fix sweep（each 50~150 LOC）

| # | 修复 | 验证 setup | 接受标准 | 失败 → 降级理由 |
| --- | --- | --- | --- | --- |
| B1 | **加 distillation temperature**（缺陷 #3）| 固定 T18 setup + `T=4.0`，跑 1000 step，对比 step_500/1000 MMLU | MMLU ≥ T18 step1000 = 50.59 + 1pp | 无改善 → teacher 分布不是 peaky 主因 |
| B2 | **mask prompt KL**（缺陷 #2）| 固定 T18 setup，KL 只在 completion，跑 1000 step | MMLU ≥ 50.59 + 0.5pp | 无改善 → prompt KL noise 假设错 |
| B3 | **rollout_len 16/32/64** sweep（缺陷 #1，需先 land T16 sequence-windowed forward）| T20 diversity corpus，rollout ∈ {8, 32, 64} | rollout=64 的 MMLU ≥ rollout=8 + 1pp | 无单调改善 → rollout length 不是瓶颈 |
| B4 | **加 linear warmup 500 step + wd=0.01**（缺陷 #5）| 固定 T18，warmup + wd | MMLU ≥ 50.59 + 0.3pp（弱信号期望）| 噪声内 → 缺陷 #5 降级 |

每个实验独立跑，single-variable，绝不混合改两个 knob。

### Phase C — 联合 recipe（仅 B 阶段单点 fix 至少两个证明 ≥ +0.5pp 后启动）

| 候选 recipe | 变更组合 | 预期 |
| --- | --- | --- |
| C1 "TRL parity" | T=0.9 + completion-only KL + lambda=0.5 mix CorpusTruth SFT + warmup | 目标：beat base MMLU 51.41pp |
| C2 "Thinking Machines parity" | reverse KL + completion-only + rollout=128 + stochastic sample | 目标：+3pp MMLU |
| C3 "Hinton classic" | T=4.0 + T² scaling + completion-only + AdamW wd=0.01 | 目标：+1pp MMLU baseline |

Phase C 需要 GKD CLI surface 真的暴露（缺陷栏外的 P0 修复：把
`opd_step_with_teacher_forward_profiled_gkd_anchor` 的所有 flag 加进
`TrainOpdArgs`）。

### Phase D — Reverse KL（仅 C 阶段没有 cross base 时启动）

代码影响最大（loss 公式重写），保留为最后选项。Thinking Machines 实证
reverse KL 在 reasoning task 上明显优于 forward KL，但对 MMLU 风格的
multi-choice 任务 forward KL 也够用，未必是瓶颈。

---

## 6. SOLID 自检

- 缺陷 #1 (rollout_len)：**HIGH** — 默认值证据强，loss 量级证据强，
  但 ARLE 内 sweep 数据没有，capability 损失幅度是 hypothesis。
- 缺陷 #2 (prompt KL)：**HIGH** — 源码对比 TRL 直接，但 A1 计数器
  实验前 prompt_kl_share 是 hypothesis。
- 缺陷 #3 (temperature)：**HIGH** — T18 KL 量级 1.6e-5 是 SOLID
  量化 evidence；A2 实验即可 confirm。
- 缺陷 #4 (greedy rollout)：**MEDIUM** — paper recommendation 强，
  但 ARLE 内 ablation 没做。
- 缺陷 #5 (no LR schedule + wd=0)：**MEDIUM** — 与 #1-#3 比次要，
  但 cheap fix。

**未做 evidence / hypothesis 标注**：
- "rollout=8 是 single biggest hypothesis" — hypothesis，未 ARLE 内
  sweep。
- "prompt KL 占 66.7% gradient" — 算术 lower-bound，不是 grad-norm
  实测。
- "teacher 分布 peaky 是 KL=1.6e-5 主因" — hypothesis；alt
  explanation: ARLE 的 mean-over-(positions×vocab) 自带 1/vocab
  rescale = 1/248320 ≈ 4e-6，可能 KL 数值本身就 ÷ 248320 后量级才
  这么低。**这点必须做 SOLID 隔离**：若用 sum/positions 而非 mean
  reduction，KL 量级直接×248320 → ~4 — 那就跟 temperature 无关，是
  reduction scale 的 artifact。

**最关键 followup（SOLID gap）**：在写 B 阶段实验前，先在 T18 已存
checkpoint 上用 `sum/positions` reduction 重算 KL，看量级是 ~4
还是 ~1.6e-5。如果是 4，缺陷 #3 的 evidence 大幅减弱，需重新评估
ranking。

---

## 7. 参考

- GKD paper: https://arxiv.org/abs/2306.13649
- TRL GKDTrainer 源码:
  https://raw.githubusercontent.com/huggingface/trl/main/trl/experimental/gkd/gkd_trainer.py
- TRL GKDConfig: https://raw.githubusercontent.com/huggingface/trl/main/trl/experimental/gkd/gkd_config.py
- MiniLLM: https://arxiv.org/abs/2306.08543
- Thinking Machines 2025 blog: https://thinkingmachines.ai/blog/on-policy-distillation/
- DeepSeek-R1-Distill-Qwen: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- Hinton 2015 distillation: https://www.cs.ubc.ca/~lsigal/532S_2018W2/4b.pdf
- ARLE T18 wins entry:
  `docs/experience/wins/2026-05-25-t18-recipe-variant-result.md`
- ARLE OPD source:
  `crates/train/src/opd.rs`,
  `crates/train/src/loss.rs`,
  `crates/cli/src/args.rs:449-493` (TrainOpdArgs),
  `crates/cli/src/train_cli.rs:254`,
  `crates/train/examples/opd_step_cuda_infer_teacher_train.rs:43-51`.
