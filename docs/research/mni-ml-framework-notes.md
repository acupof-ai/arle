# mni-ml/framework — 参考分析（autograd + 训练栈）

- **Source**: <https://github.com/mni-ml/framework>
- **License**: check upstream (MIT/Apache-2 family, confirm before vendoring any code)
- **Self-description**: "A machine learning library with a TypeScript API and Rust backend. CUDA and WebGPU compatibility. Built to understand how ML frameworks and models work internally."
- **Why it matters to us**: it is **小而全** — autograd tape + CPU/CUDA/WebGPU backends + 足量 op 覆盖 + 优化器 + nn Module，代码量大约 10K 行（见下方 LOC 表），是从零写 LoRA-级训练栈最合适的参考。不是拿来抄的，是拿来**吃透**的。

---

## 1. 仓库布局（原版）

```
framework/
├── src/
│   ├── tensor.ts              ← TS Tensor 包装 (id + shape)
│   ├── nn.ts                  ← Linear/Conv1d/Conv2d/Embedding/ReLU/Sigmoid/Tanh
│   ├── module.ts              ← Module/Parameter 抽象
│   ├── optimizer.ts           ← SGD/Adam（TS 层调度）
│   ├── native-loader.ts       ← 按平台/feature 加载对应 native .node
│   ├── native/                ← Rust 原生后端（napi-rs）
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs         ← napi 导出层 (~607 LOC)
│   │       ├── tensor.rs      ← GpuTensor + TensorStore + CachingAllocator
│   │       ├── autograd.rs    ← Tape/TapeEntry/BackwardOp 分发 (~222 LOC)
│   │       ├── device.rs      ← GpuDevice 单例 (~423 LOC)
│   │       ├── allocator.rs   ← 显存/堆缓存分配器
│   │       └── ops/
│   │           ├── elementwise.rs   ← add/mul/sub/neg/exp/log/div/pow (~1.4K LOC, hot path)
│   │           ├── matmul.rs        ← CPU + cuBLAS GEMM
│   │           ├── reduce.rs        ← sum/mean/max
│   │           ├── norm.rs          ← softmax/layernorm
│   │           ├── activation.rs    ← relu/gelu/sigmoid
│   │           ├── loss.rs          ← cross_entropy (fwd+bwd)
│   │           ├── embedding.rs     ← embedding table lookup + bwd
│   │           ├── attention.rs     ← flash-attention 前向+反向
│   │           ├── conv.rs          ← conv1d/conv2d
│   │           ├── pooling.rs       ← avg/max pool
│   │           ├── dropout.rs       ← mask-cached dropout
│   │           ├── fused.rs         ← bias+gelu, residual+layernorm
│   │           ├── layout.rs        ← view/permute/contiguous
│   │           ├── data.rs          ← IntStore（存 embedding indices 等整数张量）
│   │           ├── optimizer.rs     ← AdamW step（CPU + CUDA）
│   │           └── mixed_precision.rs
│   └── web/
│       ├── tape.ts            ← 纯 TS 版 autograd tape（~70 LOC，极简参考）
│       ├── ops.ts             ← WebGPU 路径 op 集合
│       ├── store.ts           ← tensor id → buffer 映射
│       └── optimizer.ts
├── test/
│   ├── autograd.test.ts       ← ★ 梯度数值 vs 解析验证范式
│   ├── tensor.test.ts
│   ├── nn.test.ts
│   ├── module.test.ts
│   └── native.test.ts
├── toy/                       ← minitorch 风格教学版（scalar autograd → tensor autograd）
└── npm/*/                     ← 每个平台一个 npm 子包（darwin-arm64, linux-x64-gnu-cuda, ...）
```

### LOC 快照

| 区块 | LOC | 备注 |
|---|---|---|
| `native/src/autograd.rs` | 222 | **核心**：Tape + DFS 拓扑 + op 分发 |
| `native/src/tensor.rs` | 541 | GpuTensor (CPU/CUDA 双 `#[cfg]`) + Store |
| `native/src/device.rs` | 423 | GpuDevice 单例 + stream/module 管理 |
| `native/src/allocator.rs` | 84 | CachingAllocator |
| `native/src/ops/elementwise.rs` | 1403 | 最肥的文件，每个 op fwd+bwd 双路径 |
| `native/src/ops/matmul.rs` | 333 | CPU naive + cuBLAS StridedBatched |
| `native/src/ops/reduce.rs` | 480 | sum/mean/max，reduce_axis 逻辑 |
| `native/src/ops/norm.rs` | 379 | softmax/layernorm，数值稳定实现 |
| `native/src/ops/loss.rs` | 340 | cross_entropy fwd + bwd |
| `native/src/ops/optimizer.rs` | 276 | AdamW CPU + CUDA |
| `web/ops.ts` | 1686 | WebGPU 侧 op 全量（非我们关注重点） |
| **总计** | ~10.5K | — |

**可复用比例估计**：我们只训练 LoRA（base 冻结），不需要 conv/pool/dropout/attention bwd/embedding bwd/layernorm bwd 大部分。核心参考价值集中在：

- `autograd.rs`（全部，~220 LOC）
- `ops/matmul.rs`（~150 LOC 纯 API 结构）
- `ops/elementwise.rs` 的 add/mul/sub/scalar 分支（~300 LOC）
- `ops/reduce.rs` 的 sum/mean（~200 LOC）
- `ops/norm.rs` 的 softmax（~150 LOC）
- `ops/loss.rs` cross_entropy（~100 LOC）
- `ops/optimizer.rs` AdamW（~80 LOC CPU + ~80 LOC CUDA）

**估算对我们**：照这个浓度，LoRA-only 的 autograd+AdamW 底座在 1500–2500 行 Rust 可拿下（不含 CUDA kernel，因为 LoRA 的矩阵非常小，cuBLAS SGEMM 够）。

---

## 2. 核心设计（值得吸收的抽象）

### 2.1 Tape + BackwardOp 分发（`native/src/autograd.rs`）

**核心对象**：

```rust
pub type TensorId = usize;

pub enum SavedContext {
    None,
    Tensor(TensorId),
    Tensors(SmallVec<[TensorId; 4]>),
    TensorAndScalar(TensorId, f32),
    TensorsAndShape(SmallVec<[TensorId; 4]>, Vec<usize>),
    // ... 每个 op 按需扩一个变体
}

pub enum BackwardOp {
    Add, Mul, Sub, Neg, MulScalar, Exp, Log,
    MatMul, Gelu, Relu, Sum, Mean, Max,
    View, Permute, Contiguous,
    Softmax, LayerNorm, Embedding, CrossEntropy,
    FlashAttention, ResidualLayerNorm, BiasGelu,
    Dropout, Div, Sigmoid, Pow,
    Conv1d, Conv2d, AvgPool2d, MaxPool2d,
    // + GPU 变体
}

pub struct TapeEntry {
    pub op: BackwardOp,
    pub output_id: TensorId,
    pub input_ids: SmallVec<[TensorId; 2]>,
    pub saved: SavedContext,
}

pub struct Tape {
    entries: Vec<TapeEntry>,
    enabled: bool,
}
```

**关键观察**：

1. **`TensorId = usize`（不是 `Rc<Tensor>`）**。所有 tensor 住在一个 `TensorStore: Vec<Option<GpuTensor>>`，id 是 slot index。autograd 只传 id，不传引用 —— 避免 Rust 的借用检查地狱。**我们直接照搬**。
2. **`SavedContext` 是 enum**，不是 `Box<dyn Any>`。每个 op 的"保存给反向的东西"形态有限，enum 变体覆盖即可 —— 代价是每加一个新 op 需要加一个变体，但换来零动态分派和清晰的 bwd 签名。**照搬**。
3. **`BackwardOp` 是 enum**，不是 trait object。`dispatch_backward` 是一个 big match。**照搬** —— 30 个变体 match 对编译器没压力，比 vtable 查找快。
4. **Tape backward 实现 ~75 LOC**：
   - DFS 收集"对 loss 有贡献"的 tensor 集合（relevant set）
   - 在该集合上再做 DFS 后序（拓扑序）
   - 从 loss 开始 grad = ones_like，按拓扑序 dispatch bwd
   - grad 累加到 `HashMap<TensorId, TensorId>`，对 `requires_grad=true` 的原始参数，`store.accumulate_grad()` 落到 tensor 自己的 `.grad` 字段
   - 这个实现对**一次性 backward**是正确的；多次 backward / retain_graph 另做
5. **`enabled` flag** 允许 backward 内部临时关 tape（避免反向里再录反向）。**照搬**。

### 2.2 TensorStore（`native/src/tensor.rs`）

```rust
pub struct TensorStore {
    pub(crate) tensors: Vec<Option<GpuTensor>>,
    pub(crate) free_ids: Vec<TensorId>,
    alloc: CachingAllocator,
}

pub struct GpuTensor {
    pub data: CudaSlice<f32>,       // 或 Vec<f32> for CPU
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub size: usize,
    pub requires_grad: bool,
    pub grad: Option<TensorId>,
    pub adam_m: Option<CudaSlice<f32>>,   // optimizer state 直接挂 tensor 上
    pub adam_v: Option<CudaSlice<f32>>,
}
```

**关键观察**：

1. **optimizer state 挂 tensor 本身**（`adam_m`, `adam_v`），不是 optimizer 里维护一个 `HashMap<ParamId, State>`。好处：tensor 释放时 state 自动释放；坏处：只支持一种 optimizer。我们只要 AdamW，**照搬**。
2. **slot-recycled free list**（`free_ids: Vec<TensorId>`），释放后 slot 复用。必要，避免长期跑 id 单调增长到无限大。**照搬**。
3. **CachingAllocator** 是典型 PyTorch 式 caching allocator。我们第一版不要，直接 `cudarc::CudaDevice::alloc`，等压测到分配瓶颈再做。
4. **`#[cfg(feature = "cuda")]` vs `#[cfg(any(feature = "cpu", feature = "webgpu"))]` 双路径**。值得学习的纪律：数据 layout、is_contiguous 这类 shared 逻辑一份；只有数据类型（`Vec<f32>` vs `CudaSlice<f32>`）和 kernel launch 不一样。我们沿用这个模板，把 CPU 路径当成"参考实现 + 梯度数值校验用"。

### 2.3 AdamW（`native/src/ops/optimizer.rs`）

CPU 路径 ~45 LOC：

```rust
bc1 = 1.0 - beta1^step;    // bias correction
bc2 = 1.0 - beta2^step;
for each param {
    g = param.grad;
    if weight_decay > 0 { data *= 1 - lr*wd; }   // AdamW 分离式 weight decay
    m = beta1*m + (1-beta1)*g;
    v = beta2*v + (1-beta2)*g*g;
    m_hat = m / bc1;
    v_hat = v / bc2;
    data -= lr * m_hat / (sqrt(v_hat) + eps);
}
```

CUDA 路径同逻辑，写成 fused kernel。**照搬**，这是标准 AdamW，没有发挥空间。

### 2.4 napi 绑定（我们不需要）

`native/src/lib.rs` 全部是 napi-rs 把 Rust op 暴露给 TypeScript。**我们不需要这一层** —— agent-infer 是纯 Rust，直接函数调用即可。**省掉 ~600 LOC**。

---

## 3. 我们的裁剪清单

### 3.1 必须要（移植/重写）

| mni-ml 模块 | 我们的模块 | 裁剪说明 |
|---|---|---|
| `autograd.rs` | `autograd::tape` | 全量移植，`SavedContext` / `BackwardOp` 只保留 LoRA+GRPO 用到的变体 |
| `tensor.rs` TensorStore | `autograd::store` | 照搬 slot + free list 模式，`GpuTensor` 包 `cudarc::CudaSlice<f32>`，第一版不做 caching allocator |
| `ops/matmul.rs` CUDA 路径 | `autograd::ops::matmul` | 直接 cuBLAS SGEMM，LoRA 矩阵小，不上 Marlin/Triton |
| `ops/elementwise.rs` add/mul/sub/mul_scalar | `autograd::ops::elementwise` | 只搬 4 个 op，砍掉 exp/log/pow/div 等 |
| `ops/reduce.rs` sum/mean | `autograd::ops::reduce` | 砍掉 max（GRPO 不用） |
| `ops/norm.rs` softmax/log_softmax | `autograd::ops::softmax` | log_softmax 一起实现，数值稳定版 |
| `ops/loss.rs` cross_entropy | 不直接用；换成 GRPO loss | cross_entropy 作为 grad-check 的参考实现 |
| `ops/optimizer.rs` AdamW | `autograd::optim::adamw` | 全量移植 CUDA 路径 |

### 3.2 不要（明确砍掉）

- `conv.rs`, `pooling.rs` — LLM 不用
- `embedding.rs` — base 冻结，embedding 不训
- `attention.rs` (flash-attention bwd) — base 冻结，不穿过 attention backward
- `fused.rs`, `mixed_precision.rs` — 优化项，第二版再说
- `dropout.rs` — LoRA 可选 dropout，第二版再说
- `layout.rs` view/permute/contiguous — **需要**但第一版用最简 API（只支持 contiguous），后续再补
- `data.rs` IntStore — 不需要
- `allocator.rs` CachingAllocator — 第一版用 cudarc 原生分配
- 整个 `web/` 目录 — WebGPU 不在路线
- 整个 `npm/` 目录 + napi 绑定 — 纯 Rust 项目不需要
- `toy/` — 教学目录，只读不动

### 3.3 我们新增的（mni-ml 没有）

| 新增 | 位置 | 原因 |
|---|---|---|
| LoRA adapter 层（历史实验，已删除） | 旧 `train::lora` | mni-ml 没有 LoRA 概念；当前树不保留无调用者实现 |
| 与 agent-infer base forward 的 hook 点（历史想法，未保留） | 旧 `train::lora::hook` | 让 `Linear(x)` 在 base 侧执行后加 `B·A·x`；两套前向共享 base weights |
| GRPO loss + advantage 归一化 | `train::grpo` | mni-ml 只有 cross_entropy |
| Trajectory buffer | `train::rollout` | mni-ml 是 supervised，没有 RL 循环 |
| Reward verifier trait | `train::reward` | 同上 |
| LoRA delta double-buffer 热切 | `train::weight_sync` | mni-ml 是单进程单模型 |
| 梯度 check harness | `autograd::tests::grad_check` | mni-ml 的 autograd.test.ts 是参考范式 |

---

## 4. 值得复制的测试范式

`test/autograd.test.ts` 的典型结构（TS 版）：

```typescript
it("add backward", () => {
    const a = Tensor.randn([2, 3]); a.requiresGrad = true;
    const b = Tensor.randn([2, 3]); b.requiresGrad = true;
    const c = a.add(b);
    const loss = c.sum();
    loss.backward();
    // 解析解：∂(sum(a+b))/∂a = ones_like(a)
    expect(a.grad.toFloat32()).toAllBeCloseTo(ones);
});
```

**关键要素**：

1. 每个 op 都有单独的 bwd 测试
2. 比对值 = 解析解（闭式可求的简单情况）
3. 对复杂 op（softmax, cross_entropy），加 **数值梯度 check**：
   ```
   grad_num[i] = (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps)
   assert_allclose(grad_num, grad_analytical, rtol=1e-3)
   ```

**我们 M0 的验收 = 每个 op 有一个 grad_check 单测，double precision 跑 CPU 路径，和 CUDA 路径 bitwise-ish 对拍。**

---

## 5. 值得抄的"节约了多少人月"笔记

| 坑 | mni-ml 的解 | 省了我们多少 |
|---|---|---|
| tensor 生命周期 | `TensorId: usize` + `Store` | 2 周（试 `Rc<RefCell>` / `Arc<RwLock>` 都会崩） |
| savedcontext 形态 | enum 变体，不是 dyn Any | 1 周 |
| backward 拓扑序 | DFS relevant set + 后序 | 1 周 |
| AdamW 分离 weight decay | 照标准实现 | 2–3 天 |
| cuBLAS StridedBatched 配置 | 代码里现成 | 2 天（第一次用 cuBLAS 永远会踩 row-major/col-major） |
| CPU / CUDA 双路径 `#[cfg]` 骨架 | 现成 | 1 周 |

合计：**6–8 周的"填坑"省掉**。剩下的工作是：选择性移植 + 和 agent-infer 的 base forward 打通 + LoRA/GRPO/rollout 逻辑。

---

## 6. License / 合规

- **不直接 vendor 代码**。上游是教学框架，我们是生产系统。我们**读、理解、重写**，不复制粘贴（除非是"SGEMM cuBLAS API 形态"这种无可争议的公共知识）。
- 参考来源在 PR commit message 和本文件里明确致谢。
- 如果某个具体函数（如数值稳定 softmax 实现）直接借鉴，在实现文件 header 注释写明"structure inspired by mni-ml/framework <path>"。

---

## 7. Open questions（本调研未解决）

1. **CPU 路径要不要留？** mni-ml 留了，用于无 GPU 环境。我们的约束是"单机 CUDA first，Metal 支线"。CPU 路径的价值是 **grad-check 的黄金参考**（bitwise stable，double 精度），建议**保留最小集**（只够 grad-check 的 op）。
2. **和 `cuda-kernels` 的 device 复用**：mni-ml 有 `GpuDevice::instance()` 单例；agent-infer 现在有自己的 CUDA context 管理。spike 阶段需要验证两者能否共进程共 context（预期可以，`cudarc` 就支持多 `Arc<CudaDevice>` 指向同一 GPU 上下文）。
3. **LoRA adapter 如何接 agent-infer base forward**：需要一个 hook 点让 `W @ x` 执行完后立刻执行 `B @ (A @ x)` 并相加。最简方案：在 `autograd::ops::linear` 里提供 `linear_with_lora(x, W_frozen, A, B)`，base 侧执行走 agent-infer 的 merged-QKV / gate-up 路径，LoRA 分支独立 cuBLAS 两次小 GEMM。
4. **Base weights 的 ownership**：agent-infer 目前 `Arc<Weights>`，我们新建一个 `autograd::FrozenView` 零拷贝包装，不参与 tape。

---

## 8. 结论

mni-ml/framework 是**合格的教学参考**，恰好覆盖我们需要的"小而完整 autograd + Rust CUDA 后端"形态。核心抽象（TensorId+Store, SavedContext enum, BackwardOp dispatch, AdamW with embedded state）直接照搬能省 6–8 周。

我们的实现 = **mni-ml 的裁剪子集 + LoRA adapter + GRPO + rollout 闭环 + 和 agent-infer base forward 打通**，不做 conv/pool/attention-bwd/embedding-bwd 等 LoRA-only 场景用不到的部分。

下一步：按 [`docs/plans/rust-agent-rl-single-node.md`](../plans/rust-agent-rl-single-node.md) 执行 M0–M3。
