# Weak-to-Strong 在线蒸馏实验总结

## 结论

两个 0.8B 弱模型的策略偏移量，经 w2s 蒸馏后，使 27B 强模型在 GSM8K 上准确率提升 1.52 个百分点。

| 配置 | GSM8K 准确率 | 正确数 / 有效数 |
|------|-------------|----------------|
| 27B 基线 | 94.42% | 186 / 197 |
| w2s 100 步 α=0.5 | 95.43% | 188 / 197 |
| w2s 200 步 α=1.0 | 95.94% | 189 / 197 |

## 方法

核心思路是不直接蒸馏弱模型的输出，而是蒸馏弱模型从预训练到指令微调的策略偏移量。

策略偏移量定义为后训练 logits 减去预训练 logits，在 logit 空间计算。

$$\Delta T = z_{\text{post-RL}} - z_{\text{pre-RL}}$$

代理教师由学生自身 logits 加上两个弱模型偏移量的平均构成。

$$z_{\text{proxy}} = z_s + \alpha \cdot T \cdot \frac{\Delta T_1 + \Delta T_2}{2}$$

损失函数为反向 KL 散度加两个正则项。反向 KL 是 mode-seeking，使学生分布趋向代理教师的众数。

$$\mathcal{L} = T^2 \cdot \text{KL}_{\text{reverse}}(\pi_s \| \pi_{\text{proxy}}) + \beta_1 \cdot \text{KL}(\pi_{\text{new}} \| \pi_{\text{old}}) + \beta_2 \cdot \text{KL}(\pi_{\text{new}} \| \pi_{\text{base}})$$

两个正则项分别约束学生不偏离上一步适配器和基座模型，防止灾难性遗忘。

## 实验设置

学生模型为 ThinkingCap-Qwen3.6-27B-FP8。

两个弱模型对共享同一个预训练基座 qwen35-08b-clean，后训练模型分别为 qwen35-08b-w8a16 和 qwen35-08b-w8a16b。

训练数据为 GSM8K 训练集 1000 条 prompt。

适配器配置为 LoRA rank 16，alpha 32，目标模块 q_proj 和 v_proj。

## 关键工程修复

适配器保存时，保存函数提前创建了输出目录，而检查点写入逻辑拒绝向已存在目录合并。删除了提前创建目录的调用。

INT8 量化的弱模型在部分输入上会输出 NaN logits，经偏移量传播后污染学生权重。在一致性门控处增加 NaN 检测，跳过此类样本。

构建环境中，ELKEID 安全代理会在 CUDA 设备可见时杀死 nvcc 编译进程。构建阶段设置 CUDA_VISIBLE_DEVICES 为空。

OpenSSL 3.0 移除了 SSL_get_peer_certificate，native-tls 无法链接。reqwest 和 hf-hub 切换到 rustls-tls。

## 数据观察

200 步训练中，16 步因 NaN 跳过，108 步因学生置信度过高跳过，76 步实际更新。

置信度过滤阈值为 0.99，学生对多数样本已有较高置信度，实际更新步数有限。

损失值在 0.01 到 0.04 之间波动，无明显下降趋势。这符合 KL 散度作为优化目标的特性，损失绝对值不直接对应生成质量。
