---
title: "Tuning: EP 零冗余权重同步与 Resharding 优化 (ZeroSync)"
tags: [status/verified, category/comm, category/memory]
last_updated: 2026-05-27
source: [outputs/logs/vlm_0527_142116_2739_verify_eptrans_bs128.log]
confidence: high
status: active
open_questions:
  - "在极多卡（如 64+ 卡）并发环境下，HCOLL 算子建立连接与小张量同步时的通信握手开销对性能的稀释程度"
contradictions: []
---

# Profiling & Tuning Report: EP 零冗余权重同步与 Resharding 优化 (ZeroSync)

## 1. 瓶颈分析 (Bottleneck Analysis)
- **初始性能**：
  在 16 卡 NPU（Ascend）的多卡多模态强化学习训练环境下，未开启 ZeroSync 时，每次大 Step 更新模型权重（`update_weights`）的平均端到端同步耗时约为 **55.42 秒**（由于 MoE 专家层全量 Broadcast 导致在第 2/3 步时同步时间恶化至 59.0s 以上）。
- **Profiling 现象与瓶颈**：
  - **通信冗余**：2B5 MoE 模型包含 128 个专家（单专家 FC1/FC2 维度为 `(2048, 2048)`）。在训练 EP=16 拓扑下，每张卡实际仅需保留 8 个专家的局部权重。但在未优化的 Broadcast 模式下，发送端会将包含全部 128 个专家的完整权重张量广播给所有 16 个 Rank，导致每个节点承受了 $\frac{15}{16}$ 的严重通信带宽冗余。
  * **显存压力（瞬时峰值）**：所有接收节点在同步瞬间必须临时开辟能够容纳全量专家权重的 HBM（显存）空间，并在本地切片过滤出自身对应的 8 个专家后释放。这带来了极高的瞬时显存分配峰值，极大增加了长周期训练中多卡发生 OOM 崩溃的概率。

## 2. 优化方案 (Optimization Strategy)
### 方案 A: 基于 PyTorch 原生 `dist.scatter` 的零冗余权重同步与自适应 Resharding
- **原理**：
  * **零冗余 Scatter**：在推理端权重同步加载阶段（Mojo 侧 [model_loader.py](file:///Users/bytedance/posttrain/codes/maas_engine_vlm2b5/3rdparty/xpu_gpt/xpu_gpt/core/mojo/model_loader.py#L412-L425)），发送端在本地先利用 `ep_split` 将 128 个专家切分成 16 份（每份 8 个专家），然后通过 PyTorch 原生的 `dist.scatter` 集合通信算子将对应的切片定向分发给 16 个引擎节点。接收端仅需要预先分配局部切片大小的显存（`local_shape=(8, 2048, 2048)`）。
  * **自适应 Resharding 通路**：
    为了解耦训练侧与推理侧可能存在的 EP 并行度差异（即满足 `train_ep_size != infer_ep_size` 的调优诉求），采用“共享内存（SHM）先合后分”的设计：
    1. **训练端（Actor 侧）**：各 FSDP 分片进程提取参数时识别 `DTensor`，利用 `full_tensor()` 完整收集并还原。随后通过 `WeightManager` 合并全量 128 个专家的参数，存入 CPU 共享内存中。
    2. **推理端（Mojo 侧）**：引擎 Rank 0 每次都读取共享内存中的全量 128 个专家张量，再由 `ep_zero_redundant_sync` 函数**依据推理侧自身的 `self.ep_size` 重新进行 Scatter 切分并分发**（例如若调优为 8EP，则每个 Rank 认领 16 个专家）。这避免了基于 TensorRTL 跨进程双边点对点 P2P 连接的高额握手成本及死锁风险。
- **实施步骤**：
  在 [runtime_env.yaml](file:///Users/bytedance/posttrain/codes/maas_engine_vlm2b5/runtime_env.yaml#L342) 中显式配置环境变量：
  ```yaml
  # 激活 EP 零冗余 Scatter 传输
  "MAAS_XLLM_EP_ZERO_REDUNDANT_SYNC": "1"
  ```
  该变量会经由 [ray.sh](file:///Users/bytedance/posttrain/codes/maas_engine_vlm2b5/ray.sh) 自动覆盖传导至各推理 Worker 进程。

## 3. 优化效果 (Results)
- **最终性能**：
  在 16 卡 NPU (Ascend 2739)、`bsz=128` 的图文多模态长周期实测下，开启优化后，多步权重同步的平均耗时降至 **53.56 秒**（平均绝对值缩短 **1.86秒**，整体效率提升约 **3.36%**）。
  在网络和算子建立稳态的第 2 步起，单次同步耗时由基线的 **59.07 秒** 锐减至 **52.42 秒**（单步绝对收益达 **6.65秒**，性能提升达 **11.26%**）。
- **显存与稳定性收益**：
  MoE 专家同步时的接收数据缓冲区大小被压缩至原本的 $\frac{1}{16}$。消除了显存分配的突发峰值，使 HBM 使用更加平滑，彻底杜绝了因权重同步引发的连环 OOM，是保证多 Step 长跑稳态收敛的防御性机制。
- **注意事项**：
  在同步的第一轮（预热期），由于需要等待 HCOLL 算子与卡间通信通道的初次链接建立，底层启动延迟占大头，优化效果会被握手延迟掩盖（优化前后耗时在 48.0s ~ 48.1s 左右持平）。

## 4. 关联知识
- [[wiki/05_fact/vlm_2b5_inference_service_golden_config|VLM 2B5 推理服务化部署黄金配置]]
- [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
