---
title: "Tuning: VLM 23B RL torch infer 与 XG 推理后端性能对比"
tags: [status/partial, category/scheduling, category/compute, category/comm, category/memory]
last_updated: 2026-06-17
source:
  - run_torch_infer_vlm_23b.sh
  - run_xg_vlm_23b.sh
  - outputs/logs/vlm_0606_070312_torch_infer_23b_full_8nodes_vit_responsefallback_bs256_long_bs256_57steps.log
  - outputs/run_time_envs/runtime_env_0606_070312_torch_infer_23b_128rank.yaml
  - outputs/logs/vlm_0616_065614_xg_23b_rollout_ep64_kv64_tok2048_hccl400c1024_200step_bs64.log
  - outputs/run_time_envs/runtime_env_0616_065614_xg_23b.yaml
  - outputs/logs/vlm_0616_191446_xg_23b_rollout_ep128_kv64_tok2048_if128_b8_kvprobe_hccl400c1024_200step_bs64.log
  - outputs/DEBUG/hbm_8node_0616_191446_ep128_sync0_hccl300.jsonl
  - outputs/logs/vlm_0616_210520_xg_23b_rollout_ep128_kv64_tok2048_if128_b16_kvprobe_hccl400c1024_200step_bs64.log
  - outputs/DEBUG/hbm_8node_0616_210507.jsonl
  - outputs/logs/vlm_0616_234100_xg_23b_rollout_ep128_kv64_tok2048_if128_ctx16_b8_kvprobe_hccl400c1024_200step_bs64.log
  - outputs/DEBUG/hbm_8node_0616_234100_ep128_ctx16_b8.jsonl
  - outputs/logs/vlm_0617_015800_xg_23b_rollout_ep128_kv64_tok2048_if96_b8_kvprobe_hccl400c1024_200step_bs64.log
  - outputs/DEBUG/hbm_8node_0617_015800_ep128_if96_b8.jsonl
  - outputs/logs/vlm_0617_073951_xg_23b_rollout_ep128_kv64_tok2048_if144_b8_kvprobe_hccl400c1024_200step_bs64.log
  - outputs/DEBUG/hbm_8node_0617_073951_ep128_if144_b8.jsonl
  - outputs/logs/vlm_0617_095100_xg_23b_rollout_ep128_kv64_tok2048_if160_b8_kvprobe_hccl400c1024_200step_bs64.log
  - outputs/DEBUG/hbm_8node_0617_095100_ep128_if160_b8.jsonl
  - outputs/logs/vlm_0617_121607_xg_23b_rollout_ep128_kv64_tok2048_if144_b12_kvprobe_hccl400c1024_200step_bs64.log
  - outputs/DEBUG/hbm_8node_0617_121607_ep128_if144_b12.jsonl
  - wiki/02_precision/vlm_torch_infer_response_vit_embeds_unified_cache.md
confidence: high
status: active
open_questions:
  - "XG 0616 日志当前只完成 3 个 global_step，不能替代 70+ / 200 step 长线漂移与稳定性结论。"
  - "torch infer 0606 日志完成 57 step 后在 step58 前后触发 torch_npu stream timeout / raylet dead，仍需独立修复长跑稳定性。"
  - "XG ep128 的 b8/b16/ctx16/if96/if144/if160/b12 对比仅覆盖早期 step2，能指导下一轮调参，但不能证明 200 step 最优配置。"
  - "XG response-level vit_embeds 透传与 kv80 尚待新 run 验证；当前只能说 b12 日志证明旧路径仍在 Actor 侧 RECOMPUTE。"
contradictions: []
---

# Profiling & Tuning Report: VLM 23B RL torch infer vs XG

## 1. 瓶颈分析 (Bottleneck Analysis)

### 口径边界

- torch infer 证据来自 `vlm_0606_070312_torch_infer_23b_full_8nodes_vit_responsefallback_bs256_long_bs256_57steps.log`，完成 `global_step=1..57`，`trainer.total_training_steps=400`。该历史 run 的有效 Hydra 配置为 `data.batch_size=256`、`rollout.generation.n=8`、`train/batch_size=2048`、`rollout.tp_size=8`、`rollout.dp_size=2`、`rollout.enable_async=True`、`scheduler/model batch=512`、`inferencer_async_runtime_chunk_size=10240`、`max_tokens_per_step=256`。当前 `run_torch_infer_vlm_23b.sh` 默认已变为 scheduler/model batch `384`、chunk `8192`，不能把当前脚本默认值反推到 0606 历史日志。
- XG 证据来自 `vlm_0616_065614_xg_23b_rollout_ep64_kv64_tok2048_hccl400c1024_200step_bs64.log`，当前只完成 `global_step=1..3`，`trainer.total_training_steps=200`。有效配置为 `data.batch_size=64`、`rollout.generation.n=8`、`train/batch_size=512`、`rollout.ep_size=64`、`rollout.tp_size=1`、`rollout.enable_async=False`、`scheduler/model batch=8`、`inferencer_async_runtime_chunk_size=2048`、`max_tokens_per_step=2048`、`XPU_KV_NUM_BLOCKS=64`、W8A8 rollout。

### 性能对比

| 指标 | torch infer 23B 0606 | XG 23B 0616 | 备注 |
|---|---:|---:|---|
| 完成 step | 57 | 3 | XG 仅早期样本 |
| `train/batch_size` | 2048 | 512 | 均为 `data.batch_size * generation.n` |
| 平均 `train/timing/step` | 1902.80s | 2611.06s | XG 单 step 更慢且 batch 只有 1/4 |
| 平均 `train/timing/data` | 1386.88s | 2145.67s | rollout/data 阶段是 XG 最大耗时 |
| 平均 `train/speed_samples` | 1.0773 samples/s | 0.1984 samples/s | torch infer 约 5.43x |
| 平均 `train/speed_tokens` | 3197.84 tokens/s | 496.92 tokens/s | torch infer 约 6.44x |
| 平均 `sync_weight` | 22.14s | 169.69s | XG 权重同步明显更重 |
| 平均 `collect_quant_amax` | 0.00013s | 16.85s | XG W8A8 路径存在额外量化统计成本 |
| 平均 `logprobs_diff` | 0.01652 | 0.03382 | XG step1 为 0.045，step2/3 回到约 0.028 |
| `response_length.clipfrac` | 0 | 0.1999 | XG 早期约 16%-26% response 打满 4096 |

### XG 关键瓶颈

1. **rollout 有效并发远低于 torch infer**：XG 在首轮 monitor 中显示 `total_reqs=512, running_reqs=64, pending_reqs=448, reqs_per_client=[8,8,8,8,8,8,8,8]`，说明 512 条 rollout 样本只能以 64 in-flight、每 client 8 条推进。torch infer 首轮为 `total_reqs=2048, running_reqs=2048, reqs_per_client=[256,...,256]`，中后段也可见 `total_reqs=512, running_reqs=512`。这直接解释了 XG `train/timing/data` 更高、`speed_samples` 更低。
2. **XG scheduler/model batch 太小**：XG 实跑 `XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE=8`、`XLLM_MODEL_MAX_BATCH_SIZE=8`、`XLLM_INFERENCER_ASYNC_RUNTIME_CHUNK_SIZE=2048`；torch infer 0606 实跑 `scheduler/model batch=512`、chunk `10240`。XG 当前配置更像保守稳定性口径，不是吞吐优先口径。
3. **权重同步与量化路径成本明显**：XG 平均 `sync_weight=169.69s`，torch infer 为 `22.14s`；XG `collect_quant_amax=15.4~18.6s/step`，torch infer 近似为 0。这部分不是 rollout 生成本身，但会拉高端到端 step。
4. **HBM/工作区压力仍在热路径出现**：XG 日志中可见多次 `[MLP_Group_BigEP_Dyw8a8] low-HBM empty_cache before FC1/FC2 prefill workspace`，说明 BigEP W8A8 prefill 工作区仍接近低 HBM 阈值。当前 runtime 已关闭 `XPU_CP_MOE_RELEASE_CTX_WORKSPACE_HOT_PATH=0`，避免旧热路径释放导致严重退化，但低 HBM empty_cache 仍是调度和算子工作区瓶颈信号。
5. **ViT 不是主要耗时**：XG 日志里 ViT embedding probe 常见 `0.0178s`、`0.0930s` 量级，而单 step `data` 为 `1867~2522s`。因此当前核心瓶颈应集中在 LLM rollout 生成、EP64 BigEP 调度、HCCL/同步、HBM 工作区，而不是 ViT 编码。

### XG ep128 b8 -> b16 负向调参经验

本轮在 8 节点 128 rank 上将 `rollout.ep_size` 提升到 `128` 后，进一步比较了 `scheduler/model batch=8` 与 `16`。这次对比应优先看 `global_step=2`，因为 step1 混有更重的初始化、编译、首轮服务热身和长 response 分布。

| 指标 | ep128 + b8 step2 | ep128 + b16 step2 | 变化 |
|---|---:|---:|---:|
| 日志 | `vlm_0616_191446_..._b8_...log` | `vlm_0616_210520_..._b16_...log` | - |
| `train/timing/step` | 1974.33s | 2074.29s | 慢 99.96s / 5.1% |
| `train/timing/data` | 1488.46s | 1586.49s | 慢 98.03s / 6.6% |
| `train/timing/sync_weight` | 178.40s | 195.97s | 慢 17.57s / 9.8% |
| `train/speed_samples` | 0.2593 | 0.2468 | 低 4.8% |
| `train/speed_tokens` | 548.72 | 527.69 | 低 3.8% |
| `response_length.mean` | 1615.23 | 1637.15 | 长 1.4% |
| `train/num_tokens` | 1.083M | 1.095M | 多 1.0% |
| HBM 采样峰值 | max 65494 MB / avg 60691.66 MB | max 65496 MB / avg 61547.21 MB | 峰值均接近 65.5GB |

结论：`scheduler_max_schedule_batch_size`、`scheduler_max_schedule_context_batch_size` 和 `model_max_batch_size` 一起从 `8` 提到 `16` 没有带来吞吐收益，反而让 step2 端到端慢约 `5.1%`。变慢几乎全部来自 rollout/data 阶段，说明瓶颈更像 decode 调度粒度、长尾请求收敛和算子工作区竞争，而不是 actor update。HBM 稳态看似还有余量，但完整 step 采样峰值已贴近 `65536 MB`，不能把稳态低水位直接解释为可以无风险继续增加 KV 或 decode batch。

调参经验：

- 不要把 `model_max_batch_size` 和普通 decode scheduler batch 盲目加到 `16`；在当前 23B XG W8A8 + EP128 口径下，b16 已经是负收益样本。
- 如果要利用 HBM，下一步应拆开控制量：优先回退 `model_max_batch_size=8`，只单独尝试提高 context/prefill 调度，或在有 `decode_evicted`/swap 证据时再调 `XPU_KV_NUM_BLOCKS`。
- 判断显存余量必须基于完整 RL step 峰值，而不是 rollout 稳态窗口。b8/b16 两轮 HBM 峰值都接近 65.5GB，说明 post-rollout/update/同步窗口仍是 OOM 风险点。
- 本轮没有看到明确 KV 不足信号；若日志中 `decode_evicted=0`、无 swap/no block/cache full，则不要把 `XPU_KV_NUM_BLOCKS` 当作优先性能旋钮。

### XG ep128 ctx16_b8 持平/负收益调参经验

在 `b16` 负收益后，进一步只放大 context/prefill 调度：保持 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS=128`、`XPU_KV_NUM_BLOCKS=64`、`rollout.ep_size=128`、`model_max_batch_size=8`、`scheduler_max_schedule_batch_size=8`，仅将 `scheduler_max_schedule_context_batch_size=16`。该 case 至少跑到 `global_step=2`，按 step2 与历史 b8 口径对比。

| 指标 | ep128 + b8 step2 | ep128 + ctx16_b8 step2 | 变化 |
|---|---:|---:|---:|
| 日志 | `vlm_0616_191446_..._b8_...log` | `vlm_0616_234100_..._ctx16_b8_...log` | - |
| `train/timing/step` | 1974.33s | 1979.38s | 慢 5.05s / 0.3% |
| `train/timing/data` | 1488.46s | 1499.35s | 慢 10.89s / 0.7% |
| `train/timing/sync_weight` | 178.40s | 195.04s | 慢 16.64s / 9.3% |
| `train/speed_samples` | 0.2593 | 0.2587 | 低 0.2% |
| `train/speed_tokens` | 548.72 | 546.59 | 低 0.4% |
| `response_length.mean` | 1615.23 | 1612.40 | 基本持平 |
| `response_length.clipfrac` | 未单独记录 | 0.1543 | - |
| `train/num_tokens` | 1.083M | 1.082M | 基本持平 |
| step2 HBM 时间窗 | max 65494 MB / avg 60691.66 MB | max 62860 MB / avg 56445.77 MB | ctx16 step2 峰值较低，但 full run 仍到 65499 MB |

结论：单独把 `scheduler_max_schedule_context_batch_size` 提到 `16` 没有带来可见吞吐收益，step2 端到端基本持平但略慢。step2 HBM 峰值低于历史 b8 采样，但本 run 在 step1 训练收尾仍出现 `65499 MB` 全局峰值，因此不能据此继续加 `XPU_KV_NUM_BLOCKS` 或 decode/model batch。

调参经验：

- `ctx16_b8` 不能作为更优配置；在 if144 验证前，最优候选仍是 `ep128 + if128 + kv64 + ctx8/b8/model8`。
- 性能瓶颈仍体现为 rollout 长尾 decode：step2 从 `total_reqs=512,running=128,pending=384` 到尾部 `running=5,pending=0`，尾部请求耗时明显。
- 下一组应验证 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS=96` 是否能降低 HBM 峰值和长尾竞争；若 step2 变慢，则回到 `if128_ctx8_b8` 稳定口径。

### XG ep128 if96_b8 负收益调参经验

在 `ctx16_b8` 未带来收益后，进一步回退 context/model batch 到 `8`，只将 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS` 从 `128` 降到 `96`，验证降低在途请求是否能缓解 HBM 峰值、decode 长尾或工作区竞争。该 case 至少跑到 `global_step=2`，按 step2 与历史 b8 口径对比。

| 指标 | ep128 + if128_b8 step2 | ep128 + if96_b8 step2 | 变化 |
|---|---:|---:|---:|
| 日志 | `vlm_0616_191446_..._if128_b8_...log` | `vlm_0617_015800_..._if96_b8_...log` | - |
| `train/timing/step` | 1974.33s | 2279.70s | 慢 305.37s / 15.5% |
| `train/timing/data` | 1488.46s | 1801.77s | 慢 313.31s / 21.0% |
| `train/timing/sync_weight` | 178.40s | 189.39s | 慢 10.99s / 6.2% |
| `train/speed_samples` | 0.2593 | 0.2246 | 低 13.4% |
| `train/speed_tokens` | 548.72 | 482.26 | 低 12.1% |
| `response_length.mean` | 1615.23 | 1646.56 | 长 1.9% |
| `response_length.clipfrac` | 未单独记录 | 0.1641 | - |
| `train/num_tokens` | 1.083M | 1.099M | 多 1.5% |
| HBM 完整 run 峰值 | max 65494 MB / avg 60691.66 MB | max 65501 MB / avg 62370.70 MB | 峰值未下降 |

结论：单独把 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS` 从 `128` 降到 `96` 是明确负收益。它没有降低完整 step HBM 峰值，反而把 step2 端到端拉长约 `15.5%`，主要退化来自 rollout/data 阶段。step2 的 response length 只长约 `1.9%`，不足以解释 `21.0%` 的 data 时间回退。

调参经验：

- `if96_b8` 不能作为稳定或高性能口径；在 if144 验证前，最优候选仍是 `ep128 + if128 + kv64 + ctx8/b8/model8`。
- 完整 run HBM 峰值仍贴近 `65.5GB`，降低 in-flight 没有释放真正的峰值窗口，说明峰值更可能来自 step 边界、训练/update、权重同步或工作区瞬态，而不是仅由 rollout 主体并发决定。
- 若继续追求吞吐，应回到 `if128_b8` 附近做小步探索；继续降低 in-flight 只会削弱并发，增加尾部占比。
- `XPU_KV_NUM_BLOCKS` 仍不应盲目增大：当前 `kv64` 完整峰值已经贴近卡上限，且日志未显示明确 `decode_evicted`/KV block 不足是主瓶颈。

### XG ep128 if144_b8 正收益调参经验

在 `if96_b8` 明确负收益后，回到 `ctx8/b8/model8`，只将 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS` 从 `128` 小步提高到 `144`。该 case 至少跑到 `global_step=2`，按 step2 与历史 `if128_b8` 口径对比。

| 指标 | ep128 + if128_b8 step2 | ep128 + if144_b8 step2 | 变化 |
|---|---:|---:|---:|
| 日志 | `vlm_0616_191446_..._if128_b8_...log` | `vlm_0617_073951_..._if144_b8_...log` | - |
| `train/timing/step` | 1974.33s | 1796.14s | 快 178.19s / 9.0% |
| `train/timing/data` | 1488.46s | 1322.53s | 快 165.93s / 11.1% |
| `train/timing/sync_weight` | 178.40s | 183.34s | 慢 4.94s / 2.8% |
| `train/speed_samples` | 0.2593 | 0.2851 | 高 9.9% |
| `train/speed_tokens` | 548.72 | 603.30 | 高 9.9% |
| `response_length.mean` | 1615.23 | 1615.71 | 基本持平 |
| `response_length.clipfrac` | 未单独记录 | 0.1641 | - |
| `train/num_tokens` | 1.083M | 1.084M | 基本持平 |
| HBM 完整 run 峰值 | max 65494 MB / avg 60691.66 MB | max 65496 MB / avg 60883.68 MB | 峰值基本不变 |

结论：单独把 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS` 从 `128` 提高到 `144` 是当前最明确的正收益样本。step2 的 response length 与 token 数几乎持平，端到端提升主要来自 rollout/data 阶段，说明当前 XG EP128 仍受有效 in-flight 限制。

调参经验：

- 当前最优候选更新为 `ep128 + if144 + kv64 + ctx8/b8/model8`。
- HBM 峰值仍贴近 `65.5GB`，不能因为 rollout 稳态或 step2 尾部水位较低就直接增加 `XPU_KV_NUM_BLOCKS`。
- 下一组可在同一控制面继续小步试 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS=160`；若 step2 继续改善，再考虑二分搜索上界，若 OOM/HCCL timeout 或 step2 回退，则回到 `if144_b8`。

### XG ep128 if160_b8 持平/负收益调参经验

在 `if144_b8` 成为当前最优候选后，进一步只将 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS` 从 `144` 提高到 `160`，保持 `XPU_KV_NUM_BLOCKS=64`、`rollout.ep_size=128`、`XLLM_MODEL_MAX_BATCH_SIZE=8`、`XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE=8`、`XLLM_SCHEDULER_MAX_SCHEDULE_CONTEXT_BATCH_SIZE=8`。该 case 至少跑到 `global_step=2`，按 step2 与 `if144_b8` 对比。

| 指标 | ep128 + if144_b8 step2 | ep128 + if160_b8 step2 | 变化 |
|---|---:|---:|---:|
| 日志 | `vlm_0617_073951_..._if144_b8_...log` | `vlm_0617_095100_..._if160_b8_...log` | - |
| `train/timing/step` | 1796.14s | 1790.48s | 快 5.66s / 0.3% |
| `train/timing/data` | 1322.53s | 1304.64s | 快 17.89s / 1.4% |
| `train/timing/sync_weight` | 183.34s | 202.30s | 慢 18.96s / 10.3% |
| `train/speed_samples` | 0.2851 | 0.2860 | 高 0.3% |
| `train/speed_tokens` | 603.30 | 595.94 | 低 1.2% |
| `response_length.mean` | 1615.71 | 1583.29 | 低 2.0% |
| `train/num_tokens` | 1.084M | 1.067M | 低 1.5% |
| `train/rollout/logprobs_diff` | 0.0300 | 0.0617 | 明显变差 |
| 完整 run HBM 峰值 | max 65496 MB / avg 60883.68 MB | max 65471 MB / avg 61782.90 MB | 峰值均贴近上限 |
| step2 HBM 时间窗峰值 | max 62096 MB / avg 58936.00 MB | max 62676 MB / avg 53925.84 MB | if160 step2 峰值略高 |

结论：`if160_b8` 不是新的最优。它在 step2 端到端仅快约 `0.3%`，属于噪声级收益；同时 `speed_tokens` 低于 `if144_b8`，`logprobs_diff` 从 `0.0300` 劣化到 `0.0617`，且完整 run HBM 峰值仍贴近 `65.5GB`。因此当前推荐配置应回到 `ep128 + if144 + kv64 + ctx8/b8/model8`。

调参经验：

- 不要继续单独提高 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS`。`if160` 已经让 8 个 rollout client 达到 `running_reqs=160`、每 client 20 条，但 step2 几乎没有收益。
- 若继续追求吞吐，应优先尝试在 `if144` 上小步提高 scheduler/model batch，例如 `b12`，并严格以 step2、HBM 完整峰值、`decode_evicted` 和 `logprobs_diff` 判定。
- `XPU_KV_NUM_BLOCKS` 仍应作为第二优先级：当前日志出现 `decode_evicted`，但未看到明确 `kv_cache_swap_in/out`；在没有 KV block 不足实锤前，不应同时增大 KV 与 batch，以免无法归因。
- 完整 run HBM 峰值已经贴近卡上限，任何 `b12` 或 `kv72/80` 实验都必须配套 8 节点 HBM 采样，并至少跑到 `global_step=2`。

### XG ep128 if144_b12 负收益与 ViT RECOMPUTE 发现

在 `if160_b8` 基本持平后，回到 `if144`，把 `scheduler/model/context batch` 从 `8` 提到 `12`。该 case 至少跑到 `global_step=2`，按 step2 与 `if144_b8` 对比。

| 指标 | ep128 + if144_b8 step2 | ep128 + if144_b12 step2 | 变化 |
|---|---:|---:|---:|
| 日志 | `vlm_0617_073951_..._if144_b8_...log` | `vlm_0617_121607_..._if144_b12_...log` | - |
| `train/timing/step` | 1796.14s | 1987.91s | 慢 191.77s / 10.7% |
| `train/timing/data` | 1322.53s | 1453.85s | 慢 131.32s / 9.9% |
| `train/timing/sync_weight` | 183.34s | 205.60s | 慢 22.26s / 12.1% |
| `train/timing/save_model_to_shm` | 176.79s | 215.49s | 慢 38.70s / 21.9% |
| `train/speed_samples` | 0.2851 | 0.2576 | 低 9.6% |
| `train/speed_tokens` | 603.30 | 555.73 | 低 7.9% |
| `response_length.mean` | 1615.71 | 1656.96 | 长 2.6% |
| `train/rollout/logprobs_diff` | 0.0300 | 0.0280 | 基本健康 |
| step2 rollout HBM 峰值 | 57184 MB | 58151 MB | b12 高约 967 MB |

结论：`if144_b12` 不是更优配置。它相比当前最优 `if144_b8` 慢约 `10.7%`，主要退化仍在 rollout/data 阶段，并额外拉高 `sync_weight` 与 `save_model_to_shm`。因此 batch 方向应回退到 `ctx8/b8/model8`。

更重要的是，这份日志暴露了新的性能口径问题：Actor 侧反复打印 `[ViT-Probe] path=RECOMPUTE`，例如 `outputs/logs/vlm_0617_121607_...log:33116` 附近同类日志被聚合为 `repeated 256x across cluster`。同一日志没有看到 `DirectEngineWorker forwarded final response vit_embeds`、`forwarded current vit_embeds`、`VanillaRollout forwarding vit_embeds` 或 `attached response vision_embeddings` 成功信号。这说明 rollout 侧 ViT embedding 没有随 response 进入 RL batch，Actor/ref 仍在重跑 ViT。

调参经验：

- `/dev/shm/vit_cache` 是节点本地共享内存，不是跨 8 节点的全局 cache。它可以保留为单节点或同节点 fallback，但不能作为 23B 8 节点 XG 的主复用机制。
- XG 后续性能验证必须先打开 response-level `vit_embeds` 透传，参考 torch infer 多节点方案：`MAAS_VLM_USE_ROLLOUT_VIT_EMBEDS=1`，让 embedding 跟随 sample/Ray object 走，而不是让 Actor 按 hash 去本地 `/dev/shm` 查。
- 验证成功信号应是 rollout 日志出现 `forwarded final/current vit_embeds` 与 `VanillaRollout forwarding vit_embeds`，Actor 侧出现 `[ViT-Probe] path=BYPASS`，并且不再大规模出现 `path=RECOMPUTE`。
- 下一组参数应回到已知最优 batch 口径，只单独验证 `kv80 + if144 + b8` 与 response-level `vit_embeds`，不要同时再放大 `b12/b16`。

## 2. 优化方案 (Optimization Strategy)

### 方案 A: 先分离稳定性与吞吐口径

- **原理**: XG 当前 `scheduler/model batch=8`、`max_inflight=64` 是稳定性优先口径；直接拿它和 torch infer 的 `2048` 首轮 in-flight 比吞吐，会暴露瓶颈但不能说明 XG 已经调到吞吐最优。
- **实施步骤**:
  ```bash
  # 只做日志复盘，不直接改线上 long run
  rg -n "Training\\. global_step|Inference Proxy Monitor|low-HBM empty_cache|Traceback|RuntimeError" \
    outputs/logs/vlm_0616_065614_xg_23b_rollout_ep64_kv64_tok2048_hccl400c1024_200step_bs64.log
  ```

### 方案 B: XG 吞吐调优优先看有效 in-flight

- **原理**: `data.batch_size=64` 会生成 `64 * 8 = 512` 条 rollout 样本；但实跑只有 64 条并发，导致 8 波推进。下一轮调优应先验证可否在不触发 OOM/HCCL timeout 的前提下提高 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS`、`XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE`、`XLLM_MODEL_MAX_BATCH_SIZE`，而不是再改算法超参 `generation.n` 或训练 batch。
- **注意**: 任何提升并发的尝试都必须和 HBM probe、`npuMalloc retries`、`low-HBM empty_cache`、HCCL timeout、step2+ 稳定性一起看，不能只看 step1 速度。

### 方案 D: EP128 后优先拆分 prefill/context 与 decode batch

- **原理**: `b8 -> b16` 的 step2 对比显示，统一放大 scheduler/context/model batch 会让 rollout/data 变慢，且完整 step HBM 峰值已接近卡上限。因此下一轮不应继续整体加大，而应把 prefill/context 和 decode/model 批大小拆开验证。
- **已验证结果**: `ctx16_b8` step2 为 `1979.38s`，相比 b8 基线 `1974.33s` 略慢；`if96_b8` step2 为 `2279.70s`，相比 b8 基线慢 `15.5%`；`if144_b8` step2 为 `1796.14s`，相比 `if128_b8` 快 `9.0%`；`if160_b8` step2 为 `1790.48s`，相比 `if144_b8` 仅快 `0.3%` 且 `logprobs_diff` 劣化到 `0.0617`；`if144_b12` step2 为 `1987.91s`，相比 `if144_b8` 慢 `10.7%`。当前最优仍是 `if144_b8`。
- **下一轮建议**: 当前最优稳定候选为 `rollout.ep_size=128`、`MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS=144`、`XLLM_MODEL_MAX_BATCH_SIZE=8`、`XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE=8`、`XLLM_SCHEDULER_MAX_SCHEDULE_CONTEXT_BATCH_SIZE=8`。下一组只单独验证 `XPU_KV_NUM_BLOCKS=80` 与 response-level `vit_embeds` 透传；若新日志不能出现 `path=BYPASS`，不要把该 run 纳入纯性能对比。

### 方案 C: 单独压缩同步和量化统计成本

- **原理**: 当前 XG `sync_weight` 比 torch infer 多约 147.55s/step，`collect_quant_amax` 额外约 16.85s/step。若 rollout 生成仍慢，这不是第一瓶颈；但在提高 in-flight 后，这两项会变成端到端 step 的固定税。
- **候选方向**: 复查 W8A8 权重同步路径、`MAAS_RL_SKIP_COLLECT_QUANT_AMAX` 是否能在已验证安全的实验口径下关闭、以及 file/shm sync 的实际路径。没有日志证明前不要把这些改动写成已验证优化。

## 3. 优化效果 (Results)

- 本次不是优化落地，而是两后端并行配置和性能瓶颈沉淀。
- 已确认 torch infer 0606 在 57 个 step 内保持 `logprobs_diff=0.0153~0.0180`，平均 step `1902.80s`，平均 `speed_samples=1.0773`，但 step58 前后出现 `RuntimeError: c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 107020`，随后 raylet dead，不能称为长跑稳定通过。
- 已确认 XG 0616 在前 3 个 step 中 reward 非零，`logprobs_diff` 从 step1 `0.0452` 回到 step2/3 `0.0280/0.0283`，但平均 step `2611.06s`、平均 `speed_samples=0.1984`，吞吐远低于 torch infer。主要性能瓶颈是 rollout in-flight 只有 64、scheduler/model batch 只有 8、W8A8 权重同步和量化统计成本高、低 HBM 工作区压力仍在热路径出现。
- 已确认 0617 `if144_b8` 是当前早期 step2 最优候选；`if160_b8` 只有噪声级收益且精度指标变差，`if144_b12` 明确负收益。0617 b12 日志还证明 XG 旧路径没有把 rollout ViT embedding 透传到 Actor，性能对比前必须先消除 `path=RECOMPUTE`。

## 4. 关联知识

- [[wiki/02_precision/vlm_torch_infer_response_vit_embeds_unified_cache|torch infer VLM rollout response 直传 vit_embeds 统一 ViT cache 方案]]
- [[wiki/00_bugfix/vlm_torch_infer_scheduler_batch_env_override|Bugfix: torch infer VLM 默认 env 覆盖 scheduler/model batch 导致 RL step 劣化]]
- [[wiki/03_tuning/xpu_gpt_w8a8_rl_scheduler_memory_tuning|Tuning: xpu-gpt W8A8 RL scheduler batch 与显存水位调优]]
