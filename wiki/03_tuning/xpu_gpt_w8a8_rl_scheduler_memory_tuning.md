---
title: "Tuning: xpu-gpt W8A8 RL scheduler batch 与显存水位调优"
tags: [status/verified, category/scheduling, category/memory]
last_updated: 2026-06-03
source:
  - outputs/logs/vlm_0602_165949_2614_xg_graph_bs128.log
  - outputs/logs/vlm_0602_165949_2614_xg_graph_bs128_filter.log
  - outputs/logs/vlm_0602_210957_2614_xg_graph_perfopt_bs128.log
  - outputs/logs/vlm_0602_210957_2614_xg_graph_perfopt_bs128_filter.log
confidence: high
status: active
open_questions:
  - "768/0.85 的收益只有约 1%，是否能通过 1024 或更细的 prefill/decode 调度桶进一步放大，需要单独 A/B 验证"
  - "本次对比只覆盖 step 3-5 与 11-15 的 8 个训练 step，不能替代 100+ step 长线吞吐与稳定性结论"
contradictions: []
---

# Profiling & Tuning Report: xpu-gpt W8A8 RL scheduler batch 与显存水位调优

## 1. 瓶颈分析 (Bottleneck Analysis)
- **初始性能**：
  在 VLM 2B5 xpu-gpt W8A8 RL 训练、`bs=128`、`rollout.generation.n=8`、DP16/EP16 配置下，调整前日志 `vlm_0602_165949_2614_xg_graph_bs128.log` 的同口径 8 个 step（3-5 与 11-15）平均 `train/timing/step` 为 **427.329s**，`train/timing/data` 即 rollout 阶段为 **203.945s**。
- **Profiling 现象**：
  端到端 step 中 rollout 约占 47.7%，是最大单项耗时；`old_logprobs`、`update_actor`、`collect_quant_amax` 相对稳定。推理侧调度 batch 与模型 batch 上限从 512 提升后，主要影响 rollout 调度吞吐，而不是 Actor 更新或量化统计。
- **瓶颈类型**：
  主要是 **scheduling + memory**。`gpu_memory_utilization` 提高释放更多 KV/cache 调度空间，`scheduler_max_schedule_batch_size`、`scheduler_max_schedule_context_batch_size`、`model_max_batch_size` 提高允许 xLLM/xpu-gpt 在单轮调度中容纳更多请求。

## 2. 优化方案 (Optimization Strategy)
### 方案 A: 提高 rollout 调度 batch 上限与显存利用水位
- **原理**：
  在 xpu-gpt W8A8 rollout 中，训练侧每个 RL step 会产生 1024 条响应样本。若 scheduler/model batch 上限偏保守，decode/prefill 请求更容易被拆成更多调度轮次，rollout 阶段成为主耗时。提高调度 batch 与显存利用水位可以减少调度排队和拆批开销，但会增加 KV/cache 相关显存压力。
- **实施步骤**：
  在 `run_xg_vlm_2b5.sh` 中将 rollout 配置从 512/0.7 调整为 768/0.85：
  ```bash
  rollout.gpu_memory_utilization=0.85 \
  ++rollout.scheduler_max_schedule_batch_size=768 \
  ++rollout.scheduler_max_schedule_context_batch_size=768 \
  ++rollout.model_max_batch_size=768 \
  ++rollout.inferencer_async_runtime_chunk_size=10240 \
  ```
  对照日志确认 `inferencer_async_runtime_chunk_size=10240` 未变化，因此本次可归因的配置差异是：
  ```text
  gpu_memory_utilization: 0.7 -> 0.85
  scheduler_max_schedule_batch_size: 512 -> 768
  scheduler_max_schedule_context_batch_size: 512 -> 768
  model_max_batch_size: 512 -> 768
  ```

## 3. 优化效果 (Results)
- **调整后统计值**：

  | step范围 | rollout(s) | old_logprobs(s) | offload_weights(s) | save_model_to_shm(s) | sync_weight(s) | update_actor(s) | collect_quant_amax(s) | step all(s) |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 3-5 平均 | 202.379 | 29.928 | 12.829 | 14.500 | 32.023 | 96.816 | 37.857 | 426.466 |
  | 11-15 平均 | 195.488 | 30.260 | 12.633 | 14.517 | 32.208 | 96.457 | 38.254 | 419.951 |

- **同口径 8 step 对比**：

  | 指标 | 调整前 512/0.7 | 调整后 768/0.85 | 差异 |
  |---|---:|---:|---:|
  | rollout(s) | 203.945 | 198.072 | -5.873 (-2.88%) |
  | old_logprobs(s) | 30.046 | 30.135 | +0.090 (+0.30%) |
  | offload_weights(s) | 12.959 | 12.707 | -0.253 (-1.95%) |
  | save_model_to_shm(s) | 14.375 | 14.510 | +0.135 (+0.94%) |
  | sync_weight(s) | 33.313 | 32.138 | -1.174 (-3.53%) |
  | update_actor(s) | 94.899 | 96.591 | +1.692 (+1.78%) |
  | collect_quant_amax(s) | 37.726 | 38.105 | +0.380 (+1.01%) |
  | step all(s) | 427.329 | 422.394 | -4.936 (-1.15%) |
  | speed_samples | 2.398 | 2.426 | +0.028 (+1.16%) |
  | speed_tokens | 5391.956 | 5410.693 | +18.736 (+0.35%) |

- **结论**：
  `512/0.7 -> 768/0.85` 是一个小幅正收益调优。端到端 step 平均缩短 **4.936s**，约 **1.15%**；主要收益来自 rollout 缩短 **5.873s** 与 `sync_weight` 缩短 **1.174s**。`update_actor` 与 `collect_quant_amax` 略慢，抵消了部分收益。
- **副作用/注意事项**：
  - 本次优化没有证明 768 是最优值，只证明在这组日志环境下比 512/0.7 略快。
  - 提高 `gpu_memory_utilization` 会压缩显存余量；长线运行仍需观察 OOM、KV cache eviction、worker restart 与 response length 分布。
  - `vlm_0602_201949_2614_xg_graph_perfopt_bs128.log` 未进入训练 step，失败原因是 worker 启动阶段权重 key/meta 链路异常，不能作为性能对比基线。

## 4. 关联知识
- [[wiki/05_fact/vlm_2b5_rl_rollout_timing|Fact: VLM 2.5B RL NPU Rollout 阶段的真实耗时约束]]
- [[wiki/05_fact/vlm_2b5_rl_rollout_quantization|VLM 2.5B RL 训推转换低精度 Rollout 量化实现 (D_W8A8C8)]]
- [[wiki/05_fact/xpu_gpt_torch_infer_merge_xg_comparison|Fact: xpu_gpt + torch inference 合并后 xg 回归对比结论]]
