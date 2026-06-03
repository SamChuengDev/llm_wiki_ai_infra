---
title: "Fact: VLM 2B5 xpu_gpt 32rank Actor 与 Rollout 部署形态"
tags: [status/stable, category/fact, vlm, xpu_gpt, ray, fsdp]
last_updated: 2026-06-03
source: [run_xg_vlm_2b5.sh, runtime_env_xg_32rank.yaml, ray_xg_32rank.sh, maas_finetune/trainer/rl/dist_utils.py, maas_finetune/trainer/rl/rollout/ray_driver.py, outputs/logs/vlm_0603_130806_xg_2nodes_2b5_bs128.log]
confidence: high
status: active
open_questions:
  - "3 节点、4 节点或更多节点扩展时，rollout replica 数按节点数线性增加的规则仍需对应规模实测确认。"
contradictions: []
---

# Fact: VLM 2B5 xpu_gpt 32rank Actor 与 Rollout 部署形态

## 1. 核心事实 (Core Fact)

在 VLM 2B5 xpu_gpt RL 双节点 32 NPU 验证口径下，Actor 与 Rollout 的并行拓扑不是同一种 DP 语义：

- **Rollout**: `rollout.tp_size=1`, `rollout.ep_size=16`，每个 xLLM rollout replica 占用 `1 * 16 = 16` 张 NPU。双节点各 16 NPU 时，Ray 会放置 2 个 rollout replica，因此 rollout 层面可理解为 **DP=2，每节点 1 个 16-rank 副本**。
- **Actor**: `actor.fsdp_size=-1`, `actor.ep_size=2`, `actor.sp_size=4`，在 32 rank 训练 world 内形成 **1 个跨 32 rank 的 FSDP 训练模型**，不是每节点 1 个完整 actor 副本。按代码公式，模型副本维度 `dp_replicate_size=1`，数据 dispatch 维度为 `world_size / sp_size = 32 / 4 = 8`。

已验证日志 `outputs/logs/vlm_0603_130806_xg_2nodes_2b5_bs128.log` 显示该配置在 2450 + 26100 双节点 32 NPU 上连续完成 `global_step=10`，未见 `Traceback`、`RuntimeError`、`NPU out of memory`、`Raylet is terminated`、`RPCError`、`GLOBAL MISS` 等 fatal 信号。

## 2. 设计背景 (Background & Rationale)

Rollout 与 Actor 使用不同并行策略，是因为二者职责不同：

- Rollout 负责 xLLM W8A8 推理生成。当前 xpu_gpt VLM 2B5 推理基线是单节点本地 16 rank，主 LLM 使用 `EP=16, TP=1`，伴随 Companion ViT 通过本地 IPC 提供图像 embedding。因此多节点 rollout 扩展采用“多个本地 16-rank xLLM replica”的方式，而不是一个跨节点 32-rank xLLM replica。
- Actor 负责训练更新。`fsdp_size=-1` 在训练侧表示 FSDP shard 覆盖完整 world size；结合 `sp_size=4` 后，训练数据维度为 8，但这些数据维度全部参与同一个 FSDP shard 结构，不形成多个完整模型副本。
- 多节点 rollout 必须启用 `MAAS_XG_ROLLOUT_USE_NODE_IP=1`，让每个 Ray 节点上的 xLLM replica 使用本机 IP 作为本地 torch distributed rendezvous 地址，避免 worker 节点继承 head 节点 IP 导致连接错配。

## 3. 应用场景与约束 (Constraints)

- 该事实适用于 VLM 2B5 xpu_gpt RL 的双节点 32 NPU 验证口径，配置为 `actor.fsdp_size=-1`, `actor.ep_size=2`, `actor.sp_size=4`, `rollout.ep_size=16`, `rollout.tp_size=1`。
- Rollout 的 DP 扩展来自 Ray alive NPU 节点数与每个 replica 占卡数：`replicas_per_node = node_npu_count / (rollout.tp_size * rollout.ep_size)`。双节点每节点 16 NPU 时为每节点 1 个 replica，总 rollout replica 数为 2。
- Actor 的“模型副本 DP”与“数据 dispatch DP”必须区分：`fsdp_size=-1` 使模型副本维度为 1，但 `sp_size=4` 使 32 rank 下数据 dispatch 维度为 8。
- 若扩展到 3 节点、4 节点或更多节点，并保持每节点 16 NPU、`rollout.ep_size=16`、`rollout.tp_size=1`，预期 rollout replica 数分别为 3、4 或更多；但对应规模仍需实测确认，不应直接视为已验证事实。

## 4. 相关配置或引用 (References)

```bash
# run_xg_vlm_2b5.sh
actor.fsdp_size=-1
actor.ep_size=2
actor.sp_size=4

++rollout.ep_size=16
rollout.tp_size=1
```

```yaml
# runtime_env_xg_32rank.yaml
"MAAS_XG_ROLLOUT_USE_NODE_IP": "1"
"XLLM_INFERENCER_torch_distributed_addr": "${XLLM_INFER_IP}"
```

```python
# maas_finetune/trainer/rl/dist_utils.py
world_size = dist.get_world_size()
dp_size = world_size // sp_size
fsdp_size = fsdp_size if fsdp_size > 0 else world_size
fsdp_size = fsdp_size // sp_size
dp_replicate_size = dp_size // fsdp_size
dp_shard_size = fsdp_size

# 32rank, sp=4, fsdp_size=-1:
# dp_size=8, fsdp_size=8, dp_replicate_size=1, dp_shard_size=8
```

```text
# outputs/logs/vlm_0603_130806_xg_2nodes_2b5_bs128.log
step=1  reward_mean=0.166992  logprobs_diff=0.039705
step=2  reward_mean=0.147461  logprobs_diff=0.0285343
step=3  reward_mean=0.140625  logprobs_diff=0.0271547
step=4  reward_mean=0.174805  logprobs_diff=0.0289488
step=5  reward_mean=0.148438  logprobs_diff=0.0281688
step=6  reward_mean=0.172852  logprobs_diff=0.0304108
step=7  reward_mean=0.116211  logprobs_diff=0.0286119
step=8  reward_mean=0.214844  logprobs_diff=0.0293454
step=9  reward_mean=0.126953  logprobs_diff=0.0292935
step=10 reward_mean=0.201172  logprobs_diff=0.0288541
```

## 5. 沉淀与关联

- 关联知识点: [[wiki/05_fact/vlm_2b5_vit_llm_deployment_architecture|Fact: VLM 2B5 VIT与LLM的混合部署架构约束]]
- 关联知识点: [[wiki/05_fact/vlm_2b5_inference_service_golden_config|VLM 2B5 推理服务化部署黄金配置]]
- 关联知识点: [[wiki/05_fact/vlm_2b5_rl_rollout_quantization|VLM 2.5B RL 训推转换低精度 Rollout 量化实现]]
