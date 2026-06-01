---
title: "Fact: xpu_gpt + torch infer 合并验证排查清单"
tags: [fact, vlm, rl, torch-infer, xllm-direct, npu, status/active]
last_updated: 2026-05-29
source:
  - "maas_engine_vlm2b5 torch infer merge debug session, 2026-05-29"
  - "outputs/logs/vlm_0529_142128_torch_infer_2b5_bs2.log"
  - "outputs/logs/vlm_0529_143654_torch_infer_2b5_bs2.log"
  - "maas_engine_torch_infer/outputs/logs/vlm_0528_002344.log"
---

# xpu_gpt + torch infer 合并验证排查清单

本文记录 xpu_gpt 后端与 torch inference (`xllm_direct`) 后端合并验证时新增的排查经验。这里沉淀的是事实约束与诊断顺序，不把早期 step 指标直接定性为最终修复。

## 1. OOM 与 logprobs_diff 必须拆开判断

同一次 RL 验证中可能先遇到 `NPU out of memory`，后续又出现 `logprobs_diff` 异常。两者不能自动合并为同一个问题。

判断 OOM 是否仍存在，必须看最新日志的首个异常和显存探针：

- 首个错误栈：`Traceback`、`RuntimeError`、`NPU out of memory`。
- Actor 侧：`optimizer_step`、`save_model_to_shm.after_final_empty_cache` 前后的 NPU `allocated/reserved` 与 device `used/total`。
- Rollout 侧：`after sync_weight.update_weights` 后的 device `used/total`。

案例中 `vlm_0529_143654_torch_infer_2b5_bs2.log` 前 2 step 无 OOM，Rollout 权重同步后约 `50.4/61.3GB`，因此主线应从 OOM 转向精度一致性，而不是继续盲目调小 batch 或 KV 参数。

## 2. runtime_env 相同不代表代码包相同

Ray job 的 `runtime_env_*.yaml` 只记录环境变量快照，不足以证明两次运行使用完全相同代码。Ray submission 还会上传 `working_dir` zip 包，日志里的 `runtime_env.working_dir` 形如：

```text
gcs://_ray_pkg_*.zip
```

如果两次运行的 runtime env SHA256 一致，但 `logprobs_diff` 或 Actor/Rollout PPL 表现差异巨大，必须继续核对：

- 最新本地代码是否已经同步到远端。
- Ray job list 中 `runtime_env.working_dir` 的包 ID。
- 关键源码行号是否变化，例如日志中的 `actor.py:<lineno>`。

案例中 `0529_142128` 与 `0529_143654` 的 runtime env SHA256 一致，但后者使用更新后的代码包，前 2 step `logprobs_diff` 从 `0.675873/0.393848` 回落到 `0.0241889/0.0143639`。因此不能只凭 runtime env 判断实验等价。

## 3. torch infer 的 `xpu` kernel backend 不是 xperf_gpt

torch inference 路径中：

```text
XLLM_MODEL_CLS=torch
XLLM_INFERENCER_KERNEL_BACKEND=xpu
XPU_BACKEND_TYPE=npu
```

这里的 `xpu` 是 xLLM torch inferencer 的 kernel backend registry 名，用于选择 `3rdparty/xllm/xllm/backend/inferencer/torch/layers/xpu/` 下的 NPU 适配层。它不是 `xperf_gpt`，也不应把 `xperf_gpt` 当作本路径必须验证的组件。

如果启动报：

```text
AssertionError: Unknown kernel backend: xpu
```

优先检查 `torch/layers/__init__.py` 的 backend registry 是否注册了 `xpu`，以及 `layers/xpu/` 模块是否被同步到远端。不要去改 xperf_gpt。

## 4. 模型路径与配置口径要从日志确认

torch infer 源分支成功日志使用过 `/data01/sft_output/`，其 `ModelSpec` 配置里可见 `max_position_embeddings=524288`、`rope_scale=400`。合并分支脚本默认可能使用 `/data01/model_weights/2b5_saved_model`，其 xLLM `model_cfg` 里可见 `max_position_embeddings=32768`、`rope_scaling.factor=400`。

这类差异不应直接定性为问题根因，但必须作为精度排查前置事实记录：

- 以日志中的 `ModelSpec __init__ model_dir=...` 和 xLLM `model_cfg` 为准。
- 同时检查 Actor 与 Rollout 是否加载同一口径路径。
- 对比源分支时，不只比较 shell 脚本，还要比较实际运行日志里的模型配置。

## 5. 早期 step 只能作为阶段性信号

torch infer 源分支 `vlm_0528_002344.log` 里，bsz=128 的 41 step 可稳定在 `logprobs_diff=0.015~0.017`。合并分支 `vlm_0529_143654_torch_infer_2b5_bs2.log` 前 2 step 回到 `0.014~0.024`，只能说明短程指标恢复到健康区间，不能替代长跑验证。

继续验证时至少记录：

- `global_step` 连续增长。
- `train/rollout/logprobs_diff` 是否保持 0.01~0.03 量级。
- `actor_ppl` 与 `rollout_ppl` 是否同步。
- `train/reward/final_rewards.mean` 是否非全零。
- 日志尾部是否无 Traceback/Raylet termination/OOM。

## Related

- [[wiki/00_bugfix/npu_empty_virt_addr_cache_oom|Bugfix: empty_virt_addr_cache 导致 NPU 物理显存泄露引发 OOM]]
- [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
- [[wiki/02_precision/vlm_rl_actor_rope_logprobs_diff|VLM RL 长序列 RoPE 偏移导致 logprobs_diff 发散]]
