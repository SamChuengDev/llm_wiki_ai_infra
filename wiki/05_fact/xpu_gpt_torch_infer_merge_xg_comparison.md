---
title: "Fact: xpu_gpt + torch inference 合并后 xg 回归对比结论"
tags: [fact, vlm, rl, xpu-gpt, torch-infer, merge, precision, performance, status/active]
last_updated: 2026-06-02
source:
  - "outputs/logs/vlm_0525_220913_24167_thnk_parser_enabled_fix_prefix_cache_bs128.log"
  - "outputs/logs/vlm_0525_220913_24167_thnk_parser_enabled_fix_prefix_cache_bs128_443steps.log"
  - "outputs/run_time_envs/runtime_env_0525_220913.yaml"
  - "outputs/logs/vlm_0601_165516_2450_after_merge_bs128.log"
  - "outputs/logs/vlm_0601_165516_2450_after_merge_bs128_filter.log"
  - "outputs/run_time_envs/runtime_env_0601_165516.yaml"
  - "2026-06-02 xpu_gpt merge validation analysis session"
confidence: high
status: active
open_questions:
  - "CPU_AFFINITY_CONF 从 2 变为 1,npu_affine:1 对 step timing 的实际贡献尚未通过 A/B 实验单独量化。"
contradictions: []
---

# Fact: xpu_gpt + torch inference 合并后 xg 回归对比结论

## 1. 核心事实 (Core Fact)

2026-06-02 对比 xpu_gpt 合并前后两次 VLM 2B5 RL 验证时，必须以完整日志中的实际运行配置为准，而不是当前工作区脚本或 filter 日志。完整日志结构化对比显示：

- Hydra 训练、Actor、Ref、Data、Rollout 参数一致，唯一差异是 trajectory 输出文件时间戳。
- LLM target 权重路径文本不同，但最终解析到同一份 W8A8 文件，`cmp` 内容一致。
- Companion ViT checkpoint 路径文本不同，但 `cmp` 内容一致。
- 环境变量存在少量差异，包括 `CPU_AFFINITY_CONF`、host 地址、0525 额外 logprobs debug 变量、0525 额外 draft/spec 文本变量。
- 性能上合并后单个 RL step 平均减少约 50~60s，分项主要来自 `sync_weight`、`update_actor`、`offload_weights` 和 `data` 的共同下降；`CPU_AFFINITY_CONF` 从 `2` 变为 `1,npu_affine:1` 是候选解释之一，但尚未单独 A/B 定量。
- 投机推理在两次 xg RL 路径中均未被实锤启用；配置文本出现 `eagle_llm` 或 draft model 不等于推理实际生效，不能作为本次精度差异的解释。

## 2. 对比对象与证据来源

合并前完整日志在 24167 远端保留，本地只同步了 filter/指标截取：

```text
before full log:
24167:/data01/zcs/maas_engine_vlm2b5/outputs/logs/vlm_0525_220913_24167_thnk_parser_enabled_fix_prefix_cache_bs128.log

before local metric slices:
outputs/logs/vlm_0525_220913_24167_thnk_parser_enabled_fix_prefix_cache_bs128_443steps.log
outputs/logs/vlm_0525_220913_24167_thnk_parser_enabled_fix_prefix_cache_bs128_filter.log

after full log:
outputs/logs/vlm_0601_165516_2450_after_merge_bs128.log

after local metric slice:
outputs/logs/vlm_0601_165516_2450_after_merge_bs128_filter.log
```

不要用当前工作区里的 `run_xg_vlm_2b5.sh` 反推 `0601_165516` 的真实运行参数。当前脚本内容已经与该时间戳实验不一致，真实口径应从完整日志开头打印的 Python dict 配置、`debug: environ(...)`、xLLM `model_config` / launcher 配置中提取。

## 3. Hydra 配置对比结论

从两个完整日志开头的 `{'actor': ...}` 配置块做结构化 flatten diff，结果为：

```text
HYDRA_DIFF_COUNT=1

trainer.trajectory_logger.stream:
  before: /data01/zcs/maas_engine_vlm2b5/outputs/rollout//trajectory_vlm_0525_220913.jsonl
  after : /data01/zcs/maas_engine_vlm2b5/outputs/rollout//trajectory_vlm_0601_165516.jsonl
```

关键运行参数一致：

```yaml
data:
  batch_size: 128
  max_length: 12288
  max_concurrency: 2048
  train: /data01/datasets/MathVerse/data_with_think_sp_train.parquet

rollout:
  engine: xllm
  enable_vlm: true
  enable_reasoning: true
  thinking_parser.enabled: true
  model.path: /share_data/2b5_saved_model_for_xg/
  quant_mode: W8A8
  ep_size: 16
  tp_size: 1
  generation.n: 8
  generation.max_new_tokens: 4096
  generation.temperature: 1.0
  generation.top_p: 1.0
  model_max_batch_size: 512
  scheduler_max_schedule_batch_size: 512
  scheduler_max_schedule_context_batch_size: 512
  inferencer_async_runtime_chunk_size: 10240

actor:
  model.path: /share_data/2b5_saved_model_torchinfer/
  quant.prequant_qscale_path: /share_data/2b5_saved_model_for_xg/saved_model/W8A8/model.safetensors
  dynbsz_max_length: 32768
  sp_size: 4
  ep_size: 2

ref:
  model.path: /share_data/2b5_saved_model_torchinfer/
  dynbsz_max_length: 65536
  sp_size: 4
  ep_size: 2

trainer:
  total_training_steps: 500
  tracking_name: ypq_test
```

## 4. 环境变量对比结论

完整日志中的实际 `debug: environ(...)` 对比显示，主 xLLM / Companion 关键环境变量大体一致：

```text
MAAS_MODEL_DIR:
  before: /share_data/2b5_saved_model_for_xg/
  after : /share_data/2b5_saved_model_for_xg/

COMPANION_MAAS_MODEL_DIR:
  before: /share_data/2b5_saved_model_for_xg/
  after : /share_data/2b5_saved_model_for_xg/

COMPANION_VIT_CONFIG_PATH:
  before: /share_data/2b5_saved_model_for_xg/vit_config.json
  after : /share_data/2b5_saved_model_for_xg/vit_config.json

XLLM_MODEL_CLS: xpu
XLLM_INFERENCER_CLS_TYPE: xpu
XLLM_PROCESSOR_INPUT: ark_vlm_dialog
COMPANION_XLLM_PROCESSOR_INPUT: ark_vlm_dialog
XPU_KV_NUM_BLOCKS: 1280
XLLM_MODEL_NUM_BLOCKS: 1280
XLLM_MODEL_MAX_LENGTH: 8192
XLLM_MAX_NEW_TOKENS: 2400
MAX_PREFILL_BATCH: 64
XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE: 64
XLLM_SCHEDULER_MAX_SCHEDULE_CONTEXT_BATCH_SIZE: 64
```

确认存在的环境差异：

```text
XLLM_INFERENCER_torch_distributed_addr:
  before: 10.101.24.167
  after : 10.101.24.50

CPU_AFFINITY_CONF:
  before: 2
  after : 1,npu_affine:1

0525 only:
  MAAS_LOGPROBS_DIFF_DEBUG_DIR=/data01/zcs/maas_engine_vlm2b5/outputs/debug
  MAAS_LOGPROBS_DIFF_DEBUG_DUMP=1
  MAAS_LOGPROBS_DIFF_DEBUG_STEPS=59
  MAAS_LOGPROBS_DIFF_DEBUG_THRESHOLD=0
  MAAS_LOGPROBS_DIFF_DEBUG_TOPK_SAMPLES=8
  MAAS_LOGPROBS_DIFF_DEBUG_TOPK_TOKENS=8
  XLLM_DRAFT_MODEL_MODEL_DIR=/data01/zcs/maas_engine_vlm2b5/runtime/draft_meta
  XLLM_INFERENCER_EAGLE_KV_CACHE_SHARE_LAYER_INDEX=21
```

`0601_165516` 完整日志中还出现：

```text
Warning: dcmi_get_affinity_cpu_info_by_device_id is not supported.
The npu_affine configuration of CPU_AFFINITY_CONF will be disabled.
```

这说明 `CPU_AFFINITY_CONF` 是一个真实环境差异，但其对性能差异的贡献需要单独 A/B 验证，不能仅凭该 warning 定性为根因。

## 5. 权重与最终模型配置对比

xLLM worker 的 `model_config model_conf` 对比显示，主 LLM target 最终模型配置只在权重路径文本上不同：

```text
before:
  vanilla_checkpoint_path=/share_data/2b5_saved_model_for_xg/saved_model/global_step_0/model.safetensors
  xpu_model_path=/share_data/2b5_saved_model_for_xg/saved_model/global_step_0/model.safetensors

after:
  vanilla_checkpoint_path=/share_data/2b5_saved_model_for_xg/saved_model/W8A8/model.safetensors
  xpu_model_path=/share_data/2b5_saved_model_for_xg/saved_model/W8A8/model.safetensors

cmp_llm_exit=0
```

这两个 LLM 文件路径均解析到同一份 W8A8 权重内容：

```text
/share_data/2b5_saved_model_for_xg/saved_model/global_step_0/model.safetensors
  -> /share_data/2b5_saved_model_torchinfer/saved_model/W8A8/model.safetensors

/share_data/2b5_saved_model_for_xg/saved_model/W8A8/model.safetensors
  -> /share_data/2b5_saved_model_torchinfer/saved_model/W8A8/model.safetensors
```

Companion ViT 路径文本也不同，但内容一致：

```text
before:
  vanilla_checkpoint_path=/share_data/2b5_saved_model_for_xg/saved_model/vit.safetensors

after:
  vanilla_checkpoint_path=/share_data/2b5_saved_model_for_xg/saved_model/global_step_0/vit.safetensors

cmp_vit_exit=0
```

因此，合并前后 xg 回归对比中，不能把“模型权重不同”作为已证实的精度或性能差异原因。

## 6. 精度与性能指标观察

按前 126 个共同 step 对齐比较 `0525_220913` 与 `0601_165516`：

```text
train/rollout/logprobs_diff:
  after - before mean: +0.0013407
  median: +0.00100885
  positive: 126/126
  relative mean: +5.856%

train/rollout/logprobs_corr:
  after - before mean: -0.0156137
  positive: 2/126

train/timing/step:
  after - before mean: -63.1755s
  positive: 0/126

train/timing/sync_weight:
  after - before mean: -30.9158s

train/timing/update_actor:
  after - before mean: -26.752s

train/timing/offload_weights:
  after - before mean: -7.07616s

train/timing/data:
  after - before mean: -8.77326s

train/reward/final_rewards.mean:
  after - before mean: about -0.000263

train/actor/entropy:
  after - before mean: about -0.0000021

train/rollout/response_length.mean:
  after - before mean: about -2.22 tokens
```

从数量级看，合并后单个 RL step 稳态大约减少 50~60s，和 126 step 对齐均值 `-63.1755s` 一致。该性能改善不是单一分项贡献：`sync_weight` 下降约 31s，`update_actor` 下降约 27s，`offload_weights` 和 `data` 也有较小下降。

当前已知的非代码环境差异中，最值得纳入性能解释候选的是 CPU 绑核设置：

```text
CPU_AFFINITY_CONF:
  before: 2
  after : 1,npu_affine:1
```

但 `0601_165516` 日志同时提示 `npu_affine` 不被支持并被禁用，因此这只能作为性能差异的候选假设。若要定量确认，需要在相同代码、相同模型、相同数据口径下只切换 `CPU_AFFINITY_CONF` 做 A/B。

聚合均值对比：

```text
0525 before, first 126 steps:
  logprobs_diff.mean: 0.0229987
  logprobs_corr.mean: 0.937227
  timing/step.mean: 548.004s
  sync_weight.mean: 56.2087s
  update_actor.mean: 120.758s
  offload_weights.mean: 18.9411s
  data.mean: 277.938s
  reward.mean.mean: 0.527661
  entropy.mean: 0.063357
  response_length.mean: 1364.46

0601 after, first 120 filter steps:
  logprobs_diff.mean: 0.0243194
  logprobs_corr.mean: 0.923814
  timing/step.mean: 485.803s
  sync_weight.mean: 25.3057s
  update_actor.mean: 94.3234s
  offload_weights.mean: 11.8604s
  data.mean: 269.541s
  reward.mean.mean: 0.52356
  entropy.mean: 0.063916
  response_length.mean: 1369.15
```

`0601_165516` 完整日志曾观察到：

```text
global_step=126
train/rollout/logprobs_diff=0.0296265
train/rollout/logprobs_corr=0.749243
train/reward/final_rewards.mean=0.563477
train/timing/step=476.835
```

这些是回归对比事实，不构成“长线精度已通过”的结论。终极精度仍需遵守 70+ step / 200 step 长跑标准。

## 7. 应用场景与约束 (Constraints)

- 比较合并前后配置时，优先使用完整日志里的 Hydra dict、`debug: environ(...)` 和 xLLM `model_config`，不要只看 filter 日志或当前脚本。
- `runtime_env_*.yaml` 可作为辅助，但完整日志中的实际 `environ` 更接近进程真实状态；例如 `0601_165516` 的 `runtime_env` 文本中曾缺少部分模型目录变量，但完整日志显示运行时已补齐。
- 权重路径不同必须继续检查 symlink 和 `cmp`，不能把路径字符串不同等同于权重内容不同。
- 投机推理相关字段只能在 draft forward/worker 调用、accept/reject 统计或等价证据存在时才算生效；否则按未生效处理。
- 性能变化需拆解到 `sync_weight`、`update_actor`、`offload_weights`、`data` 等分项，不应只看总 step time。

## 8. 沉淀与关联

- 关联知识点: [[wiki/05_fact/torch_infer_merge_debug_checklist|Fact: xpu_gpt + torch infer 合并验证排查清单]]
- 关联知识点: [[wiki/05_fact/vlm_2b5_rl_rollout_quantization|VLM 2.5B RL 训推转换低精度 Rollout 量化实现]]
- 关联知识点: [[wiki/05_fact/vlm_2b5_inference_service_golden_config|VLM 2B5 推理服务化部署黄金配置]]
- 关联知识点: [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
- 关联知识点: [[wiki/00_bugfix/vlm_2b5_vit_checkpoint_fallback|M8 2B5 VLM Companion ViT 权重路径 fallback 缺失导致启动失败]]
