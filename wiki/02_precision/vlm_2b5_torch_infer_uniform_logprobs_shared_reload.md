---
title: "Precision: VLM 2B5 torch infer shared language reload 缺失导致均匀 logprobs"
tags: [status/active, category/diff, category/torch-infer, bf16, vlm, m8-2b5]
last_updated: 2026-06-25
source:
  - outputs/logs/vlm_0625_095043_mathverse23b_bf16_skip_langreload_0625_clean18_bs64.log
  - outputs/logs/vlm_0625_145529_24189_torchinfer_bs128.log
  - outputs/logs/vlm_0625_152731_24189_torchinfer_bs128.log
  - outputs/logs/vlm_0625_155139_24189_torchinfer_bs128.log
  - outputs/logs/vlm_0625_155759_24189_torchinfer_bs128.log
  - 3rdparty/xllm/xllm/backend/inferencer/torch/torch_inferencer.py
  - 3rdparty/xllm/xllm/backend/inferencer/torch/utils/weight_loader.py
confidence: high
status: active
open_questions:
  - "当前只验证到 2B5 torch infer step1 恢复健康量级，不能替代 step5、70+ step 或 200 step 长线漂移验证。"
  - "0624 23B torch infer 纯文本曾有 skip shared language reload 后 logprobs_diff 降到 0.0097 的阶段性证据；2B5 与 23B 的 shared SeedVL reload 语义不能互相外推，后续需要按模型/weight mapping 口径做 guard 或分别复验。"
contradictions: []
---

# Precision Alignment Report: VLM 2B5 torch infer shared language reload 缺失导致均匀 logprobs

## 1. 差异现象 (Deviation Symptoms)

- **现象**: 2026-06-25 下午合并 23B XG/torch infer precision 相关代码后，24189 单机 2B5 torch infer BF16 回归从健康量级退化为近似均匀 logprobs：`train/rollout/logprobs_diff≈4.0`、`rollout/ppl=152894`、reward 全 0。
- **数据类型 / 运行模式**: M8 2B5 VLM，Actor/ref BF16，rollout 为 `XLLM_MODEL_CLS=torch` 的 torch infer DirectEngine，单机 16 NPU，`BATCH_SIZE=128`，`train/batch_size=1024`。
- **关键症状**:
  - `rollout/ppl=152894` 接近 vocab size `155136`，说明 rollout logprob 分布近似均匀。
  - bad run 的 `_update_weights` 只出现 `shared_vision_model` reload，没有 `shared_language_model` reload。
  - 恢复 `shared_language_model` reload 后，step1 回到 `logprobs_diff=0.0144124`、`rollout/ppl=1.07827`、reward 非零。

## 2. 时间线与指标对照

### 2.1 上午 torch infer 健康证据

23B MathVerse torch infer run 仍健康：

```text
outputs/logs/vlm_0625_095043_mathverse23b_bf16_skip_langreload_0625_clean18_bs64.log
```

关键指标：

| step | logprobs_corr | logprobs_diff | rollout/ppl | reward.mean |
|---:|---:|---:|---:|---:|
| 1 | 0.995755 | 0.018415 | 1.26219 | 0.0175781 |
| 2 | 0.996030 | 0.0177421 | 1.26087 | 0.0507812 |

这说明上午 torch infer BF16 路径本身仍可达到 `~0.018` 量级。

### 2.2 下午 2B5 bad run：均匀 logprobs

bad run 1：

```text
outputs/logs/vlm_0625_145529_24189_torchinfer_bs128.log
```

| step | logprobs_corr | logprobs_diff | rollout/ppl | reward.mean |
|---:|---:|---:|---:|---:|
| 1 | 0 | 3.99452 | 152894 | 0 |
| 2 | 0 | 4.10988 | 152894 | 0 |

bad run 2：

```text
outputs/logs/vlm_0625_152731_24189_torchinfer_bs128.log
```

step1:

```text
logprobs_corr=0
logprobs_diff=4.03934
rollout/ppl=152894
reward.mean=0
```

这轮曾调整/核对 TP/EP 相关口径，但症状仍是 `~4.0` 和 `rollout/ppl=152894`，因此本次 2B5 异常不应优先归因于 TP/EP 配置。

### 2.3 日志签名：缺失 language reload

bad run 中 update_weights 只看到 shared vision reload：

```text
_update_weights.before_load shared_vision_model
_update_weights.after_load shared_vision_model
```

没有出现：

```text
_update_weights.before_load shared_language_model
_update_weights.after_load shared_language_model
```

这与被删除代码中的注释一致：如果 shared VLM 分支只 reload vision wrapper 而不 reload language model，语言参数可能保持 empty/mock 状态，rollout 输出会接近均匀 logprobs。

## 3. 代码回归点

当天 xLLM 相关提交：

```text
e02c4935c Fix 23B torch and XG precision rollout paths
```

该提交在 `torch_inferencer.py::_update_weights()` 的 shared VLM 分支中删除了 language model 二次 reload，留下：

```python
if vision_model is not None and getattr(self, "_vl_shared_language_model", False):
    self.load_model_weights(vision_model, state_dict=state_dict, kernel_backend=self.kernel_backend)
    return
```

对 2B5 回归而言，这个裸 `return` 使 `self.model` 的 language mapping 不再执行；日志与指标均指向这个变更导致 rollout language weights 未正确加载。

同一批变更还把 `weight_loader.py::preprocess_mappings()` 的 `expert_tp_size/expert_tp_rank` 参数从函数签名和调用中删除，但 mapping 配置仍可能引用 `expert_tp_size`。恢复 language reload 后，第一次复验立即暴露：

```text
outputs/logs/vlm_0625_155139_24189_torchinfer_bs128.log

_update_weights.before_load shared_language_model
NameError: name 'expert_tp_size' is not defined
```

因此本次 2B5 回归包含两个连续问题：

1. `torch_inferencer.py` shared VLM 分支缺失 language reload，导致均匀 logprobs。
2. `weight_loader.py` 普通 mapping 预处理作用域缺少 `expert_tp_size/expert_tp_rank`，导致恢复 language reload 后直接 NameError。

## 4. 修复方案 (Alignment Fix)

### 4.1 恢复 2B5 所需的 shared language reload

修复口径：

```python
if vision_model is not None and getattr(self, "_vl_shared_language_model", False):
    log_update_weight_memory("_update_weights.before_load shared_vision_model")
    self.load_model_weights(vision_model, state_dict=state_dict, kernel_backend=self.kernel_backend)
    log_update_weight_memory("_update_weights.after_load shared_vision_model")

    log_update_weight_memory("_update_weights.before_load shared_language_model")
    self.load_model_weights(self.model, state_dict=state_dict, kernel_backend=self.kernel_backend)
    log_update_weight_memory("_update_weights.after_load shared_language_model")
    return
```

### 4.2 恢复 mapping eval 需要的 expert TP 变量

修复口径：

```python
if parallel_info.use_ep:
    expert_tp_size = 1
    expert_tp_rank = 0
else:
    expert_tp_size = dp_size * tp_size
    expert_tp_rank = tp_rank + tp_size * dp_rank

mappings = preprocess_mappings(
    mappings,
    hf_model_config,
    tp_size,
    tp_rank,
    expert_tp_size,
    expert_tp_rank,
)
```

并恢复函数签名：

```python
def preprocess_mappings(
    mappings,
    config,
    tp_size,
    tp_rank,
    expert_tp_size=1,
    expert_tp_rank=0,
):
```

该修改不是把普通权重强制改成 expert 切片；它只是让 mapping 表达式里引用到的变量有定义。expert mappings 仍由 `preprocess_expert_mappings()` 单独处理。

## 5. 对齐验证 (Validation)

修复验证日志：

```text
outputs/logs/vlm_0625_155759_24189_torchinfer_bs128.log
```

runtime package 签名：

```text
_update_weights.before_load shared_vision_model
_update_weights.after_load shared_vision_model
_update_weights.before_load shared_language_model
_update_weights.after_load shared_language_model
```

step1 指标：

```text
train/rollout/logprobs_corr=0.977934
train/rollout/logprobs_diff=0.0144124
train/rollout/ppl=1.07827
train/reward/final_rewards.mean=0.166016
train/actor/ppl=1.08127
```

阶段结论：

- 本次 2B5 torch infer 均匀 logprobs 回归已在 step1 被打回健康量级。
- `rollout/ppl` 从 `152894` 恢复到 `1.07827`，reward 从全 0 恢复到 `0.166016`，说明 language weights reload 问题被直接击中。
- 该证据只覆盖 early-step 回归修复，不能替代 70+ / 200 step 长线漂移验证。

## 6. 边界与后续风险

- **不要把 2B5 结论无条件外推到 23B**: 0624 23B torch infer 纯文本曾有 skip shared language reload 后 `logprobs_diff≈0.0097` 的阶段性证据；本次 2B5 则相反，缺失 language reload 会直接均匀化。后续若要合并长期修复，需要按模型 size、mapping 文件或 shared wrapper 实际覆盖范围做精确 guard，或者重新跑 23B 回归。
- **不要把本次问题归因到 ViT**: bad run 是 rollout language logprob 接近 vocab-size 均匀分布；修复点在 torch infer update_weights / weight_loader，和 `/dev/shm/vit_cache`、response-level `vit_embeds` 不是同一类问题。
- **不要只看 step1 宣告终局通过**: 当前修复目标只是恢复 2B5 torch infer early-step 健康量级；仍需至少 step1-step5 稳定在 `~0.014-0.02`，并进入 70+ / 200 step 长线漂移验证。

## 7. 关联知识

- [[wiki/02_precision/vlm_23b_torch_infer_bf16_logpdiff_8x_boundary|VLM 23B torch infer BF16 纯文本 logprobs_diff 8.x 定界]]
- [[wiki/02_precision/vlm_torch_infer_frozen_vlm_weight_sync_reward|torch infer frozen VLM 权重未同步导致 reward 低于合并前]]
- [[wiki/02_precision/vlm_torch_infer_response_vit_embeds_unified_cache|torch infer VLM rollout response 直传 vit_embeds 统一 ViT cache 方案]]
- [[wiki/05_fact/torch_infer_merge_debug_checklist|xpu_gpt + torch infer 合并验证排查清单]]
