---
title: "Precision: VLM 23B torch infer BF16 纯文本 logprobs_diff 8.x 定界"
tags: [status/active, category/diff, category/torch-infer, bf16, vlm, m8-23b]
last_updated: 2026-06-24
source:
  - outputs/logs/vlm_0608_162254_torch_infer_23b_full_8nodes_unified_vit_noshm_clean_retry_bs256_bs256.log
  - outputs/logs/vlm_0624_143127_torch_infer_23b_full_8nodes_fixsharedload_0624_precision_bs64.log
  - outputs/logs/vlm_0624_160814_torch_infer_23b_text_gsm8k_boundary_0624_precision_bs64.log
  - outputs/logs/vlm_0624_173330_torch_infer_23b_text_gsm8k_weightmap_dpreduce_0624_precision_bs64.log
  - outputs/logs/vlm_0624_191357_torch_infer_23b_text_gsm8k_moerestore_0624_precision_bs64.log
  - outputs/logs/vlm_0624_203347_torch_infer_23b_text_gsm8k_skip_langreload_0624_precision_bs64.log
  - 3rdparty/xllm/xllm/backend/inferencer/torch/utils/weight_loader.py
  - 3rdparty/xllm/xllm/backend/distributed.py
  - 3rdparty/xllm/xllm/backend/inferencer/torch/layers/distributed_ops.py
  - 3rdparty/xllm/xllm/backend/inferencer/torch/torch_inferencer.py
  - 3rdparty/xllm/xllm/backend/inferencer/torch/layers/xpu/moe.py
  - maas_finetune/trainer/rl/rl_trainer.py
confidence: medium
status: active
open_questions:
  - "去掉 shared SeedVL 分支的 language model 二次 reload 后，纯文本 GSM8K 已连续 3 step 降至 logprobs_diff≈0.0097；这是灾难级 0.09/8.x 的阶段性修复证据，但仍未完成 70+ / 200 step 长线漂移验证。"
  - "后续仍需用 0608 健康口径长跑确认 torch infer BF16 NPU kernel_backend=xpu 下的 MoE layout / update_weights reload / DP-TP 通信一致性。"
  - "本文的 xpu 字样仅指 xLLM torch infer 的 NPU kernel backend 目录名，不代表 xpu_gpt/XG W8A8 rollout。"
contradictions: []
---

# Precision Alignment Report: VLM 23B torch infer BF16 纯文本 logprobs_diff 8.x 定界

## 1. 差异现象 (Deviation Symptoms)

- **现象**: 2026-06-24 合并 XG 23B 相关代码后，23B torch infer BF16 rollout 的 step1 `logprobs_diff` 从历史健康的 `~0.017` 爆到 `8.x`，同时 actor PPL 达到数万、reward 全 0。
- **数据类型 / 运行模式**: M8 23B，Actor/ref 为 BF16 训练，rollout 为 `XLLM_MODEL_CLS=torch` 的 torch infer BF16；8 节点 128 rank；`kernel_backend='xpu'` 表示 xLLM torch infer 的 NPU kernel backend，不是 `xpu_gpt` / XG W8A8 后端。
- **关键边界**: 纯文本 GSM8K 复现了同量级问题，因此本轮 `8.x` 爆炸主因不在 ViT 透传、图片 embedding 或 `/dev/shm/vit_cache`。

## 2. 指标对照

### 2.1 健康基线：0608 23B torch infer 8 节点

日志：

```text
outputs/logs/vlm_0608_162254_torch_infer_23b_full_8nodes_unified_vit_noshm_clean_retry_bs256_bs256.log
```

step1-step5 均稳定在健康量级：

| step | actor/ppl | rollout/ppl | logprobs_corr | logprobs_diff | reward.mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.25741 | 1.25488 | 0.995969 | 0.0175404 | 0.0463867 |
| 2 | 1.25093 | 1.24846 | 0.995937 | 0.0170525 | 0.0380859 |
| 3 | 1.25497 | 1.25325 | 0.996064 | 0.0172062 | 0.0493164 |
| 4 | 1.25605 | 1.25376 | 0.996003 | 0.0173718 | 0.0375977 |
| 5 | 1.25389 | 1.25141 | 0.995872 | 0.0173097 | 0.0449219 |

### 2.2 当前坏 run：VLM 与纯文本均炸到 8.x

VLM run：

```text
outputs/logs/vlm_0624_143127_torch_infer_23b_full_8nodes_fixsharedload_0624_precision_bs64.log
```

step1:

```text
actor/ppl=46178.8
rollout/ppl=11.6056
logprobs_corr=0.418161
logprobs_diff=8.3152
reward.mean=0
```

纯文本 GSM8K run：

```text
outputs/logs/vlm_0624_160814_torch_infer_23b_text_gsm8k_boundary_0624_precision_bs64.log
```

step1:

```text
prompt_length.mean=86.4688
actor/ppl=35370
rollout/ppl=11.5427
logprobs_corr=0.431408
logprobs_diff=8.03042
reward.mean=0
```

**结论**: 纯文本数据集没有图片输入，仍出现 `logprobs_diff=8.03042`，因此当前 8.x 不是 ViT cache / response-level `vit_embeds` 的主因。

### 2.3 阶段性回收补丁后的 run

日志：

```text
outputs/logs/vlm_0624_173330_torch_infer_23b_text_gsm8k_weightmap_dpreduce_0624_precision_bs64.log
```

step1:

```text
actor/ppl=1.20906
rollout/ppl=1.15262
logprobs_corr=0.822867
logprobs_diff=0.0973188
reward.mean=0.732422
```

**解释边界**:

- 这说明部分回收补丁确实击中了导致 `8.x` 的关键影响面：模型不再是灾难级坏分布，reward 也恢复非零。
- 但 `0.0973188` 仍远高于 0608 基线 `0.017`，只能称为阶段性恢复，不能宣告 torch infer 23B 精度修复完成。

### 2.4 0.09 残差修复：跳过 shared language 二次 reload

坏 run：

```text
outputs/logs/vlm_0624_191357_torch_infer_23b_text_gsm8k_moerestore_0624_precision_bs64.log
```

step1:

```text
logprobs_corr=0.820776
logprobs_diff=0.0985974
reward.mean=0.736328
```

该 run 的 update_weights 路径中，shared VLM 分支先加载 `SeedVLForConditionalGeneration loaded=1237`，随后又额外进入 `GPTSeedM8ForCausalLM loaded=905` 的 language model reload。

代码核查发现 `3rdparty/seed_inference_models/seed_inference_models/seed_vl.py::SeedVLForConditionalGeneration.load_weights()` 已通过 `_HF_PREFIX_MAP` 将 stripped `transformer.*` / `lm_head.*` key 映射到 `language_model.*`，即 shared wrapper load 已覆盖 language+vision。额外对 `self.model` 做第二次 language reload，反而会让已 post-process / restored 的 torch infer MoE/linear layout 再经历一次不等价加载。

修复口径：

```python
if vision_model is not None and getattr(self, "_vl_shared_language_model", False):
    self.load_model_weights(vision_model, state_dict=state_dict, kernel_backend=self.kernel_backend)
    return
```

验证 run：

```text
outputs/logs/vlm_0624_203347_torch_infer_23b_text_gsm8k_skip_langreload_0624_precision_bs64.log
```

runtime package 签名：

```text
_update_weights.enter state_dict=1756 vision_model=True shared_vl=True
_update_weights.before_load shared_vision_model
load_model_weights.after_load_actual model=SeedVLForConditionalGeneration loaded=1237
_update_weights.after_load shared_vision_model
```

该 run 没有旧的 `_update_weights.before_load shared_language_model`，也没有 `GPTSeedM8ForCausalLM loaded=905` 二次 reload。

step1-step3 指标：

| step | logprobs_corr | logprobs_diff | reward.mean |
|---:|---:|---:|---:|
| 1 | 0.996318 | 0.00976326 | 0.804688 |
| 2 | 0.996474 | 0.00969214 | 0.720703 |
| 3 | 0.996245 | 0.00986933 | 0.65625 |

step3 probe:

```text
overall_diff=0.00986933
corr=0.996245
valid_samples=512/512
valid_tokens=1097397
sample_mean_q50/q90/q95/q99=[0.008286,0.01312,0.01573,0.01873]
max_sample_mean=0.0209331
```

**阶段结论**:

- 23B torch infer BF16 纯文本 GSM8K 的 `0.09` 残余偏差已被该补丁打到 `0.0097~0.0099`，短测准入目标通过。
- 这也反向说明 0.09 的主因不是 ViT、图片 embedding、`/dev/shm/vit_cache`，而是 shared SeedVL update_weights reload 语义。
- 该结论仍不能替代 70+ / 200 step 长线漂移验证；必须继续用 0608 健康基线口径长跑确认不会后段漂移。

### 2.5 验证控制面补丁：禁用 checkpoint 必须覆盖 final save

同一验证 run 完成 step3 后失败在最终 checkpoint 保存：

```text
save checkpoint triggered, global_step=3, triggered_by_last=True
TypeError: Encoding objects of type OSError is unsupported
raylet: /data01/ray_tmp/session_* is over 99% full
```

根因是 `MAAS_DISABLE_CHECKPOINT=1` 虽然进入 runtime env，但 `RLTrainer._save_checkpoint()` 没有使用该全局开关，`trainer.save_every_n_steps=99999` 只能避开周期保存，避不开最后一步 `triggered_by_last=True`。

最小控制面修复：

```python
def _save_checkpoint(self, force: bool = False, sync: bool = False):
    if DISABLE_CHECKPOINT:
        logger.info(
            f"MAAS_DISABLE_CHECKPOINT=1, skip save checkpoint. global_step={self.global_step}, force={force}, sync={sync}"
        )
        return
```

该补丁只影响显式设置 `MAAS_DISABLE_CHECKPOINT=1` 的验证任务；默认训练保存语义不变。

## 3. 已实锤影响精度的代码面

高风险变更集中在 `3rdparty/xllm` 的提交：

```text
95a96ef259 fix: torch infer RL 权重重载与 SeedVL m8 适配
```

该提交触碰了 torch infer 核心路径，包括：

- `xllm/backend/inferencer/torch/utils/weight_loader.py`
- `xllm/backend/distributed.py`
- `xllm/backend/inferencer/torch/layers/distributed_ops.py`
- `xllm/backend/inferencer/torch/torch_inferencer.py`
- `xllm/backend/inferencer/torch/layers/xpu/moe.py`

### 3.1 普通权重 mapping 被 expert TP/EP 口径污染

问题写法把普通 mappings 也传入 `expert_tp_size/expert_tp_rank`：

```python
mappings = preprocess_mappings(
    mappings,
    hf_model_config,
    tp_size,
    tp_rank,
    expert_tp_size,
    expert_tp_rank,
)
```

阶段性回收写法：

```python
mappings = preprocess_mappings(mappings, hf_model_config, tp_size, tp_rank)
expert_mappings = preprocess_expert_mappings(
    expert_mappings,
    hf_model_config,
    expert_tp_size,
    expert_tp_rank,
    ep_size,
    ep_rank,
)
```

解释：普通 attention / embedding / non-expert 权重应按 TP 切片；只有 expert mappings 应按 expert TP/EP 口径处理。错误地让普通权重走 expert TP 口径，可能导致 128 rank 下非 expert 权重切片错位。

### 3.2 DP/EP all-reduce 被 TP 优先逻辑吞掉

问题写法近似为：

```python
if self.tp_size > 1:
    get_tensor_parallelism_msger().all_reduce(x.tensor)
```

阶段性回收写法：

```python
if x.tensor.numel() <= 0:
    return x
if self.group_type == CommGroupType.DP:
    get_data_parallelism_msger().all_reduce(x.tensor)
elif self.group_type == CommGroupType.EP:
    get_world_msger().all_reduce(x.tensor)
elif self.tp_size > 1:
    get_tensor_parallelism_msger().all_reduce(x.tensor)
```

解释：23B torch infer 8 节点口径是 `tp_size=8, dp_size=2`。DP/EP 通信不能因为 `tp_size>1` 就全部落到 TP group，否则 rollout 端 MoE/并行层的聚合语义会和 Actor/ref 不一致。

### 3.3 SeedVL shared language reload 路径不完整

阶段性修复在 shared VLM 模式下，除了加载 `vision_model` wrapper，还显式通过 `self.model` 的 language loader 处理 stripped RL sync keys：

```python
self.load_model_weights(
    vision_model,
    state_dict=state_dict,
    kernel_backend=self.kernel_backend,
)
self.load_model_weights(
    self.model,
    state_dict=state_dict,
    kernel_backend=self.kernel_backend,
)
```

解释：如果 wrapper mapping 只覆盖 vision 侧，而 RL sync state dict 里的语言权重是 `transformer.h.*`、`lm_head.*` 等 stripped key，则只加载 wrapper 可能让 language model 保留未初始化或旧 layout 状态。该问题能解释 actor PPL 数万与 reward 全 0 的灾难级表现。

## 4. 剩余高疑点

当前 `0.097` 仍明显异常，下一步不应再回到 ViT 或环境变量猜测，而应继续沿 torch infer BF16 NPU 路径定界。

### 4.1 `layers/xpu/moe.py` 的 MoE layout/reload 复杂化

注意：这里的 `xpu` 是 xLLM torch infer 的 NPU kernel backend 目录名，不是 `xpu_gpt`。

相对 0608 参考仓，当前 `xllm/backend/inferencer/torch/layers/xpu/moe.py` 新增了：

- `_gate_is_raw_layout`
- forward 期间 lazy `xprocess_weights_after_loading`
- `_new_param_preserving_attrs`
- `xrestore_weights_before_loading`
- 大量 memory/layout 日志

当前阶段性恢复 run 中仍可见大量：

```text
[MEM][xpu_moe] before gate transpose ...
[MEM][xpu_moe] after ffn1 transpose ...
patch_xllm_moe_model_weigth_loader patched=82
Cannot create tensor with interal format while allow_internel_format=False
```

这类信号说明 MoE layout 在 update/load/post-process/sleep 之间仍是最值得继续验证的区域。当前证据不足以把它定性为根因，但它符合“模型能跑、PPL 正常、logprobs_diff 仍明显偏大”的残余误差形态。

### 4.2 仍需排查的 torch infer BF16 代码面

- `torch_inferencer.py::sleep()` 中 `patch_xllm_moe_model_weigth_loader(self.model)` 与 update_weights reload 的交互。
- `post_process_model(..., loaded_full_param_names)` 对 MoE module 的 idempotent 跳过/强制处理逻辑。
- `xrestore_weights_before_loading()` 是否在所有 reload 前都能恢复 loader 期望的 raw layout。
- `attention.py`、`torch_async_inferencer.py`、`meta.py`、`output_processor.py` 中 0624 工作树未完全验证的改动是否影响纯文本 logprob 口径。

## 5. 0.097 阶段结论与修复目标

历史 `logprobs_diff=0.0973188` 不能定性为修复完成；它曾说明 `8.x` 灾难级错位被拉回后，torch infer BF16 language path 仍存在可观残余偏差。2026-06-24 后续补丁去掉 shared language 二次 reload 后，纯文本 GSM8K step1-step3 已恢复到 `0.0097~0.0099`，说明这条 0.09 残差主因已被阶段性击中。

- **已降低优先级**: ViT、图片 embedding、`/dev/shm/vit_cache`。纯文本 GSM8K 没有图片输入，仍从 `8.03042` 降到 `0.0973188`，说明剩余问题应继续在语言模型 reload/layout/通信路径内定界。
- **已验证主因**: SeedVL shared-language reload 与 `layers/xpu/moe.py` 的 raw/kernel layout restore/post-process 交互不等价于 0608 健康路径；`shared_vision_model` load 后再 `shared_language_model` load 会引入二次 reload/layout 处理，删除后短测恢复。
- **次级假说**: DP/EP/TP communication group 仍可能存在语义偏差，尤其是 23B `tp_size=8, dp_size=2, XLLM_MODEL_USE_EP=0` 口径下的 `AllReduce(group_type=DP/EP/TP)` 选择。通信假说应在 MoE layout lossless/idempotent 被验证后再做 A/B，避免同时移动多个变量。

修复目标分三层：

1. **短测准入目标**: 继续使用 23B torch infer BF16 纯文本 GSM8K bsz64/batch512 边界测试，step1 `train/rollout/logprobs_diff <= 0.03`，`logprobs_corr >= 0.98`，actor/rollout PPL 同量级且 reward 非全零；日志中无 `Traceback`、`RuntimeError`、`ActorDiedError`、`RayActorError`、Gloo/RPC fatal。该目标已由 `vlm_0624_203347...` step1-step3 达成。
2. **短稳态目标**: 同口径至少连续 3 个 global_step 保持 `logprobs_diff` 在 `0.017~0.03` 区间，不出现 forward 期 `raw-layout MoE weights detected` 懒处理告警；若仍有 `Cannot create tensor with interal format...`，需证明它与 logprob 偏差无关或消除该告警。该目标已由 `vlm_0624_203347...` 阶段性达成，但需转入更长 horizon。
3. **VLM 回归目标**: 回到 23B VLM 8 节点口径后，step1-step5 应恢复到 0608 健康基线量级，即 `logprobs_diff≈0.017~0.03`、reward 非全零、无 `missing embedding` / `GLOBAL MISS` 主链路异常。最终仍必须按 70+ / 200 step 长线漂移标准验证，不能用 early step 直接宣告精度完成。

下一步建议的证伪顺序：

1. 优先按 0608 健康基线配置长跑，验证删除 shared language 二次 reload 后是否能跨过 70+ step 且最终 200 step 不漂移。
2. 若长线再次漂移，再做 MoE layout 的 lossless/idempotent 检查：验证 `xprocess_weights_after_loading()` 与 `xrestore_weights_before_loading()` 对 gate/ffn0/ffn1 的往返是否逐 tensor保持等价，并检查 run 中是否存在 forward 期 lazy xprocess。
3. 最后做通信 A/B：固定 MoE/reload 后，再比较 DP group all-reduce、EP/world group 语义与 0608 健康路径的差异。

## 6. 下一会话建议

1. **继续用纯文本 GSM8K 作为边界测试**：它排除了 ViT，首个 step 比 VLM 更快，适合快速判断 language/rollout 对齐是否恢复。
2. **不要新增 torch infer W8A8/no_quant 分支**：torch infer rollout 在当前仓系中只按 BF16 处理；W8A8 属于 XG/xpu_gpt rollout。
3. **优先做最小差异回收**：以 0608 健康参考仓为对照，逐个缩小 `weight_loader`、`distributed_ops`、`torch_inferencer`、`layers/xpu/moe.py` 差异，不要整体回滚 XG 23B 适配。
4. **验证标准**：短期目标不是“跑起来”，而是纯文本 step1 `logprobs_diff` 回到 `~0.017-0.03` 量级；`0.097` 只能说明灾难级问题缓解，不能算修复。
5. **长线标准**：即使 early step 回到健康量级，也必须继续 70+ / 200 step 观察，避免把早期低 diff 误判为长跑精度稳定。

## 7. 关联知识

- [[wiki/02_precision/vlm_torch_infer_response_vit_embeds_unified_cache|torch infer VLM rollout response 直传 vit_embeds 统一 ViT cache 方案]]
- [[wiki/02_precision/vlm_torch_infer_frozen_vlm_weight_sync_reward|torch infer frozen VLM 权重未同步导致 reward 低于合并前]]
- [[wiki/05_fact/torch_infer_merge_debug_checklist|xpu_gpt + torch infer 合并验证排查清单]]
- [[wiki/02_precision/vlm_23b_xg_logpdiff_spike_replay_handoff|VLM 23B XG logprobs_diff spike replay 定位现场]]
