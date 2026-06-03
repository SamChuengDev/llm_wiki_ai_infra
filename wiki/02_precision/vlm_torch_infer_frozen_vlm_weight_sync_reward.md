---
title: "Precision: torch infer frozen VLM 权重未同步导致 reward 低于合并前"
tags: [status/active, category/diff, category/vlm, category/torch-infer]
last_updated: 2026-06-02
source:
  - outputs/logs/vlm_0601_213829_24189_torchinfer_align_ypq_bs128_filter.log
  - outputs/logs/vlm_0602_113333_24189_torchinfer_align_ypq_bs128.log
  - /Users/bytedance/posttrain/codes/maas_engine_torch_infer/outputs/logs/vlm_0528_002344.log
confidence: high
status: active
open_questions:
  - "当前修复已恢复 early-step VLM embedding 数值与 reward/logprobs 量级，但仍需 70+ step / 200 step 长线验证。"
contradictions: []
---

# Precision Alignment Report: torch infer frozen VLM 权重未同步导致 reward 低于合并前

## 1. 差异现象 (Deviation Symptoms)
- **现象**: xpu_gpt 与 torch inference 分支合并后，torch infer 2B5 VLM RL 在关键配置对齐的情况下 reward mean 曲线显著低于合并前参考分支；但 `train/rollout/logprobs_diff` 仍在 `~0.016-0.018` 健康量级。
- **数据类型 / 运行模式**: BF16 Actor/ref 训练，torch infer DirectEngine rollout，M8 2B5 VLM，`BATCH_SIZE=128`，`rollout.engine_type=xllm_direct`，`XLLM_MODEL_CLS=torch`。
- **关键症状**:
  - 合并前 `/Users/bytedance/posttrain/codes/maas_engine_torch_infer/outputs/logs/vlm_0528_002344.log` 前 40 step reward mean 平均约 `0.33667`。
  - 合并后 `outputs/logs/vlm_0601_213829_24189_torchinfer_align_ypq_bs128_filter.log` 前 40 step reward mean 平均约 `0.120459`。
  - 同期 `logprobs_diff` 仅从合并前平均约 `0.016217` 到合并后约 `0.017737`，不足以解释 reward 大幅下降。

## 2. 定位过程 (Debugging Process)
- **配置核对**: 合并前后关键训练口径一致：`actor.ep_size=2`、`actor.sp_size=4`、`rollout.tp_size=4`、`data.batch_size=128`、`rollout.generation.n=8`、`rollout.engine_type=xllm_direct`、模型路径均指向 `/data01/sft_output` 口径。
- **数据路径排除**: 对比 step1-step5 的 prompt/answer unique 集合，交集为 100%，基本排除数据 shuffle 或样本口径导致的 reward 差异。
- **核心张量证据**: 同一 image hash 的 rollout VLM embedding 在合并前后差异巨大：
  - 合并前 hash `201384752145462263951576903013308326893`: shape `[255, 2048]`，sum `3592.565`，element mean `0.006879`。
  - 合并后坏 run: shape `[255, 2048]`，sum `-21035524.000`，element mean `-40.2794`。
- **分阶段 probe**: 临时 probe 显示 bad run 的 `pixel_values` 正常，但 vision/projector 输出异常：
  - `input.pixel_values`: mean `1.6493`，min/max 正常。
  - `vision_encoder.output`: mean `-34.4939`，数值范围异常。
  - `ln_vision.output`: mean `0.081421`，被 LayerNorm 拉回。
  - `multi_modal_projector.output`: mean `-40.2794`，再次异常放大。
- **权重同步证据**:
  - 合并前参考日志可见 `_save_model_to_shm` 同步 `vision_encoder.*`、`ln_vision.*`、`multi_modal_projector.*`。
  - 合并后坏日志中 `multi_modal_projector` 只出现在 post_process/probe 相关日志，缺少 `add shm tensor sync_k='multi_modal_projector.*'`。
  - torch infer DirectEngine worker 初始化路径存在 `mock_weights=True` 语义；若后续 update_weights 不传 frozen VLM 权重，rollout 侧 VIT/projector 会保留 mock/random 权重。

## 3. 修复方案 (Alignment Fix)
- **根因**: 合并时在 `ActorWorker._save_model_to_shm()` 中加入了 frozen VLM 权重过滤：
  ```python
  VLM_FROZEN_PREFIXES = ("vision_encoder.", "multi_modal_projector.", "ln_vision.")
  for idx, (name, tensor) in enumerate(state_dict.items()):
      if any(name.startswith(p) for p in VLM_FROZEN_PREFIXES):
          continue
  ```
  这对 xpu_gpt 后端是合理的，因为 xpu_gpt/Companion 会从自身模型路径加载冻结 VLM 模块，LLM xworker 的 update_weights 不需要 VLM 权重；但对 torch infer 后端是精度 bug，因为 torch infer rollout 初始化使用 mock weights，必须通过 SHM/update_weights 至少同步一次真实 `vision_encoder`、`ln_vision`、`multi_modal_projector`。
- **修复**: 仅在非 torch infer 后端跳过 frozen VLM 权重；torch infer 保留这些权重进入正常同步路径。
  ```python
  VLM_FROZEN_PREFIXES = ("vision_encoder.", "multi_modal_projector.", "ln_vision.")
  skip_frozen_vlm_weights = os.getenv("XLLM_MODEL_CLS", "").lower() != "torch"
  for idx, (name, tensor) in enumerate(state_dict.items()):
      if skip_frozen_vlm_weights and any(name.startswith(p) for p in VLM_FROZEN_PREFIXES):
          continue
  ```
- **兼容边界**:
  - `XLLM_MODEL_CLS=torch`: 不跳过 frozen VLM 权重，修复 torch infer rollout mock VLM 权重问题。
  - `XLLM_MODEL_CLS=xpu` 或未设置: 继续跳过 frozen VLM 权重，保持 xpu_gpt/Companion 路径原行为，避免重新引入 xpu_gpt NPU0 额外显存压力。

## 4. 对齐验证 (Validation)
- **权重同步恢复**: 修复验证日志 `outputs/logs/vlm_0602_113333_24189_torchinfer_align_ypq_bs128.log` 重新出现：
  - `add shm tensor sync_k='vision_encoder.patch_embed.proj.weight'`
  - `add shm tensor sync_k='ln_vision.weight'`
  - `add shm tensor sync_k='multi_modal_projector.0.weight'`
  - `add shm tensor sync_k='multi_modal_projector.2.weight'`
- **VLM embedding 数值恢复**: Actor BYPASS `ViT-Probe` 中 image embedding mean 从坏 run 的 `-40~-70` 恢复到 `~0.005-0.008`。
- **early-step 指标恢复**:
  - step1: `reward.mean=0.0791016`，`logprobs_diff=0.0167315`
  - step2: `reward.mean=0.104492`，`logprobs_diff=0.015885`
  - step13: `reward.mean=0.178711`，`logprobs_diff=0.0163209`
  - step14: `reward.mean=0.171875`，`logprobs_diff=0.0158979`
- **验证边界**: early-step `logprobs_diff` 与 reward 回升只能证明本次权重同步 bug 已被修复到早期健康量级；不能替代 70+ step 漂移观察，也不能替代 200 step 终局精度结论。

## 5. 关联知识
- [[wiki/02_precision/rl_precision_metrics_guide|RL 核心监控指标解析]]
- [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
- [[wiki/00_bugfix/vlm_torch_infer_vit_cache_sp_slice_logpdiff|torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸]]
- [[wiki/05_fact/torch_infer_merge_debug_checklist|xpu_gpt + torch infer 合并验证排查清单]]
