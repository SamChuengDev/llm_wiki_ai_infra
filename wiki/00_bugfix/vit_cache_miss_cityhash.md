---
title: "Bugfix: ViT Cache 100% Miss & Race Condition Mismatch"
tags: [status/resolved, priority/critical, category/caching]
last_updated: 2026-05-19
source: [outputs/logs/vlm_0518_223521_2450_verify_cache_miss_magnitude_bs2.log, outputs/logs/vlm_0519_092103_2450_verify_cache_miss_magnitude_bs128.log]
confidence: high
status: resolved
open_questions:
  - "是否还需要保留基于 MD5 的 soft-hash fallback 作为兜底？当前直接采用 VlmDialogProcessor 的严格一致性 Hash 理论上已无需 soft-hash。"
contradictions: []
---

# Bugfix Report: ViT Cache 100% Miss & Race Condition Mismatch

## 1. 现象描述 (Symptoms)
- **环境**: xLLM Rollout 引擎 (`VlmDialogProcessor`) + PPO Actor (`template.py`) 双端训练环境，启用 `ENABLE_VIT_CACHE=1`。
- **报错信息**:
  在日志中出现大面积甚至 100% ViT Cache Miss：
  ```text
  [ViT-Cache-ACTOR-MISS] rank=4 img_idx=5/29 hash=-1137098652142668159 file_exists=False
  ```
  尽管 Rollout 端已经处理相同图像并写入缓存，Actor 仍无法命中。早期日志还显示 Rollout 侧 hash 形如 19 位 `CityHash64`，Actor 侧 hash 形如 39 位 `CityHash128`，存在明显位数与格式差异。

## 2. 根本原因分析 (Root Cause)
引发该问题的根本原因是双端 Hash 算法及其依赖的动态参数不一致，具体分为两个递进阶段：

1. **Hash 算法基础不一致**：
   Actor 的 `seed_vl_prompt` 默认使用 `CityHash128`，而 Rollout 端的 `vlm_dialog_processor` 使用 `CityHash64`，并附带特定 `resize_ratio` suffix。因此同一张图像在两端生成不同 key，导致 `/dev/shm/vit_cache` 无法命中。

2. **动态参数导致手工仿真失效**：
   后续尝试在 Actor 中手工复刻 `CityHash64` 时，曾硬编码 `use_anyres=False` 与 `resize_ratio=1.0`。但 `vlm_dialog_processor` 的 `_calculate_image_hash` / `_get_images_from_raw_images` 表明：
   - `use_anyres` 取决于数据集传入的 `detail` 字段以及 vit-encoder meta 配置。
   - `resize_ratio` 是动态计算值，例如 `resize_ratio = math.sqrt(image.max_pixels / total_pixels)`。
   浮点 `resize_ratio` 会通过 `struct.pack("d", resize_ratio)` 进入 hash 输入；硬编码 `1.0` 会与真实 Rollout hash 不一致，造成部分或全部 GLOBAL MISS。

## 3. 解决方案 (Solution)
### 终极方案 (Fix)
放弃在 Actor 端人工仿造 Hash。改为在 `template.py` 中直接实例化与 Rollout 端完全相同的 `VlmDialogProcessor`，并仅用它计算和提取 100% 一致的 `CityHash64`：

- [x] 在 `template.py` 的 `_encode_image` 初始化块中，使用当前环境的 `tokenizer` 和 `model_config` 实例化 `VlmDialogProcessor` 为 `self.xllm_processor_hash`。
- [x] 保留原有 `seed_vl_prompt` 生成 `pixel_values` 的逻辑，确保前向传播输入不变。
- [x] 调用 `self.xllm_processor_hash.encode([msg])`，直接透传真实框架解析后的 `hash_img_data.image_hashes[0]` 给 Actor 消费。
- [x] 在 `actor.py` 补充适配逻辑，确保 `image_raw_hashes[i]` 被正确读取并与图片顺序对齐。

## 4. 验证结果 (Verification)
移除人工 hardcode 后，所有 `resize_ratio` 和 `use_anyres` 的不确定性都被 `VlmDialogProcessor` 内聚计算，双端 hash 一致性得到数学层面的保证。

当前日志 `outputs/logs/vlm_0519_092103_2450_verify_cache_miss_magnitude_bs128.log` 已在 `bs=128` 稳态长跑，确认消除 hash 不一致造成的伪 miss。

## 5. 沉淀与关联
- 关联知识点: [[wiki/02_precision/rl_off_policy_inconsistency|训推架构 Off-Policy 异构发散诱因与对策]]
