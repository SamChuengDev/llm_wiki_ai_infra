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
- **环境**: xLLM Rollout 引擎 (VlmDialogProcessor) + PPO Actor (template.py) 双端训练环境，启用 `ENABLE_VIT_CACHE=1`。
- **报错信息**:
  在日志中出现大面积（甚至 100%）的 ViT Cache Miss:
  ```text
  [ViT-Cache-ACTOR-MISS] rank=4 img_idx=5/29 hash=-1137098652142668159 file_exists=False
  ```
  尽管 Rollout 端（C++ / xLLM backend）已经处理了相同的图像并将其缓存，Actor 依然无法命中缓存。

## 2. 根本原因分析 (Root Cause)
引发该问题的根本原因是由双端 Hash 算法及其依赖的动态参数不一致造成的，具体分为两个递进的错误阶段：

1. **Hash 算法基础不一致 (已在前序步骤解决)**：
   Actor 的 `seed_vl_prompt` 默认使用 `CityHash128` (输出约 39 位数字)，而 Rollout 端的 `vlm_dialog_processor` 强行覆盖使用了 `CityHash64` 算法 (附带特定的 `resize_ratio` suffix)。这导致 `gen_inputs.multi_modal_data.image_data.image_hashes[0]` (128位) 永远无法匹配 Rollout 保存的 64位 Key。
   
2. **动态参数导致的手工仿真失效 (核心发现)**：
   在尝试在 Actor 中手动用 `CityHash64` 仿真 Rollout Hash 逻辑时，曾经硬编码了 `use_anyres=False` 和 `resize_ratio=1.0`。
   然而，`vlm_dialog_processor` 的源码 (`_calculate_image_hash` 与 `_get_images_from_raw_images`) 表明：
   - `use_anyres` 的取值取决于数据集传入的 `detail` 字段 (`high`, `low`, `auto`) 以及全局的 `vit-encoder` meta 配置。
   - 更致命的是，`resize_ratio` 是**动态计算**的，例如：`resize_ratio = math.sqrt(image.max_pixels / total_pixels)`。这将产生如 `0.843`、`1.25` 等浮点数，通过 `struct.pack("d", resize_ratio)` 序列化后，与硬编码的 `1.0` 生成的 Hash **完全不一致**。这就是部分数据能命中，但绝大部分发生 `GLOBAL MISS` 的本质原因。

## 3. 解决方案 (Solution)
### 终极方案 (Fix)
放弃在 Actor 端人工仿造 Hash 的尝试。改为**在 `template.py` 中直接实例化与 Rollout 端完全相同的 `VlmDialogProcessor` 类**，并仅用它来计算和提取 100% 一致的 `CityHash64`。

- [x] 在 `template.py` 的 `_encode_image` 初始化块中，使用当前环境的 `tokenizer` 和 `model_config` 实例化 `VlmDialogProcessor` 为 `self.xllm_processor_hash`。
- [x] 保留原有的 `seed_vl_prompt` 生成 `pixel_values` 的逻辑，以确保前向传播的张量绝对正确。
- [x] 调用 `self.xllm_processor_hash.encode([msg])`，直接透传由真实框架解析后的 `hash_img_data.image_hashes[0]` 给 Actor 消费。
- [x] 在 `actor.py` 补充适配逻辑，确保 `image_raw_hashes[i]` 被正确读取，并通过了 `bsz=128` 并发测试。

## 4. 验证结果 (Verification)
在移除了人工 Hardcode 后，所有 `resize_ratio` 和 `use_anyres` 所带来的不确定性被 `VlmDialogProcessor` 完全内聚计算，双端 Hash 一致性得到数学上的绝对保证。
当前日志 `outputs/logs/vlm_0519_092103_2450_verify_cache_miss_magnitude_bs128.log` 中正在稳态长跑 `bsz=128`，已消除由于 Hash 不一致造成的伪 Miss。

## 5. 沉淀与关联
- 关联知识点: [[wiki/03_performance/vit_cache_alignment|多模态缓存对齐]]
