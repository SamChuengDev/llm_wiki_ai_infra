---
title: "Bugfix: VLM RL Template 未复用 rollout vit_embeds 导致 Actor 侧重复图片编码"
tags: [status/resolved, priority/high, category/vlm, category/cache, category/precision, xpu_gpt, torch_infer, rl]
last_updated: 2026-06-02
source:
  - outputs/logs/vlm_0601_165516_2450_after_merge_bs128_filter.log
  - outputs/logs/vlm_0601_213829_24189_torchinfer_align_ypq_bs128_filter.log
  - maas_finetune/trainer/rl/template.py
  - commit:58187c25f99f83db9f212e79538ee928c7a47c86
confidence: high
status: active
open_questions:
  - "该修复已覆盖 xg 120 step 与 torch inference 62 step 短中程验证；仍不替代 200 step 长线精度漂移结论。"
contradictions: []
---

# Bugfix Report: VLM RL Template 未复用 rollout vit_embeds 导致 Actor 侧重复图片编码

## 1. 现象描述 (Symptoms)

- **环境**: VLM 2B5 RL，xpu_gpt / xg 与 torch inference 双后端合并后验证，`BATCH_SIZE=128`，rollout 侧可向训练样本透传 `vit_embeds`。
- **现象**: 数据样本已经包含 rollout 侧预计算的 `vit_embeds` 时，`RLTemplate.encode()` 仍然逐条扫描 `messages` 并调用 `_encode_image()` 重新编码图片，再在输出末尾把 `vit_embeds` 原样塞回 `out`。
- **风险表现**:
  - Actor/ref 侧重复执行图片读取、hash、预处理与 ViT 相关准备，浪费 CPU/IO 与 DataProcessor 时间。
  - `pixel_values/image_grid_hw` 与 `vit_embeds` 同时存在时，后续路径更容易混淆“原始图片像素”和“已预计算 embedding”两种语义。
  - 本地重新编码得到的 `total_image_tokens` 可能与 rollout 侧 embedding token 数不一致；旧逻辑只校验本地 `_encode_image()` 的 token 数，未直接校验 `vit_embeds.shape[0]`。

该问题不一定表现为单一 Traceback，更容易表现为 merge 后双后端 VLM 训练路径中的重复编码、token 对齐风险或精度/性能不稳定。

## 2. 根本原因分析 (Root Cause)

旧 `RLTemplate.encode()` 的控制流没有把 `vit_embeds` 当成可直接复用的 BYPASS 数据源：

```python
for msg in data["messages"]:
    ...
    pixel_values, image_grid_hw, num_image_tokens, raw_image_hash = self._encode_image(image_url_data)
    pixel_values_list.append(pixel_values)
    image_grid_hw_list.append(image_grid_hw)
    total_image_tokens += num_image_tokens

...

if "vit_embeds" in data and data["vit_embeds"] is not None:
    out["vit_embeds"] = data["vit_embeds"]
```

也就是说，即使 rollout 已经提供 `vit_embeds`，Actor/ref 模板侧仍会重新编码图片，并以本地图片编码结果参与 token 数检查。对于 cached ViT embedding 路径，真正应校验的是：

```text
vit_embeds.shape[0] == input_ids 中 image_token_id 的数量
```

而不是再次依赖 Actor 侧 `_encode_image()` 的 `num_image_tokens`。这与已沉淀的经验一致：cached `vit_embeds` 与 `pixel_values` 是不同语义，前者应作为完整图像 embedding 传给模型 BYPASS，不应再当作像素输入重新处理。

## 3. 解决方案 (Solution)

### Fix

提交 `58187c25f99f83db9f212e79538ee928c7a47c86` 修改 `maas_finetune/trainer/rl/template.py`：

1. 先读取 rollout 侧预计算 embedding：

   ```python
   rollout_vit_embeds = data.get("vit_embeds")
   use_rollout_vit_embeds = rollout_vit_embeds is not None
   total_image_token_ids = (input_ids == self.image_token_id).sum().item()
   ```

2. 若存在 `vit_embeds`，跳过 Actor 侧 `_encode_image()`，并把 `total_image_tokens` 直接对齐到 `input_ids` 中的 image token 数：

   ```python
   if use_rollout_vit_embeds:
       total_image_tokens = total_image_token_ids
   else:
       # only then scan messages and _encode_image(...)
   ```

3. 增加 `vit_embeds` token 数校验：

   ```python
   if use_rollout_vit_embeds and getattr(rollout_vit_embeds, "shape", None) is not None:
       vit_embed_tokens = rollout_vit_embeds.shape[0]
       if vit_embed_tokens != total_image_token_ids:
           logging.getLogger(__name__).warning(
               f"Discarding sample: vit_embeds tokens ({vit_embed_tokens}) "
               f"does not match replaced token ids ({total_image_token_ids})"
           )
           return None
   ```

4. 只有真的生成了 `pixel_values` 时才校验 `pixel_size == hw_size`；`vit_embeds` BYPASS 路径不再要求像素输入存在。

### 禁忌

- 不要在已有 `vit_embeds` 的样本上再强制走 `_encode_image()`，这会把 cached embedding 路径重新拖回图片预处理路径。
- 不要把 cached `vit_embeds` 当作 `pixel_values`，也不要在模板层引入 sequence-dim 切分。
- 不要把该修复解读为 200 step 长线精度漂移已经完成验证；它只修复模板层 BYPASS 数据复用与 token 校验。

## 4. 验证结果 (Verification)

### 静态检查

```text
python3 -m py_compile maas_finetune/trainer/rl/template.py
git diff --check -- maas_finetune/trainer/rl/template.py
```

均通过。

### xpu_gpt / xg 后端

验证日志：

```text
outputs/logs/vlm_0601_165516_2450_after_merge_bs128_filter.log
```

该日志包含 120 个 `global_step`，未发现 `Traceback`、`InvalidImageURL`、`RuntimeError`、`ValueError`、`AttributeError`、`TypeError`、`Raylet is terminated`，也未发现新增的 `Discarding sample: vit_embeds tokens` 或 `Discarding sample: total_image_tokens` 警告。

关键尾部指标：

```text
step=116 reward_mean=0.651367 logprobs_diff=0.0255698
step=117 reward_mean=0.620117 logprobs_diff=0.0258919
step=118 reward_mean=0.617188 logprobs_diff=0.0236118
step=119 reward_mean=0.574219 logprobs_diff=0.0240159
step=120 reward_mean=0.59375  logprobs_diff=0.0258458
```

### torch inference 后端

验证日志：

```text
outputs/logs/vlm_0601_213829_24189_torchinfer_align_ypq_bs128_filter.log
```

该日志包含 62 个 `global_step`，同样未发现上述错误或新增 discard 警告。

关键指标：

```text
step=1  reward_mean=0.0253906 logprobs_diff=0.0180918
step=2  reward_mean=0.0205078 logprobs_diff=0.0169911
step=60 reward_mean=0.262695  logprobs_diff=0.0170843
step=61 reward_mean=0.303711  logprobs_diff=0.0175972
step=62 reward_mean=0.331055  logprobs_diff=0.0170734
```

结论：该修复已在双后端验证中保持无 Traceback、无新增 token mismatch discard，并维持健康的短中程 `logprobs_diff` 量级。最终长线漂移仍需按 200 step 标准单独判断。

## 5. 沉淀与关联

- 诊断时先判断样本是否已有 `vit_embeds`：
  - 有 `vit_embeds`：模板层应跳过 `_encode_image()`，直接传递 `out["vit_embeds"]`，并校验 `vit_embeds.shape[0]` 与 image token 数一致。
  - 无 `vit_embeds`：才走原图片 URL 的 `_encode_image()` 路径，并产出 `pixel_values/image_grid_hw/image_raw_hashes`。
- 关联知识点:
  - [[wiki/00_bugfix/vlm_torch_infer_vit_cache_sp_slice_logpdiff|Bugfix: torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸]]
  - [[wiki/00_bugfix/vlm_rl_xg_hash_image_size_check|Bugfix: VLM 2B5 xg RL hash 路径未关闭 xLLM 图片尺寸检查导致 InvalidImageURL]]
  - [[wiki/00_bugfix/vlm_cache_hash_mismatch|Bugfix: VLM Cache Miss Due to Cross-Boundary Hash Mismatch]]
  - [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
