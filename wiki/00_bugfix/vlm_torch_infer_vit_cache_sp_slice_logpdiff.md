---
title: "Bugfix: torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸"
tags: [status/resolved, priority/high, category/cache, category/precision, vlm, torch_infer]
last_updated: 2026-05-30
source:
  - outputs/logs/vlm_0530_190651_24189_torchinfer_bs128.log
  - outputs/logs/vlm_0530_193421_24189_torchinfer_bs128.log
  - outputs/logs/vlm_0530_195746_24189_torchinfer_bs128.log
  - outputs/logs/vlm_0529_230400_2450_bs128_torch_infer_2b5_bs128.log
confidence: high
status: active
open_questions:
  - "该修复已验证前 2 个 global_step 恢复到健康量级；70+ / 200 step 长线漂移仍需按长跑标准单独验证。"
contradictions: []
---

# Bugfix Report: torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸

## 1. 现象描述 (Symptoms)

- **环境**: VLM 2B5 RL，torch inference 后端，24189，`BATCH_SIZE=128`，模型路径 `/data01/model_weights/2b5_saved_model`。
- **现象**: merge 后 torch infer 在 24189 上重新验证时，`train/rollout/logprobs_diff` 从健康基线约 `0.018` 飙升到 `0.06~0.10`。
- **坏日志样例**:
  ```text
  outputs/logs/vlm_0530_190651_24189_torchinfer_bs128.log
  global_step=1, train/rollout/logprobs_corr:0.715809,
  train/rollout/logprobs_diff:0.0790442

  outputs/logs/vlm_0530_193421_24189_torchinfer_bs128.log
  global_step=1, train/rollout/logprobs_corr:0.773182,
  train/rollout/logprobs_diff:0.0618672
  ```
- **健康对照**:
  ```text
  outputs/logs/vlm_0529_230400_2450_bs128_torch_infer_2b5_bs128.log
  global_step=1, train/rollout/logprobs_corr:0.983419,
  train/rollout/logprobs_diff:0.0179593
  ```

## 2. 根本原因分析 (Root Cause)

这次不是模型权重问题，也不是 `max_position_embeddings` 主因。根因是两个问题叠加：

1. **`/dev/shm/vit_cache` 旧 cache 污染新任务**  
   `VitEmbedCache.put()` 对已存在文件只做 `utime()`，不会覆盖旧内容。如果新任务启动时 `/dev/shm/vit_cache` 里残留旧 embedding，Actor 会命中旧特征。

   证据：坏日志中 Actor BYPASS 的 ViT embedding 均值长期在 `-68~-70`，健康日志在 `-35~-40`。清空 cache 后，均值回到健康区间。

2. **Actor `compute_logprobs` 错误地对 cached `vit_embeds` 做 sequence-dim SP 切片**  
   cached `vit_embeds` 是 rollout 侧已经算好的完整图像 embedding，传给模型时应作为 `image_embeds` 走 BYPASS。模型内部 `get_sp_input_embeds()` 已经会对 cached ViT embedding 做 hidden-dim SP 切分：

   ```python
   image_embeds = _slice_cached_vit_emb_for_sp(image_embeds)
   ```

   如果外层再调用 `slice_and_pad_for_sequence_parallel(..., dim=0)`，就会把图像 token 序列维切成约 `1/4`，导致 Actor 用残缺的图像 embedding 对齐完整的 image token mask，直接放大 `logprobs_diff`。

   证据对比：

   | 日志 | step1 diff | BYPASS shape 总 token 数 | ViT mean |
   |---|---:|---:|---|
   | 健康基线 `0529_230400` | `0.0179593` | `261560` | `-35~-40` |
   | 坏日志 `0530_190651` | `0.0790442` | `61772` | `-68~-70` |
   | 只清 cache 后 `0530_193421` | `0.0618672` | `69520` | `-35~-40` |
   | 修复 SP 切片后 `0530_195746` | `0.0178897` | `300443` | `-35~-40` |

其中 `0530_193421` 说明 cache 污染被修正后仍然不够，因为 `vit_embeds` 依旧被 sequence-dim 切短；`0530_195746` 说明移除该切片后恢复到健康量级。

## 3. 解决方案 (Solution)

### Fix

1. 在 RLTrainer setup 入口清空全局 ViT cache，避免跨任务复用旧 embedding：

   ```python
   def _setup(self):
       get_global_vit_cache().clear()
       self._init_rollout().wait()
   ```

2. 在 `compute_logprobs` 中，`vit_embeds` 存在时只做 BYPASS 参数整理，不再做 sequence-dim SP 切片：

   ```python
   if "vit_embeds" in kwargs:
       kwargs.pop("pixel_values", None)
       kwargs.pop("image_grid_hw", None)
       kwargs["image_embeds"] = kwargs.pop("vit_embeds")
   ```

### 禁忌

- 不要把 cached `vit_embeds` 当作 `pixel_values` 一样做 `slice_and_pad_for_sequence_parallel(dim=0)`。
- 不要用修改共享模型目录、覆盖权重或强行改 `max_position_embeddings` 来掩盖这个问题。
- `/share_data/...` 是共享目录时，不要把它当作单机本地目录跨节点拷贝覆盖。

## 4. 验证结果 (Verification)

24189 上重新验证 torch infer bs128：

```text
outputs/logs/vlm_0530_195746_24189_torchinfer_bs128.log

global_step=1:
train/rollout/logprobs_corr:0.984695
train/rollout/logprobs_diff:0.0178897

global_step=2:
train/rollout/logprobs_corr:0.981221
train/rollout/logprobs_diff:0.0167843
```

该结果与健康基线 `0.0179593 / 0.983419` 对齐。日志未发现 `Traceback`、`RuntimeError`、`ValueError`、`Raylet is terminated`。

注意：这是短程精度恢复证据，不能替代 70+ / 200 step 长跑。长线 `logprobs_diff` 漂移仍需按 `VLM 2.5B RL logprobs_diff 长线漂移精度对齐` 的标准单独验证。

## 5. 沉淀与关联

- 快速诊断方式：
  - 对比健康/坏日志中 `[ViT-Bypass] Using pre-computed image_embeds shape=torch.Size([... , 2048])` 的总 token 数。
  - 对比 `[ViT-Probe] path=BYPASS` 的 `mean`，健康区间约 `-35~-40`，明显偏到 `-68~-70` 时优先怀疑 cache 污染。
  - 若 mean 健康但 shape 总量只有健康基线约 `1/4`，优先怀疑 cached `vit_embeds` 被 sequence parallel 错切。
- 关联知识点:
  - [[wiki/00_bugfix/vit_cache_miss_cityhash|Bugfix: ViT Cache 100% Miss & Race Condition Mismatch]]
  - [[wiki/00_bugfix/vlm_cache_hash_mismatch|Bugfix: VLM Cache Miss Due to Cross-Boundary Hash Mismatch]]
  - [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
