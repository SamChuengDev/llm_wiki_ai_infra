---
title: "Precision: torch infer VLM rollout response 直传 vit_embeds 统一 ViT cache 方案"
tags: [status/active, category/diff, category/cache, vlm, torch_infer, vit_cache]
last_updated: 2026-06-06
source:
  - outputs/logs/vlm_0606_070312_torch_infer_23b_full_8nodes_vit_responsefallback_bs256_long_bs256.log
  - outputs/logs/vlm_0602_113333_24189_torchinfer_align_ypq_bs128.log
  - maas_finetune/trainer/rl/rollout/vit_embed_cache.py
  - maas_finetune/trainer/rl/rollout/ray_driver.py
  - maas_finetune/trainer/rl/rollout/registry.py
  - maas_finetune/trainer/rl/template.py
  - maas_finetune/trainer/rl/actor.py
  - maas_finetune/trainer/rl/functional.py
  - 3rdparty/xllm/xllm/backend/inferencer/torch/torch_async_inferencer.py
  - 3rdparty/xllm/xllm/backend/processor/output_processor.py
confidence: medium
status: active
open_questions:
  - "23B full 8 节点 bsz=256 已验证到早期多 step 未复现 0.18 量级 logprobs_diff；仍需 70+ / 200 step 长线漂移验证。"
  - "xpu-gpt + Companion ViT 路径尚未适配 response-level vision_embeddings attach，不能直接套用 torch infer 的统一方案。"
contradictions: []
---

# Precision Alignment Report: torch infer VLM rollout response 直传 vit_embeds 统一 ViT cache 方案

## 1. 差异现象 (Deviation Symptoms)

- **现象**: 23B full 8 节点 torch inference 合并后，旧的 ViT cache/backfill 口径在 step1 出现 `train/rollout/logprobs_diff=0.181686`，明显劣于 merge 前 128p 基线 `0.0184433` 与 no-rollout-vit 对照 `0.0283056`。
- **数据类型 / 运行模式**: VLM RL，Actor BF16 训练，rollout 为 torch inference backend；2B5 单节点与 23B full 8 节点均涉及 ViT embedding 复用。
- **关键判断**: cache miss 本身最坏应退化为重算，不应造成 `0.18` 量级 diff。更可疑的是多节点下 `/dev/shm/vit_cache` 这类节点本地 cache 与 Actor/Rollout placement 不一致，或 async streamed embedding 取早导致样本 embedding 语义不稳定。

## 2. 当前 ViT cache 实现分层

### 2.1 xLLM torch infer 进程内 vision cache

- **位置**: `3rdparty/xllm/xllm/backend/inferencer/torch/torch_inferencer.py`
- **数据结构**: `LRUDict[int, torch.Tensor]`
- **开关**: `infer_cfg.use_vision_cache`，RL 脚本中通常由 `rollout.enable_device_cache=True` 打开。
- **作用边界**: 只在 torch inferencer 内部复用 vision embedding，命中时跳过 vision model；这是 rollout 内部性能 cache，不负责把 embedding 传给 Actor/ref。

### 2.2 RL 侧 `/dev/shm/vit_cache` 文件 cache

- **位置**: `maas_finetune/trainer/rl/rollout/vit_embed_cache.py`
- **默认目录**: `/dev/shm/vit_cache`
- **格式**: `image_hash.pt`
- **写入点**: `torch_async_inferencer.py` 在 rollout 得到 `vision_embedding` 后调用 `get_global_vit_cache().put(...)`。
- **读取点**: `actor.py::_inject_vit_cache()` 在 Actor old_logprobs / collect_amax / update 阶段按 raw hash 读取，并做跨 rank `all_gather`。任一 rank miss 时所有 rank 共同回退到 VIT recompute，避免 FSDP AllGather 死锁。
- **单机适配**: 适合单节点同容器内进程间共享。
- **多机限制**: `/dev/shm` 是节点本地目录，不是跨节点全局 cache。8 节点时不同节点的 Actor/Rollout 进程无法通过它共享同一份 embedding。

### 2.3 rollout response 直传 `vit_embeds`

- **写入/attach**:
  - `torch_async_inferencer.py` 将 `image_embeddings.detach().cpu()` 挂到 `GenerateOutput.vision_embeddings` 或 query 属性。
  - `output_processor.py` 在 final response 阶段 attach `vision_embeddings`。
- **RL 侧传递**:
  - `ray_driver.py` 优先从 final response 取 `vision_embeddings`，并校验 token length。
  - `registry.py` 放入 `RolloutResponse.extra["vit_embeds"]`。
  - `processor.py` / `template.py` 将 `vit_embeds` 放进 non-tensor batch。
  - `functional.py` 将 `vit_embeds` 转为 `image_embeds`，避免 Actor/ref 重跑 ViT。
- **开关**: `MAAS_VLM_USE_ROLLOUT_VIT_EMBEDS=1`。
- **统一优势**: embedding 跟随 sample 数据流走 Ray object / batch，而不是依赖节点本地文件系统；单节点和多节点语义一致。

## 3. 单机 / 多机统一方案

推荐统一到 **rollout response 直传 `vit_embeds`**：

```bash
export MAAS_VLM_USE_ROLLOUT_VIT_EMBEDS=1
export MAAS_VLM_KEEP_PIXEL_VALUES_WITH_VIT_EMBEDS=1
```

设计要求：

- torch inferencer 必须只在 final response 侧暴露完整 `vision_embeddings`，避免使用 async streamed 首包 embedding 作为最终样本特征。
- `ray_driver.py` 必须做 `sum(image_token_lengths) == vit_embeds.shape[0]` 校验；不匹配时丢弃该 `vit_embeds`，不要静默注入。
- `template.py` 必须校验 `vit_embeds.shape[0] == input_ids 中 image token 数`。
- `functional.py` 不能对 `vit_embeds` 做 sequence-dim SP 切片；它应作为完整 `image_embeds` 传入模型，由模型内部按 hidden-dim 做必要切分。
- `/dev/shm/vit_cache` 保留为 fallback：只有在 `MAAS_VLM_USE_ROLLOUT_VIT_EMBEDS=0 && ENABLE_VIT_CACHE=1` 时，Actor 才应走旧的文件 cache 注入路径。

## 4. xpu-gpt 路径边界

该统一方案当前只建议用于 **torch inference backend**。

xpu-gpt 路径仍依赖已验证的 xpu_gpt + Companion ViT 架构。它要接入 response-level `vit_embeds`，至少需要额外证明：

- xpu-gpt rollout 能在最终 response 上稳定拿到每个请求的完整 image embedding；
- image token length 与 response embedding shape 能逐样本严格对齐；
- Companion / xpu_gpt 的图文特征语义与 Actor/ref 所需 `image_embeds` 语义一致；
- 小 bsz 和 bsz=128 长跑均无 `GLOBAL MISS`、token mismatch、`logprobs_diff` 漂移。

在没有上述证据前，不应为了“统一”强行改 xpu-gpt 已验证路径。

## 5. 对齐验证 (Validation)

### 23B full 8 节点 response fallback 阶段证据

日志：

```text
outputs/logs/vlm_0606_070312_torch_infer_23b_full_8nodes_vit_responsefallback_bs256_long_bs256.log
```

截至本次记录，本地已拉取日志显示：

| step | logprobs_diff | logprobs_corr | final_rewards.mean |
|---:|---:|---:|---:|
| 1 | 0.0176206 | 0.995935 | 0.0458984 |
| 2 | 0.0170788 | 0.996003 | 0.0371094 |
| 3 | 0.0172579 | 0.995990 | 0.0449219 |
| 4 | 0.0173375 | 0.996032 | 0.0419922 |
| 5 | 0.0174159 | 0.995952 | 0.0434570 |
| 6 | 0.0172209 | 0.996010 | 0.0434570 |
| 7 | 0.0171108 | 0.996089 | 0.0556641 |

同一日志中统计到：

```text
forwarded final response vit_embeds = 14738
forwarded current vit_embeds        = 0
current image_embeddings missing    = 0
attached response vision_embeddings = 14738
token mismatch                      = 0
```

这说明当前 bsz=256 早期阶段没有复现旧 `0.181686` 异常，且没有落到 `current image_embeddings` 或 missing fallback。注意：这只是阶段性证据，不能替代 70+ / 200 step 长线漂移结论。

### 2B5 单节点历史口径

`outputs/logs/vlm_0602_113333_24189_torchinfer_align_ypq_bs128.log` 显示 2B5 单节点旧口径已经完成 200 step，最终 `logprobs_diff=0.0147604`，并大量出现 `Injected embedding to cache`。这说明旧 `/dev/shm/vit_cache` 在单节点可工作。

但从架构一致性看，后续 2B5 / 23B torch infer 更推荐显式打开 `MAAS_VLM_USE_ROLLOUT_VIT_EMBEDS=1`，让单机和多机都走 response 直传；旧 `/dev/shm/vit_cache` 仅保留为回退兼容路径。

## 6. 关联知识

- [[wiki/00_bugfix/vlm_torch_infer_vit_cache_sp_slice_logpdiff|torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸]]
- [[wiki/00_bugfix/vlm_template_reuse_rollout_vit_embeds|VLM RL Template 未复用 rollout vit_embeds 导致 Actor 侧重复图片编码]]
- [[wiki/02_precision/vlm_torch_infer_frozen_vlm_weight_sync_reward|torch infer frozen VLM 权重未同步导致 reward 低于合并前]]
- [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
