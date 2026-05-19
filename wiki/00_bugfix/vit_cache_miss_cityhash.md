---
title: "Bugfix: VLM ViT 缓存穿透与 CityHash 跨端不对齐导致的 100% Cache Miss"
tags: [status/resolved, priority/high, category/cache]
last_updated: 2026-05-18
source: [outputs/logs/vlm_0518_223521_2450_verify_cache_miss_magnitude_bs2.log]
confidence: high
status: resolved
open_questions: []
contradictions: []
---

# Bugfix Report: VLM ViT 缓存穿透与 CityHash 跨端不对齐导致的 100% Cache Miss

## 1. 现象描述 (Symptoms)
- **环境**: NPU 910B, W8A8 RL 训练
- **报错信息**: 
  - Actor 侧持续抛出 `[ViT-Cache-ACTOR-MISS]` 日志，显示 `vit_cache` 100% 缓存未命中。
  - Rollout 侧写入的 cache hash（如 `1918643488377266023`，19位数）与 Actor 侧查找的 hash（如 `224859422057812115197456843597838152606`，39位数）存在明显的位数与格式差异。
  - 缓存穿透不仅导致了巨大的冗余计算开销，多进程的重算耗时差异还偶发导致 FSDP `AllGather` 超时死锁，且强行通过 Tensor soft hash 兜底时会引入 `1 ULP` 的精度底噪漂移。
- **复现脚本**: `bash ray.sh` (使用 `seed_vl_prompt`)

## 2. 根本原因分析 (Root Cause)
问题源于 **跨进程哈希算法不匹配 (Cross-Process Hash Algorithm Mismatch)**：
- **Rollout 侧 (xLLM Engine)**: 图像处理使用 `vlm_dialog_processor`，其内部 `_calculate_image_hash` 使用 `CityHash64` 算法计算原始字节和元数据的哈希，产出 19 位整数。
- **Actor 侧 (RLTemplate)**: 图像处理使用 `seed_vl_prompt` processor，其默认的哈希生成使用了标准的 `CityHash128`，产出了 39 位整数。
- 两端对于同一张原始图像字节产生了不同的哈希键值，导致跨进程共享内存 `/dev/shm/vit_cache` 永远无法命中。

## 3. 解决方案 (Solution)
### 临时方案 (Workaround)
- [x] 在 Actor 侧利用软哈希 (`torch.round(pv * 1e4) / 1e4`) 对预处理后的 `pixel_values` 进行截断取 MD5 来进行次级缓存命中兜底，但预处理底层的浮点指令集差异仍可能引发 `1 ULP` 误差。

### 终极方案 (Fix)
在 Actor 侧的 `template.py` 中绕过 `seed_vl_prompt` 默认的哈希结果，采用与 Rollout 侧严格 1:1 对齐的内联代码，重新基于原图 base64 解析出的原始 bytes 算出精确的 `CityHash64`：
- [x] **统一处理入口**: 保留 `seed_vl_prompt` 作为统一的 tokenization 和 pixel_values 处理路径（不可替换为 `ark_vlm_dialog`，因其 `__init__` 不兼容）。
- [x] **复刻 Rollout 的哈希生成逻辑**: 手动提取 raw bytes，使用等效于 `vlm_dialog_processor` 的公式：
  ```python
  import struct, cityhash
  _suffix = b"\x00" + struct.pack("d", 1.0)  # use_anyres=False, resize_ratio=1.0
  _hash_u64 = cityhash.CityHash64(raw_image_bytes + _suffix)
  raw_image_hash = str(int.from_bytes(_hash_u64.to_bytes(8, "big", signed=False), "big", signed=True))
  ```
- [x] **将正确的哈希透传至 batch**: 在 `encode` 函数内返回 `image_raw_hashes`，加入到 `rl_trainer` 的 `non_tensor_batch_keys` 白名单，最终在 `actor.py` 内部经 `_flatten_nested_list` 展平后与切割后的 `pv_list` 一一对应，完成精确缓存查询。

## 4. 验证结果 (Verification)
执行修改后的代码，无论在 Batch Size 2 还是 128 (并发含多图) 下，均确认：
1. Actor 端输出: `[ViT-Cache: old_logprobs] GLOBAL HIT — all ranks cached, using BYPASS path.`
2. Engine 端输出: `[ViT-Cache-XLLM] HIT in vit_cache, skipping ViT forward pass.`
缓存 100% 精确命中，成功消除了底层浮点 ULP 引入的微小漂移变量，并提升了训练步速与并发稳定性。

## 5. 沉淀与关联
- 关联知识点: [[wiki/02_precision/rl_off_policy_inconsistency|训推架构 Off-Policy 异构发散诱因与对策]]
