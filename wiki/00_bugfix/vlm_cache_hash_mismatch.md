---
title: "Bugfix: VLM Cache Miss Due to Cross-Boundary Hash Mismatch"
tags: [status/resolved, priority/high, category/caching]
last_updated: 2026-05-19
source: [logs/vlm_0519_172215_2450_verify_cache_miss_magnitude_bs2.log]
confidence: high
status: active
open_questions: []
contradictions: []
---

# Bugfix Report: VLM Cache Miss Due to Cross-Boundary Hash Mismatch

## 1. 现象描述 (Symptoms)
- **环境**: xLLM Engine, PyTorch (Actor), C++ Rollout Engine, FSDP
- **报错信息**:
  ```text
  [ViT-Cache-ACTOR-MISS] rank=13 img_idx=0/2 h_str='raw_image_md5[0]' hash=6597256185555529302 ... file_exists=False
  [ViT-Cache: update] GLOBAL MISS due to cross-rank mismatch: [...] — all ranks will RECOMPUTE together to prevent FSDP deadlock.
  ```
- **现象**: 在 VLM RL 训练中，开启了 `ENABLE_VIT_CACHE`，但 Actor 阶段无法命中 Rollout 阶段提取的图片特征缓存。导致 NPU 为了防止 FSDP 梯度计算死锁，强制触发所有 Rank 退回同步重算，ViT 特征旁路加速失效。

## 2. 根本原因分析 (Root Cause)
多模态特征预处理时的 Hash 指纹（Cache Key）在 Python Actor 端和 C++ Rollout 端产生撕裂：
- **Rollout 端 (XLLM)**：XLLM 的 `encoder_driver.py` 在向 Companion ViT Encoder 发送请求时，使用了 `VlmDialogProcessor.encode()` 提取的原生 `CityHash64`，并将其作为 `request_id`。推断底层会提取此 ID 作为存储特征文件的 Cache Key。
- **Actor 端 (Python)**：Actor 端在更新阶段使用的软 Hash (`hashlib.md5(pv_bytes)`) 或基于字节的二次 MD5 计算出的特征值与 Rollout 端的长整型 CityHash64 完全无法匹配，因此 Actor 端查找的 `dev/shm/vit_cache` 文件总是不存在。

## 3. 解决方案 (Solution)
### 终极方案 (Fix)
**打通 C++/Python 双端 Hash 标尺：透传 `image_hashes[0]`**
1. **Actor 端修改 (`template.py`)**：在预处理 `encode` 函数中，拦截 `VlmDialogProcessor` 生成的底层 CityHash64（即 `hash_img_data.image_hashes[0]`），放入 `image_raw_hashes` 并透传至 Actor 的 `_inject_vit_cache`。
2. **Actor 端修改 (`actor.py`)**：在查缓存时，优先读取 `image_raw_hashes` 作为缓存键去 `/dev/shm` 查询。
3. **Rollout 端修改 (`xpu_inferencer.py`)**：Companion Encoder 接收到图片特征抽取请求后，优先提取 `gen_ctx.request_ids[0]` (Rollout 透传的同一 Hash 键) 作为 Cache 保存的文件名。

## 4. 验证结果 (Verification)
修复后，日志中成功观测到：
```text
[ViT-Cache-XLLM] Computed hash_str: 'request_id_hash' -> hash: 1918643488377266023
[ViT-Cache: update] GLOBAL HIT — all ranks cached, using BYPASS path.
```
彻底打通缓存链路，避免了无意义的重复推理。

## 5. 沉淀与关联
- 关联知识点: VLM 训练缓存设计、FSDP 防死锁机制。
