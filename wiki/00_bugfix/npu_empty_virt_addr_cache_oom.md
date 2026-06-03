---
title: "Bugfix: empty_virt_addr_cache 导致 NPU 物理显存未归还引发 OOM"
tags: [status/resolved, priority/critical, category/oom]
last_updated: 2026-05-06
source: [outputs/logs/vlm_0506_201101.log, torch_npu/csrc/core/npu/NPUCachingAllocator.cpp]
confidence: high
status: active
open_questions: []
contradictions: []
---

# Bugfix Report: empty_virt_addr_cache 导致多进程架构下 NPU 物理显存泄露 OOM

## 1. 现象描述 (Symptoms)
- **环境**: NPU 61GB，VLM 2.5B RL 训练，EngineWorker 与 ActorWorker 多进程同卡架构，开启 `expandable_segments:True`。
- **报错信息**:
  ```text
  RuntimeError: NPU out of memory. Tried to allocate 194.00 MiB ...
  ```
- **复现场景**:
  EngineWorker 做了 Python 对象回收和 `empty_virt_addr_cache()` 后，`npu-smi` 物理显存仍居高不下；ActorWorker 随后在 `optimizer.step()` 阶段稳定 OOM。

## 2. 根本原因分析 (Root Cause)
`torch.npu.empty_virt_addr_cache()` 底层只解绑虚拟地址映射，等价于保留 `free_physical=false`。物理显存句柄仍留在当前进程的缓存池中，没有归还给 NPU 驱动。

在 EngineWorker 与 ActorWorker 分属不同进程时，EngineWorker 私有缓存池持有的物理显存无法被 ActorWorker 复用，从而形成跨进程假性 OOM。

## 3. 解决方案 (Solution)
在所有调用 `torch.npu.empty_virt_addr_cache()` 的位置，紧随其后调用 `torch.npu.empty_cache()`：

```python
torch.npu.empty_virt_addr_cache()
torch.npu.empty_cache()
```

`empty_cache()` 会释放物理句柄，使物理显存真正归还给驱动层，供其它进程重新申请。

## 4. 验证结果 (Verification)
修复后日志显示 EngineWorker 释放后显存占用从约 18GB 降到 `6.62GB`；ActorWorker 启动前可用显存恢复到 `33.33GB / 61.28GB`，并成功完成 optimizer step。

## 5. 关联知识
- [[wiki/00_bugfix/pytorch_to_empty_libc_leak_and_precision|Bugfix: PyTorch to_empty() 精度丢失与 libc 碎片化泄漏连环坑]]
- [[wiki/05_fact/torch_infer_merge_debug_checklist|Fact: xpu_gpt + torch infer 合并验证排查清单]]
