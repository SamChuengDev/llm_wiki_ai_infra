---
title: CANN aclnnGather (EZ1001) 算子零长维报错与规避
tags: [npu, cann, gather, operator, OOM]
last_updated: "2026-04-14"
source: "unknown"
confidence: 5
status: "verified"
open_questions: []
---

# CANN aclnnGather 算子零长维度边界崩溃

## 症状描述

在 NPU (Ascend) 上的 PyTorch 分布式推理模型中，当下游使用 `torch.gather()` 时，有一定几率遇到以下 Traceback：
```python
RuntimeError: gather:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:69 NPU function error: call aclnnGather failed, error code is 161002
[ERROR] 2026-04-14-17:27:41 (PID:3976217, Device:11, RankID:11) ERR00100 PTA call acl api failed.
AclNN_Parameter_Error(EZ1001): Size does not match at dimension 1, expected index shape to be 0
```
错误信息往往让人迷惑，因为 `expected index shape to be 0` 似乎在提示你需要一个零大小的索引张量。

## 根因与触发条件

在标准 CUDA/CPU 后端上，如果被收集张量 `input` 的 `dim` 维度长度为 0（即 `input.size(dim) == 0`），并且 `index` 也是大小适宜的张量（没有取任何元素因为取不到，或者 `input` 本身无体积），通常会静默放行或报一般的由于 `index` 非法引起的问题。
但在 CANN 算子框架中：**`aclnnGather` 对被提取维度尺寸为 0 会展现完全零容忍的保护崩溃（EZ1001），并且报出的 expected index shape 极其误导**。

**典型触发场景：**
1. 框架使用了分布式全局 Padding/Dummy batches：如 VLM RL 推理为了防死锁，会将处于空载（`actual_batch_size=0`）的 Request 強行同步走 Prefill 或 Decode。
2. 导致前置模型注意力输出的张量在 Sequence 维度是 0。
3. 下游降维或抽取首尾 Token 时（如 `flattn_output_to_2d` 中的 `torch.gather`）面对这个 Sequence 维度为 0 的产物进行采样。

## 避坑与解决规范

在所有从注意力或大段张量提取元素的过程前，必须加入对当前生成阶段**实际有效特征序列长度**的防御性校验：

```python
# 错误做法：直接采集底层可能由于 Dummy Batch 返回序列为 0 的状态
index_tensor = index.repeat(1, 1, hidden_states.shape[-1])
_hidden_list.append(torch.gather(hidden_states, dim_axis, index_tensor))

# ✅ 正确做法：提前发现空序列短路并伪造等长空状态（规避调用 CANN 算子）
if hidden_states.shape[dim_axis] == 0:
    return torch.zeros(
        (len(next_token_index), hidden_states.shape[-1]), 
        dtype=hidden_states.dtype, 
        device=hidden_states.device
    )

# （可选）安全保险：严防负索引（-1）引发的未定义泄漏
index_tensor = torch.clamp(index_tensor, min=0)
_hidden_list.append(torch.gather(hidden_states, dim_axis, index_tensor))
```

所有新增涉及序列下沉采样（Gather、IndexSelect）逻辑的代码，强烈要求加上这种尺寸阈值短路拦截！
