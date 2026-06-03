---
title: "Bugfix: xLLM 并发压测下 EncoderResultCache 驱逐触发 AssertionError"
tags: [status/fixed, priority/high, category/concurrency]
last_updated: 2026-05-20
source: [outputs/xllm_infer_logs/dp16ep16_srv_*.log]
confidence: high
status: active
open_questions: []
contradictions: []
---

# Bugfix Report: xLLM 并发压测下 EncoderResultCache 驱逐触发 AssertionError

## 1. 现象描述 (Symptoms)
- **环境**: NPU Ascend 910B，xLLM 异步推理引擎。
- **报错信息**:
  并发压测运行一段时间后，服务端在 `EncoderResultCache.put()` 驱逐旧缓存时崩溃：
  ```text
  File ".../encoder_result_cache.py", line 105, in put
    self.token_len -= lru_query.mm_token_length
  File ".../encoder_loop.py", line 42, in mm_token_length
    assert self.sequence.multi_modal_data is not None
  AssertionError
  ```

## 2. 根本原因分析 (Root Cause)
`EncoderLoop.finish_query()` 为了避免图片张量常驻显存，会在请求完成后清空 `query.sequence.multi_modal_data`。但 `EncoderResultCache` 的 LRU 驱逐逻辑仍通过旧 `ImageEncoderQuery.mm_token_length` 属性回读 `multi_modal_data` 来扣减 token 数。

如果被驱逐的 query 已经 finish，`multi_modal_data` 为 `None`，属性访问触发断言，导致服务进程崩溃。

## 3. 解决方案 (Solution)
在 `ImageEncoderQuery.from_request()` 创建 query 时持久化驱逐所需的关键元数据，例如 `token_len` 与 `hash_id`；随后修改 `mm_token_length` 和 `mm_data_hash_id` 属性：

```python
@property
def mm_token_length(self):
    if self.sequence.multi_modal_data is None:
        return self.token_len
    return self.sequence.multi_modal_data.image_data.image_token_lengths[0]

@property
def mm_data_hash_id(self):
    if self.sequence.multi_modal_data is None:
        return self.hash_id
    return str(self.sequence.multi_modal_data.image_data.image_hashes[0])
```

这样缓存驱逐不再依赖已被释放的 `multi_modal_data`。

## 4. 验证结果 (Verification)
并发 16 请求、多轮共 96 次压测无崩溃，LRU cache eviction 正常完成，CPU/NPU 显存表现符合预期。

## 5. 关联知识
- [[wiki/00_bugfix/vlm_companion_vit_rpc_crash|VLM Companion ViT RPC 连接失败与环境变量断层]]
- [[wiki/05_fact/vlm_2b5_vit_llm_deployment_architecture|Fact: VLM 2B5 VIT与LLM的混合部署架构约束]]
