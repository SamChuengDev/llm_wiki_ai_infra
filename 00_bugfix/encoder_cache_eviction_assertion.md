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
- **环境**: NPU (Ascend 910B), xLLM 异步推理引擎。
- **报错信息**:
  在并发压测（如 concurrency=16，多轮 Rounds）下，服务运行一段时间后崩溃并抛出如下异常：
  ```text
    File "/data01/zcs/maas_engine_vlm2b5/3rdparty/xllm/xllm/backend/driver/encoder_result_cache.py", line 105, in put
      self.token_len -= lru_query.mm_token_length
    File "/data01/zcs/maas_engine_vlm2b5/3rdparty/xllm/xllm/backend/driver/loops/encoder_loop.py", line 42, in mm_token_length
      assert self.sequence.multi_modal_data is not None
  AssertionError
  ```

## 2. 根本原因分析 (Root Cause)
1. 在 xLLM 的异步编码器循环 `EncoderLoop` 中，每当一个 `ImageEncoderQuery` 请求处理完毕时，会调用 `finish_query` 及时回收资源并清空 `multi_modal_data`（防止多卡图片张量常驻显存引发 OOM）：
   ```python
   def finish_query(self, query: GenerateQuery):
       assert isinstance(query, ImageEncoderQuery)
       query.sequence.multi_modal_data = None
       super().finish_query(query)
   ```
2. 同时，系统为了支持多模态图像缓存，使用 `EncoderResultCache`（LRU 缓存）来缓存已编码的图像特征。
3. 当 LRU 缓存满执行淘汰驱逐时，会从缓存中弹出旧的 `lru_query`，并通过读取 `lru_query.mm_token_length` 扣减缓存当前的总 token 数目：
   ```python
   lru_key, lru_query = self._cache.popitem(last=False)
   self.token_len -= lru_query.mm_token_length
   ```
4. 如果被淘汰的 Query 刚好是已经生成完毕并且调用过 `finish_query` 清空了 `multi_modal_data` 的历史 Query，访问 `mm_token_length` 属性就会因为 `self.sequence.multi_modal_data is None` 触发断言错误，导致服务整个崩溃。

## 3. 解决方案 (Solution)
### 终极方案 (Fix)
在 `loops/encoder_loop.py` 中对 `ImageEncoderQuery` 进行容错性修改：
1. **持久化关键元数据**：在 `from_request` 初始化创建 `ImageEncoderQuery` 时，直接将图像的 token 长度和 `hash_id` 缓存为 Query 的实例属性（例如 `self.token_len` 和 `self.hash_id`）。
2. **优雅回退（Graceful Fallback）**：修改 `mm_token_length` 和 `mm_data_hash_id` 属性的获取逻辑。若 `self.sequence.multi_modal_data` 尚未被清空，则正常从其内部读取；若已被清空为 `None`，则直接回退返回创建时保存的 `self.token_len` 和 `self.hash_id` 属性，避免断言失败。

代码改动示例：
```python
class ImageEncoderQuery(EncoderQuery):
    # ...
    @property
    def mm_token_length(self):
        if self.sequence.multi_modal_data is None:
            return self.token_len
        assert self.sequence.multi_modal_data.image_data is not None
        return self.sequence.multi_modal_data.image_data.image_token_lengths[0]

    @property
    def mm_data_hash_id(self):
        if self.sequence.multi_modal_data is None:
            return self.hash_id
        assert self.sequence.multi_modal_data.image_data is not None
        return str(self.sequence.multi_modal_data.image_data.image_hashes[0])

    @classmethod
    def from_request(cls, query: GenerateQuery) -> ImageEncoderQuery:
        # ...
        encoder_query.token_len = query.sequence.multi_modal_data.image_data.image_token_lengths[0]
        encoder_query.hash_id = str(query.sequence.multi_modal_data.image_data.image_hashes[0])
        return encoder_query
```

## 4. 验证结果 (Verification)
并发启动 16 请求并发测试，三轮共 96 次请求无一例崩溃，且 100% 成功生成。服务侧的 LRU 缓存平滑完成 eviction 动作，CPU/NPU 显存表现完全符合预期。
