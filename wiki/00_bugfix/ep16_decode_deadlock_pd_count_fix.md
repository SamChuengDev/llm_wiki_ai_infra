---
title: "EP16 MoE Decode Deadlock: pd_count Missing Prefill-Ready Requests"
date: 2026-04-28
tags: [ep16, moe, deadlock, all_to_all, hccl, decode, dp_sync]
last_updated: 2026-04-28
source:
  - tests/xllm_xg_st/xllm_client_llm.sh
  - 3rdparty/xllm/ark_text_stress_test.py
severity: critical
status: fixed
confidence: high
open_questions: []
---

# EP16 MoE Decode Deadlock — pd_count Missing Prefill-Ready Requests

## 症状

单个纯文本请求发到 DP16/EP16 VLM 服务后，**服务器端卡死不产出任何 token**。
`MIX_SCHED_STATS` 停止输出，`QueryStatesWatchDog` 显示请求持续处于 `GenerationStage.PREFILL` 状态。TTFT 无限超时，客户端永远挂起。

## 根因分析

死锁发生在 **prefill → decode 切换的 Gloo all_gather 窗口期**：

1. Rank 0（请求 owner）完成最后一个 prefill chunk：`computed_length = prompt_len`
2. `_process_prefill_step_results()` 更新 `running_prefill_querys` 状态并通过 ZMQ 广播
3. 在 **下一次 `_schedule()`** 开头，`pd_count[1]` 按如下计算：
   ```python
   pd_count[1] = len(waiting_decode) + len(running_decode_querys)
   ```
   此时请求还在 `running_prefill_querys`（尚未转移到 `waiting_decode`），所以 `pd_count[1] = 0`
4. Gloo all_gather 广播：所有 rank 看到 `d_counts = [0,0,...,0]`
5. Rank 1~15 判断 `max_d = 0` → `get_idle_signal() = True` → 进入 `wait_for_new_query(20ms)`
6. Rank 0 在同一次 `_schedule()` 内将请求移到 `waiting_decode`，随后构建 decode context，进入 **NPU EP16 decode forward**（含 MoE all-to-all HCCL）
7. **HCCL all-to-all 要求所有 16 个 rank 同时参与**，但 rank 1~15 在 sleep，死锁！

## 修复

文件：`3rdparty/xpu_gpt/xpu_gpt/adapter/xllm_adapter/xpu_async_spec_rl_inferencer.py`

在 `_schedule()` 的 `pd_count` 计算中，加入"已完成 prefill 但尚未转移到 decode 队列"的请求数：

```python
# Count requests that finished prefill but haven't been moved to waiting_decode yet.
# Without this, all_gather broadcasts d_count=0 and other EP ranks enter idle,
# skipping the decode NPU forward and deadlocking the MoE all-to-all collective.
_prefill_ready_for_decode = sum(
    1 for _rid, _rq in self.running_meta.running_prefill_querys.items()
    if _rid in self._unfinished_querys
    and _rq.sequence.computed_length + _rq.in_flight_context.inflight_length >= _rq.sequence.prompt_len
)

pd_count = [
    len(self._waiting_prefill_querys) + _has_continuing_prefill,
    len(self._waiting_decode_querys) + len(self.running_meta.running_decode_querys) + _prefill_ready_for_decode,
    int(self.stop_flag),
    _local_should_prefill,
]
```

## 验证

1. **单请求文本测试**（`xllm_client_llm.sh` Phase 1）：TTFT ≈ 7.5s，TPOT ≈ 50ms/token，`finish_reason=stop`，31 tokens 正常输出。
2. **20并发精度压测**（`ark_text_stress_test.py --concurrency 20`）：
   - Phase 1 并发 20 请求：**11.27s 完成，0 errors**
   - 16/20 MATCH，4个 DIFF（相似度 83-98%）均为 greedy decoding 在 MoE batch 下的同义词分叉，非数据串扰
   - NPU forward 时间从 **5.4s → 50ms**（减少 100x），证明 EP all-to-all 通信正常

## 关键约束

- 此 bug 仅在 EP16（BigEP）模式下触发，TP/DP 纯数据并行不受影响
- `XPU_FORCE_USE_BIGEP="1"` 激活时必然触发此场景
- `inflight_context.inflight_length` 需正确反映当前 chunk 大小，与 `computed_length` 共同判断 prefill 完成

## 相关文件

- `3rdparty/xpu_gpt/xpu_gpt/adapter/xllm_adapter/xpu_async_spec_rl_inferencer.py`: `_schedule()` L303-348
- `tests/xllm_xg_st/xllm_client_llm.sh`: 验证脚本（Phase 1 + Phase 2 精度压测）
- `3rdparty/xllm/ark_text_stress_test.py`: 精度压测主脚本
