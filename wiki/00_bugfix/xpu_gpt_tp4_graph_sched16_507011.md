---
title: "Bugfix: xpu_gpt TP4 图模式 sched16 并发触发 507011"
tags: [status/verified, priority/high, category/bugfix, xpu_gpt, tp4, graph-mode, scheduler, npu]
last_updated: 2026-07-03
source:
  - outputs/logs/vlm_0703_115656_tp4_graph_nonv2_sched8_rayxg_long200_bs8.log
  - outputs/run_time_envs/runtime_env_0703_115656.yaml
  - outputs/logs/vlm_0703_130100_tp4_graph_nonv2_sched16_rayxg_100_bs8.log
  - outputs/run_time_envs/runtime_env_0703_130100.yaml
  - runtime_env_xg_tp4_dp4.yaml
  - xpu_gpt/xpu_gpt/backend/npu/operators/attention/page_attn.py
  - 3rdparty/xllm/xllm/backend/inferencer/sampler.py
confidence: medium
status: active
open_questions:
  - "sched16 的 507011 是否由 decode 并发批量增大触发的随机数 NPU 拷贝或前序图算子异步错误，需要 plog/最小重放确认。"
  - "sched8 长跑是否能稳定到 70+ / 200 step 仍待验证。"
  - "是否可用 sched12 等中间值提升吞吐仍待单独实验。"
contradictions: []
---

# Bugfix Report: xpu_gpt TP4 图模式 sched16 并发触发 507011

## 1. 现象描述 (Symptoms)
- **环境**: 2B5 xpu_gpt W8A8 rollout，单机 2450，TP4/DP4/EP16，`XPU_DECODE_COMPILE=1`，`MAAS_PARALLEL_CONFIG=tp4`。
- **触发条件**: 在非 V2 page attention 图模式基础上，将 TP4 rollout 调度批量从 sched8 放大到 sched16：
  - `XLLM_MODEL_MAX_BATCH_SIZE=16`
  - `MAX_PREFILL_BATCH=16`
  - `XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE=16`
  - `XLLM_SCHEDULER_MAX_SCHEDULE_CONTEXT_BATCH_SIZE=16`
  - `MIN_PREFILL_BATCH=16`
- **失败签名**: sched16 run 已完成 `global_step=1`，随后下一轮 rollout 先出现 QueryStatesWatchDog 长等待，再在 `xworker4` 抛出 `ACL stream synchronize failed, error code:507011`。driver 侧 gloo/RPC 关闭属于后续级联。
- **首个实锤 traceback**:
  ```text
  [xworker4] File ".../xpu_gpt/adapter/xllm_adapter/spec_rl.py", line 258, in async_step
      current_sample_result = self.verification(gen_ctx, draft_tokens)
  [xworker4] File ".../xpu_gpt/adapter/xllm_adapter/xpu_async_spec_rl_inferencer.py", line 1677, in verification
      sample_result = self.sampling(gen_ctx, draft_tokens, logits)
  [xworker4] File ".../xpu_gpt/adapter/xllm_adapter/xpu_async_spec_rl_inferencer.py", line 1708, in sampling
      sample_result = self.reject_sampler.sample(...)
  [xworker4] File ".../3rdparty/xllm/xllm/backend/inferencer/sampler.py", line 1706, in _reject_sample_decode_requests
      _rand = _rand.to(device, non_blocking=False)
  RuntimeError: ACL stream synchronize failed, error code:507011
  ```
- **对照**: 同一远端验证目录、同模型、同图模式下，sched8 run 已跑到 `global_step=10`，未命中 `Traceback` / `507011` / `RayActorError`。

## 2. 根本原因分析 (Root Cause)
- **已确认事实**: TP4 图模式不是整体不可用。非 V2 page attention + sched8 调度口径已连续推进到 `global_step=10`；放大到 sched16 后，在 step1 后的下一轮 rollout 触发 507011。
- **已确认事实**: sched16 的首个 Python 栈落在 reject sampler 的 `_rand.to(device)`，但 507011 是 ACL stream synchronize 类错误，报错行可能只是同步暴露点，不足以证明随机数拷贝本身就是根因。
- **高度怀疑**: sched16 同时放大 model/prefill/scheduler/context/min prefill batch 到 16，增加了 decode 图模式和采样链路的并发压力；真正的 NPU stream 异常可能发生在 `_rand.to()` 之前的图模式 decode 或采样前序算子。
- **非同一问题**: 这次失败不同于 [[wiki/00_bugfix/xpu_gpt_decode_graph_cache_maxlen_mismatch|xpu_gpt decode graph cache 未按 max_length 隔离导致 12k->32k 图模式断言]]。旧问题签名是 stale graph cache 的 shape/stride assertion；本次签名是 sched16 下 sampler 同步点暴露 `507011`。

## 3. 解决方案 (Solution)
### 临时方案 (Workaround)
- [x] TP4 图模式先固定回退到已验证 sched8：
  ```yaml
  XLLM_MODEL_MAX_BATCH_SIZE: "8"
  MAX_PREFILL_BATCH: "8"
  XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE: "8"
  XLLM_SCHEDULER_MAX_SCHEDULE_CONTEXT_BATCH_SIZE: "8"
  MIN_PREFILL_BATCH: "8"
  ```
- [x] 保留非 V2 page attention decode 路径，规避 TP4 图模式 V2 path 的 507011：
  ```text
  flash_attention_decode_mtp_tensor(..., self.causal_mask_eager, ..., self.kv_head_num, -1)
  ```
- [x] `run_xg_vlm_2b5.sh` 保留 `actor.dp_mode=fsdp1`，与远端验证通过口径一致。
- [x] 不降低 `BATCH_SIZE` / `rollout.generation.n` 等 RL 算法超参来掩盖调度问题。

### 后续修复方向 (Fix)
- [ ] 若要提升吞吐，先小步试 sched12，而不是直接恢复 sched16。
- [ ] 对 sched16 失败补充 plog、NPU stream 错误前序栈或最小重放，确认 507011 的真实触发点。
- [ ] 将图模式 attention、reject sampler 和 scheduler batch 三者拆开 A/B，避免把同步暴露点误写成算子根因。

## 4. 验证结果 (Verification)
- **通过口径**: `vlm_0703_115656_tp4_graph_nonv2_sched8_rayxg_long200_bs8.log`
  - runtime env: `runtime_env_0703_115656.yaml`
  - 已观测到 `global_step=10`
  - `global_step=1` 的 `logprobs_diff=0.0385178`，`reward.mean=0.21875`
  - `global_step=2~10` 的 `logprobs_diff` 约为 `0.021~0.029`
  - reward 不是全零；其中 step8 为 0，但其他 step 非零
  - fatal 扫描未命中 `Traceback` / `507011` / `RayActorError`
- **失败口径**: `vlm_0703_130100_tp4_graph_nonv2_sched16_rayxg_100_bs8.log`
  - runtime env: `runtime_env_0703_130100.yaml`
  - 已观测到 `global_step=1` 后失败
  - `global_step=1` 的 `logprobs_diff=0.0381598`，`reward.mean=0.265625`
  - 首个实锤错误为 `sampler.py:_reject_sample_decode_requests` 中 `_rand.to(device)` 暴露的 `ACL stream synchronize failed, error code:507011`
- **结论边界**: sched8 是已验证 workaround，可支撑 TP4 图模式继续推进；这不是 70+ / 200 step 长线精度稳定结论，也不是 sched16 根因闭环。

## 5. 沉淀与关联
- 关联知识点: [[wiki/00_bugfix/xpu_gpt_decode_graph_cache_maxlen_mismatch|xpu_gpt decode graph cache 未按 max_length 隔离导致 12k->32k 图模式断言]]
- 关联知识点: [[wiki/03_tuning/xpu_gpt_w8a8_rl_scheduler_memory_tuning|xpu-gpt W8A8 RL scheduler batch 与显存水位调优]]
- 关联知识点: [[wiki/00_bugfix/ep16_decode_deadlock_pd_count_fix|EP16 MoE Decode Deadlock pd_count 修复]]
