---
title: "Bugfix: torch infer VLM 默认 env 覆盖 scheduler/model batch 导致 RL step 劣化"
tags: [status/fixed, priority/high, category/bugfix, component/torch_infer, component/xllm, component/rl]
last_updated: 2026-06-02
source:
  - outputs/logs/vlm_0601_205541_24189_torchinfer_align_ypq_bs128.log
  - /Users/bytedance/posttrain/codes/maas_engine_torch_infer/outputs/logs/vlm_0528_002344.log
  - outputs/run_time_envs/runtime_env_0601_205541_torch_infer.yaml
confidence: high
status: active
open_questions:
  - "尚需新一轮 RL 日志确认 AsyncDriverProxy 中 max_schedule_batch_size/model max_batch 均为 512，并观察 step time 是否恢复到源目录约 800s 量级。"
contradictions: []
---

# Bugfix: torch infer VLM 默认 env 覆盖 scheduler/model batch 导致 RL step 劣化

## 现象

合并仓 torch infer 2B5 bsz=128 日志 `outputs/logs/vlm_0601_205541_24189_torchinfer_align_ypq_bs128.log` 中，单个 step 明显慢于源目录对照：

- step elapsed time: `1131.7309s`
- `train/timing/data=966.465s`
- `train/rollout/logprobs_diff=0.0180878`

源目录 `/Users/bytedance/posttrain/codes/maas_engine_torch_infer/outputs/logs/vlm_0528_002344.log` 在同类配置下 step 多在 `726~807s`，因此本问题主要表现为 rollout 调度性能退化，而不是精度爆炸或 Traceback 崩溃。

## 失败签名

Hydra/rollout 配置里看起来已经设置为大 batch：

```text
scheduler_max_schedule_batch_size=512
scheduler_max_schedule_context_batch_size=512
model_max_batch_size=512
```

但 xLLM 实际启动时打印出的配置被降成了 64：

```text
sched_cfg: OrcaSchedulerConfig(... max_schedule_batch_size=64, max_schedule_context_batch_size=512, ...)
model_cfg: TorchModelConfig(... max_batch_size=64, ...)
```

运行态调度证据也匹配该退化：

- merge 慢日志 `QueryStatesWatchDog`: `DECODING:64`，同时 `PREFILL:134` 堆积。
- 源目录健康日志 `QueryStatesWatchDog`: 可见 `DECODING:256`。
- merge 慢日志 async driver 的 `first_token_duration`、`enqueue_duration`、`generate_waiting_time` 可到百秒级；源目录同类请求多为 `3~5s`。

按请求级统计，merge 慢日志平均 first token 约 `294s`，p90 约 `572s`；源目录对照平均 first token 约 `4.16s`，p90 约 `4.69s`。

## 根因

`maas_finetune/trainer/rl/rollout/ray_driver.py::_setup_envs()` 先从 rollout 配置写入显式 512：

```python
envs["XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE"] = cfg.get("scheduler_max_schedule_batch_size", 128)
envs["XLLM_SCHEDULER_MAX_SCHEDULE_CONTEXT_BATCH_SIZE"] = cfg.get("scheduler_max_schedule_context_batch_size", 16)
envs["XLLM_MODEL_MAX_BATCH_SIZE"] = cfg.get("model_max_batch_size", 512)
```

随后 VLM 分支执行：

```python
if cfg.get("enable_vlm", False):
    envs.update(VLM_DEFAULT_ENVS)
```

而 `VLM_DEFAULT_ENVS` 内含：

```python
"XLLM_MODEL_MAX_BATCH_SIZE": 64,
"XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE": 64,
```

因此显式配置的 512 被 VLM 默认值覆盖成 64。该覆盖限制了 active decode query 数，使剩余请求长期停在 PREFILL/排队阶段，放大 `train/timing/data`，导致单 step 劣化约 `300s+`。

这不是 actor/ref 侧 ViT cache miss 导致的重复 encode 问题。actor/ref 缓存复用修复后可见 `actor_ref_encode` 已不再是主要增量，但 xLLM rollout 调度仍被 64 上限卡住。

## 修复

修复策略是在合并 `VLM_DEFAULT_ENVS` 后，重新应用用户显式传入的 rollout 调度配置：

```python
if cfg.get("enable_vlm", False):
    envs.update(VLM_DEFAULT_ENVS)
    if cfg.get("scheduler_max_schedule_batch_size", None) is not None:
        envs["XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE"] = cfg.get("scheduler_max_schedule_batch_size")
    if cfg.get("model_max_batch_size", None) is not None:
        envs["XLLM_MODEL_MAX_BATCH_SIZE"] = cfg.get("model_max_batch_size")
    envs["XLLM_PROCESSOR_INPUT"] = "seed_vl_prompt"
```

如果配置没有显式传值，仍保留 VLM 默认 64，避免改变旧的保守默认行为。

## 验证

已完成的验证：

- 本地 `python3 -m py_compile maas_finetune/trainer/rl/rollout/ray_driver.py` 通过。
- 单文件同步到 2450 后，远端 `python3 -m py_compile` 通过。
- 远端 `_setup_envs` 烟测：
  - 显式配置: `sched=512`, `ctx=512`, `model=512`, `processor=seed_vl_prompt`
  - 默认配置: `sched=64`, `ctx=16`, `model=64`, `processor=seed_vl_prompt`
- 远端 xLLM Launcher 烟测确认 `sched_cfg.max_schedule_batch_size=512`、`max_schedule_context_batch_size=512`、`model max_batch=512`。

尚未完成的验证：

- 需要新一轮 RL 日志确认 `AsyncDriverProxy starting` 中 `max_schedule_batch_size=512` 与 `model max_batch_size=512`。
- 需要观察实际 step time 是否从 `1131s` 回落到源目录约 `800s` 量级；该验证完成前只能说配置覆盖路径已修复，不能宣称端到端性能已完全恢复。

## 排查提示

后续遇到 “日志里 Hydra 配置正确，但 xLLM 实际启动参数不正确” 的问题，应同时核对：

1. `outputs/logs/vlm_MMDD_HHmmss*.log` 中的 Hydra/rollout 配置。
2. 同时间戳 `outputs/run_time_envs/runtime_env_MMDD_HHmmss*.yaml`。
3. xLLM 启动时 `AsyncDriverProxy starting` 打印的 `sched_cfg` 和 `model_cfg`。
4. `ray_driver._setup_envs()` 中 env 写入顺序，尤其是默认 env 是否覆盖了显式配置。
