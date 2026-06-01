---
title: "Bugfix: xpu_gpt decode graph cache 未按 max_length 隔离导致 12k->32k 图模式断言"
tags: [status/verified, priority/high, category/bugfix, xpu_gpt, graph-cache, vlm-rl]
last_updated: 2026-06-01
source:
  - outputs/logs/vlm_0601_170605_24140_xg_graph_bs128.log
  - outputs/logs/vlm_0601_172008_24140_xg_graph_bs128.log
  - outputs/logs/vlm_0526_105736_2450_allcodes_ge_graph_bs128_filter_432steps.log
confidence: high
status: active
open_questions:
  - "当前验证覆盖到 global_step=3，证明 65==25 图模式断言已跨过；仍不能替代 70+ step / 200 step 的长线精度漂移验证。"
contradictions: []
---

# Bugfix Report: xpu_gpt decode graph cache 未按 max_length 隔离导致 12k -> 32k 图模式断言

## 1. 现象描述 (Symptoms)

- **环境**: `24140` / `mass_torch`，VLM 2B5 RL，16 NPU，xpu_gpt 后端，`XLLM_MODEL_CLS=xpu`，`XPU_DECODE_COMPILE=1`，`BATCH_SIZE=128`。
- **触发背景**: 旧图模式在较短长度口径下曾跑通 432 step；合并 torch infer 后，将长度参数如 `data.max_length` / rollout max sequence length 调到 `32768`，xpu_gpt 图模式在 decode verification 路径崩溃。
- **失败日志**: `outputs/logs/vlm_0601_170605_24140_xg_graph_bs128.log`。
- **报错信息**:

```text
File "/data01/zcs/maas_engine_vlm2b5/3rdparty/xpu_gpt/xpu_gpt/adapter/xllm_adapter/spec_rl.py", line 256, in async_step
  current_sample_result = self.verification(gen_ctx, draft_tokens)
File "/data01/zcs/maas_engine_vlm2b5/3rdparty/xpu_gpt/xpu_gpt/adapter/xllm_adapter/xpu_spec_inferencer.py", line 736, in verification
  logits, target_hidden_states = self.target_model.model_inference(fwd_args, is_context=False)
File "/data01/zcs/maas_engine_vlm2b5/3rdparty/xpu_gpt/xpu_gpt/core/pipeline/pipeline.py", line 361, in model_inference
  results = self.decode_graph(mojo_input_args, mojo_input_args.is_context)
File "/usr/local/lib/python3.11/site-packages/torch_npu/dynamo/torchair/inference/_cache_compiler.py", line 286, in compiled_fn
  return ge_kernel(*full_args)
File "<string>", line 93, in kernel
AssertionError: expected size 64==64, stride 65==25 at dim=0; expected size 65==25, stride 1==1 at dim=1
This error most often comes from a incorrect fake (aka meta) kernel for a custom op.
Use torch.library.opcheck to test your custom op.
```

## 2. 根本原因分析 (Root Cause)

日志证据指向 **xpu_gpt decode graph cache 没有按 max sequence length 隔离**，导致 32k 运行复用了旧 12k 长度口径下编译出的 GE/OM 图元数据。

关键证据：

- 当前运行日志有 `force rollout max sequence length envs to 32768 from config.max_sequence_length`，说明 `data.max_length` 已传入 rollout 侧长度环境。
- 临时诊断确认进入 `decode_graph` 前，请求侧 `mojo_input_args.npu.kv_idx` 已经是 `shape=(64, 65), stride=(65, 1)`，不是实时 `ForwardArgs.kv_idx` 仍停留在 `(64, 25)`。
- `torch_npu/dynamo/torchair/inference/_cache_compiler.py` 会组装 `full_args = parameters + args`，断言中的旧 `(64, 25)` 可以来自缓存编译图参数或旧图元数据，而不一定来自本次请求参数。
- 远端 `compile_cache` 中存在 `MojoXllmEagleTargetModel_static_gecache_.../forward/compiled_module/graph_1_20260521214304.om` 等旧文件，时间明显早于本次 32k 验证。
- `32768` 长度在 512 block 粒度加 spec margin 后对应 65 列 block；旧较短长度口径对应 25 列 block。失败签名里的 `65==25` 正好是这两个长度桶的静态图列数冲突。

注意：报错尾部提示 custom op fake kernel 只是 torchair 的泛化提示。本次排查没有把问题归因为 CANN/custom op bug；结合输入 shape、cache 时间戳和 fresh cache 验证，问题落在 stale graph cache 复用。

## 3. 解决方案 (Solution)

### 临时方案 (Workaround)

- 改变 `data.max_length` / `rollout.max_sequence_length` 后，为 xpu_gpt 图模式使用新的 `XPU_GRAPH_CACHE_DIR`，例如 `compile_cache/maxlen_32768`。
- 或者只清理目标模型对应的旧 graph cache。不要在有任务运行时全局删除 `compile_cache`，避免破坏其他正在运行或复用缓存的实验。

### 终极方案 (Fix)

在 rollout env 构造处对 xpu_gpt 图缓存目录做长度隔离：

- 文件: `maas_finetune/trainer/rl/rollout/xllm_backend/engine.py`
- 条件: `XLLM_MODEL_CLS == "xpu"` 且存在 `XPU_GRAPH_CACHE_DIR`
- 行为: 展开 `${PROJECT_ROOT}` / 环境变量后，将 graph cache dir 改为 `.../maxlen_{max_sequence_length}`，例如 `/data01/zcs/maas_engine_vlm2b5/compile_cache/maxlen_32768`

该修复只作用于 xpu_gpt 后端，不污染 torch infer 路径。

补充防护：`3rdparty/xpu_gpt/xpu_gpt/core/pipeline/pipeline.py` 中保留 decode compile 输入归一化逻辑，在静态图期望列数更大而本次 `kv_idx` 较短时进行 padding，避免短输入触发同类静态列数不匹配。

## 4. 验证结果 (Verification)

修复验证日志：`outputs/logs/vlm_0601_172008_24140_xg_graph_bs128.log`。

关键日志：

```text
set xpu graph cache dir to /data01/zcs/maas_engine_vlm2b5/compile_cache/maxlen_32768
Find graph cache dir setting: /data01/zcs/maas_engine_vlm2b5/compile_cache/maxlen_32768, will start to use graph cache
```

验证结果：

- 跨过旧失败点，未再出现 `AssertionError: expected size 64==64, stride 65==25`。
- `global_step=1`: `reward.mean=0.162109`，`rollout/logprobs_diff=0.0405777`，`rollout/logprobs_corr=0.892586`。
- `global_step=2`: `reward.mean=0.155273`，`rollout/logprobs_diff=0.0288583`，`rollout/logprobs_corr=0.920895`。
- `global_step=3`: `reward.mean=0.132812`，`rollout/logprobs_diff=0.0272053`，`rollout/logprobs_corr=0.922598`。
- 错误扫描未发现新的 `Traceback` / `AssertionError` / `RuntimeError` / `Raylet is terminated`。

结论边界：这证明 `12k -> 32k` 后的图模式断言问题已被当前修复跨过；它不是 70+ step / 200 step 长线精度漂移的最终通过结论。

## 5. 沉淀与关联

- 变更 `data.max_length`、`rollout.max_sequence_length`、`XLLM_MODEL_MAX_LENGTH` 或 `XPERF_MAX_SEQUENCE_LENGTH` 时，如果开启 `XPU_DECODE_COMPILE=1`，必须同步隔离或失效化 `XPU_GRAPH_CACHE_DIR`。
- 遇到 `65==25`、`stride 65==25` 这类静态图 shape/stride 断言，先核对本次输入 shape、compiled graph cache 时间戳、cache key 是否包含长度参数。
- 不要仅凭 torchair 报错尾部的 custom op 提示归因到底层算子；必须先用张量 shape、图缓存和 fresh cache 验证排除缓存复用问题。
- 关联知识点: [[wiki/05_fact/torch_infer_merge_debug_checklist|xpu_gpt + torch infer 合并验证排查清单]]、[[wiki/02_precision/rl_precision_metrics_guide|RL 核心监控指标解析]]
