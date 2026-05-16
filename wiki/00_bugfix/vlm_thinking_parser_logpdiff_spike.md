---
title: "Bugfix: VLM RL 训练中 thinking_parser 缺失导致 logpdiff 极端 Spike (0.3+)"
tags: [status/resolved, priority/high, category/alignment]
last_updated: 2026-05-16
source: [outputs/logs/vlm_0515_231030_229_vit_cache_bs2.log, outputs/logs/vlm_0516_113330_229_nothink_data_replay_bs2.log]
confidence: high
status: closed
open_questions: []
contradictions: []
---

# Bugfix Report: VLM RL 训练中 thinking_parser 配置缺失导致极短回复 logprobs_diff 爆炸

## 1. 现象描述 (Symptoms)
- **环境**: NPU 910B, W8A8 量化推理, FSDP (ZeRO-3) Actor, xLLM Rollout
- **报错信息/现象**:
  在 VLM RL 训练长跑（130+步）中，出现极端的 `logprobs_diff` spike。
  正常步的 diff 为 `0.014~0.065`，但在 Step 98 突然飙升至 **`0.3001`**，PPL 从 `1.07` 跳升至 `1.445`，correlation 暴跌至 `0.304`。
- **关联数据污染报错**:
  尝试用 `trajectory.jsonl` 回放此 batch 时，触发 DataLoader 丢弃异常：
  ```text
  RuntimeError: Discard threshold triggered! The discard ratio in recent batch has reached 100.0% (2/2)
  grpc_message:"InvalidArgumentError('Invalid base64 image url')"
  ```

## 2. 根本原因分析 (Root Cause)

**1. Logprobs_diff Spike 根因 (thinking_parser 缺失)**:
- 系统 `run.sh` 中未显式注入 `thinking_parser`，且模型路径不含 `qwen` 字样，导致 `XllmEngineConfig` 默认将 `thinking_parser` 彻底禁用（`enabled=False`）。
- xLLM 侧未开启 `CustomModeReasoningParser`，不会硬性注入 `<think>` prefill。模型仅受到系统 Prompt 的“软约束”。
- 遇到特定短文本任务（如要求描述简单图表），模型认为不需要复杂推理，**直接输出不到 20 个 token 的极短最终答案**（无 `<think>` 推理过程）。
- 由于 Rollout 是自回归前向计算，而 Actor 是 Teacher-forcing 整句计算，在极短文本下，两者对特定 Token 概率的排序由于量化噪声产生了严重扰动（correlation=0.30）。同时，因为总 Token 数极少，误差无法被长序列（几万 Token）稀释，导致最终 masked_mean 的 diff 飙升至 0.3。

**2. Trajectory 重放 Base64 丢失根因**:
- RL 引擎为了节省磁盘空间，在序列化写入 `trajectory.jsonl` 时，会拦截并替换掉真实的 base64 图片字符串为 `{"url": "..hidden.."}`。直接用 `jsonl` 投喂 PyDantic DataLoader 会直接触发 InvalidArgument 解析错误，导致数据 100% 被 discard。

## 3. 解决方案 (Solution)

### 终极方案 (Fix)
- **强制注入 Thinking Parser**: 在 `run.sh` 脚本中显式追加并覆盖 Hydra 的 parser 配置，强迫 Rollout 在生成时硬填充 `<think>`：
  ```bash
  ++rollout.thinking_parser.enabled=True \
  ++rollout.thinking_parser.prefix=\"'<think>'\" \
  ++rollout.thinking_parser.suffix=\"'</think>'\" \
  ++rollout.thinking_parser.default_mode=enabled \
  ++rollout.thinking_parser.modes.enable.prefill_content=\"'<think>'\" \
  ```
- **开发原始数据提取脚本**: 编写 `scripts/extract_replay_data_from_parquet.py`。不直接使用被污染的 jsonl，而是以 jsonl 的 user prompt 文本作索引，去源头（`MathVerse` 的 Parquet）原汁原味地提取带合法 base64 的原始行，并输出新的 replay `.parquet` 文件。

## 4. 验证结果 (Verification)
- 使用新的 `step98_no_think_replay.parquet` 和带有 Parser 配置的 `run.sh` 重跑训练。
- 注入的 `[Thinking-Probe]` 日志明确观察到 `CustomModeReasoningParser` 被触发，模型老老实实生成了 `Reasoning starts with: "Got it, let's see..."`。
- `logprobs_diff` 从曾经的 **`0.3001`** 被成功压回 **`0.0480`**，PPL 回归健康值 (`rollout_ppl: 1.1036`, `actor_ppl: 1.1248`)。

## 5. 沉淀与关联
- 关联知识点: [[wiki/01_alignment/logpdiff_drift|Logprobs Diff 漂移与对齐误差]]
