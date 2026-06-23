---
title: "VLM 23B XG logprobs_diff spike replay 定位现场"
tags: [status/in_progress, category/precision, xg, vlm, logprobs_diff]
last_updated: 2026-06-23
source:
  - outputs/logs/vlm_0622_124100_replay_step83_xg_23b_replay_step83_ep128_kv64_tok6144_if128_b8_chunk6144_debug_bs64.log
  - outputs/logs/vlm_0622_160204_replay_step83_strict_xg_23b_replay_step83_strict_ep128_kv64_tok6144_if128_b8_chunk6144_debug_bs64.log
  - outputs/logs/vlm_0622_173707_guard_garbled_xg_23b_guard_ep128_kv64_tok6144_if128_b8_chunk6144_dyn32_safe_bs64.log
  - outputs/DEBUG/logprobs_diff_0623_000610_replay_step83_xpusampler_budget/logprobs_diff_step1_top8.pt
  - outputs/DEBUG/logprobs_diff_0623_030200_replay_step83_forced17mask/logprobs_diff_step1_top16.pt
confidence: medium
status: active
open_questions:
  - "token_id=17 rollout_logprob=0 的来源是否来自 xLLM thinking_budget forced end 注入，仍需 no-budget replay 验证"
  - "乱码轨迹是 spike 根因、放大器还是同源症状，仍需结合 rollout token/probe 日志确认"
  - "guard 过滤只能隔离坏样本，不能作为根本修复"
contradictions: []
---

# VLM 23B XG logprobs_diff spike replay 定位现场

## 1. 差异现象 (Deviation Symptoms)

- **任务口径**: 23B VLM RL，8 节点 XG/xpu-gpt W8A8 rollout，Actor BF16，`ep128 + kv64 + tok6144 + if128 + scheduler/model/context batch=8`。
- **长跑症状**: `outputs/logs/vlm_0619_215427...` 对应 130 step 曲线中，`train/rollout/logprobs_diff` 在 step62/83 一带出现尖刺，`train/rollout/logprobs_corr` 同步下探；reward 和 PPL 大体向好，但 response length 均值逐步下降。
- **replay 症状**:
  - `vlm_0622_124100_replay_step83...` step1: `train/rollout/logprobs_diff=0.0436147`，`logprobs_corr=0.980606`，`reward.mean=0.4375`，`response_length.mean=1888.71`。
  - `vlm_0622_160204_replay_step83_strict...` step1: `train/rollout/logprobs_diff=0.0436931`，`logprobs_corr=0.981068`，`reward.mean=0.425781`，`response_length.mean=1932.12`。
  - `vlm_0622_173707_guard_garbled...` guard run step1 仍高: `logprobs_diff=0.0625356`，step2 回到 `0.0288409`；这只能说明过滤可以隔离一部分坏轨迹，不能证明根因已修。
- **数据类型**: rollout 为 xpu-gpt W8A8，Actor/old logprobs 为训练图 BF16/torch_npu 路径。

## 2. 定位过程 (Debugging Process)

- **已做重放数据集**:
  - 从历史 step83 轨迹提取了 `garbled_step83_all_prompts_replay.parquet` 和 `garbled_step83_all_prompts_replay_with_source.parquet`，后者保留 `replay_source`，用于核对 row/source。
  - strict replay 确认同一 64 prompt、每 prompt 8 条 rollout，总 batch 512；full-batch dump 显示 `samples=512 unique_rows=64 row_count_dist={8: 64}`。
- **已加诊断**:
  - `MAAS_LOGPROBS_DIFF_DEBUG_DUMP_FULL_BATCH=1` dump 全 batch 的 `input_ids/loss_mask/old_logprobs/rollout_logprobs/logprobs_diff/choice/sample_id`。
  - `scripts/inspect_logprobs_diff_dump.py` 汇总 top sample、按 response length/finish_reason 分桶、token id 贡献、zero rollout logprob 贡献。
  - rollout 侧新增 `Rollout-TokenText-Probe`，记录长 CJK/乱码输出、token/logprob 数量、finish_reason、thinking budget、weight_step。
- **当前最强 token 级线索**:
  - `outputs/DEBUG/logprobs_diff_0623_000610_replay_step83_xpusampler_budget/logprobs_diff_step1_top8.pt`:
    - raw: `overall_diff=0.05379`，`corr=0.752793`。
    - `token_id=17` 出现 619 次，覆盖 512 个 sample，贡献 `diff_sum=7263.65`，其中 rollout logprob 为 0 的 214 个 token 贡献 `7248.31`。
    - 去掉 `token_id=17 && rollout_logprob==0` 后，`diff_mean` 从 `0.05379` 降到 `0.0460109`，corr 从 `0.752793` 回到 `0.96662`。
  - `outputs/DEBUG/logprobs_diff_0623_030200_replay_step83_forced17mask/logprobs_diff_step1_top16.pt`:
    - raw: `overall_diff=0.050745`，`corr=0.760122`。
    - `token_id=17` 出现 613 次，覆盖 512 个 sample，贡献 `diff_sum=6943.79`，其中 rollout logprob 为 0 的 203 个 token 贡献 `6929.19`。
    - 去掉 `token_id=17 && rollout_logprob==0` 后，`diff_mean` 从 `0.050745` 降到 `0.0432069`，corr 从 `0.760122` 回到 `0.975799`。
- **对照 caveat**:
  - `vlm_0622_160204_replay_step83_strict...` full dump 中 `token_id=17` 没有 zero rollout logprob，`overall_diff=0.0436931`，说明 token17-zero 不是所有 replay diff 的唯一来源。
  - strict replay 的 top sample 更像长输出/length 截断带来的整体 logprob 偏差，而不是明显乱码；不能把所有 spike 都归因于乱码文本。

## 3. 当前假说与临时隔离 (Alignment Fix / Hypothesis)

- **高疑假说 A: thinking forced end token 的 rollout logprob 注入**  
  `token_id=17` 很可能对应 thinking 结束 token。若 xLLM 在 thinking budget 或 reasoning parser 路径中强制插入/结束 thinking，并把该 token 的 rollout logprob 置 0，Actor 侧重新算 old logprob 时不会同步为 0，于是单 token diff 可达 10+，直接拉低 corr。
- **高疑假说 B: `_forced_thinking_end_positions` 的 budget fallback 过宽**  
  当前主工作区仍包含 `thinking_budget_int` / `forced_by_budget` fallback 逻辑：只要 `pos >= thinking_budget` 且 rollout logprob 为 0，就可能被标记为 forced thinking end。这个逻辑用于诊断/掩码时风险较高，下一轮需要用 no-budget replay 验证。
- **临时 guard 不是根本修复**:
  - `MAAS_ROLLOUT_DISCARD_GARBLED` / `[Rollout-Garbled]` 只能在 processor 前丢弃可疑乱码组，避免坏样本进入 reward/actor。
  - guard run 捕获到 `finish_reason=length output_tokens=4096 cjk_chars=595` 的异常长 CJK 样本，并执行 discard；但它不能解释 token17 zero rollout logprob 的来源，也不能修复 rollout/actor 概率定义不一致。

## 4. 验证状态 (Validation)

- **已验证**:
  - step83 replay 数据集可以复现高于健康早期量级的 diff（`0.043~0.054`），并能通过 full-batch dump 定位到 token 级贡献。
  - guard 可以降低坏样本进入 PPO 的概率；guard run step2 指标正常到 `0.0288409`，但这只是阶段证据。
  - token17-zero rollout logprob 会显著破坏 corr；去掉这些 token 后 corr 可从 `0.75~0.76` 回到 `0.96~0.98`。
- **未完成**:
  - no-budget 版本 `_forced_thinking_end_positions` 的 8 节点 replay 尚未成功跑出 step1。最近一次尝试暴露了验证环境问题：snapshot 目录只在 head 容器存在，worker 容器没有同路径；`ray_xg_128rank_23b.sh` 固定 `--working-dir=.`，runtime env 相对 `PYTHONPATH` 也可能把旧仓路径混入 actor 反序列化。
  - 还没有 60~90 step 或 200 step 长跑证据证明 spike 已消失。按本仓规则，早期低 diff 不能称为长线精度修复。

## 5. 下一步恢复指南

1. 先不要继续长跑 guard。guard 只适合风险隔离，不适合根因定位。
2. 重新做 no-budget replay 前，确保 8 个 worker 容器都有同一 snapshot 绝对路径，或直接修正主工作区后正常 Ray working_dir 打包；不要只在 head 上改代码。
3. no-budget 对照的最小变更:
   - `_forced_thinking_end_positions` 只接受显式 `thinking_end_token_id` 且 rollout logprob 为 0 的 token；
   - 删除 `thinking_budget_int` / `forced_by_budget` fallback；
   - preflight 仅在仍存在 `thinking_budget_int|forced_by_budget` 时拒绝 replay。
4. replay 成功后优先比较:
   - step1 `train/rollout/logprobs_diff` 是否从 `0.05` 量级回落；
   - full dump 中 `token_id=17 && rollout_logprob==0` 是否消失；
   - corr 是否保持 `0.98+`；
   - top samples 是否仍集中在 length=4096 长输出。

## 6. 关联知识

- [[wiki/00_bugfix/vlm_thinking_parser_logpdiff_spike|VLM RL 训练中 thinking_parser 缺失导致 logpdiff 极端 Spike]]
- [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
- [[wiki/02_precision/rl_precision_metrics_guide|RL 核心监控指标解析]]
