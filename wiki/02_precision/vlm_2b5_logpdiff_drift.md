---
title: "Precision: VLM 2.5B RL 训练中 logprobs_diff 长线漂移精度对齐"
tags: [status/resolved, category/diff]
last_updated: 2026-05-26
source: [3rdparty/work_summary/强化学习/maas_engine/xgn_summary/logpdiff_drift_debug_plan.md]
confidence: high
status: active
open_questions:
  - "无"
contradictions: []
---

# Precision Alignment Report: VLM 2.5B RL logprobs_diff 长线漂移对齐

## 1. 差异现象 (Deviation Symptoms)
- **现象**: 在 VLM 2.5B 量化（NPU W8A8）强化学习（RL）长跑训练中，随着 `global_step` 的推进，训推不一致指标 `logprobs_diff`（训练端 Actor BF16 与推理端 Rollout W8A8 绝对对数概率偏差掩码均值）出现缓慢长线爬升（从 Step 1 的 `0.02x` 缓慢爬升，在 70+ step 之后逼近或越过 `0.1+`，伴随 `actor_ppl` 升高，且在后期生成大面积的格式脱轨，如 no-think 或 `dekor/deklar` 等乱码前缀）。
- **数据类型**: FP16 / BF16 / W8A8 混合打分与多模态特征

## 2. 定位过程 (Debugging Process)
为了定位此漂移问题，进行了几轮控制变量的消融实验（利用 `lr=0` 冻结权重等对照）：
1. **实验组 1（动态标尺与 Amax 策略）**：
   - 历史上认为 W8A8 动态量化标尺 (Amax) 更新策略（`all_reduce(MAX)` 的广播效应）在 long run 下与 AdamW 权重更新交互导致量化分辨率受损或截断（Activation Clipping）。
   - **消融结果**：此假说已被后续的 124 step 长跑证伪。在保留 NPU W8A8 + m1 量化算法的路径下，仅通过切断下述两个问题，漂移即告消失。量化标尺不是长线漂移的主因，而是误差放大器。
2. **实验组 2（ViT 缓存穿透）**：
   - 16 Rank 并发环境下，微小的 1 ULP（最小浮点精度单元）尾数误差导致字节级强哈希缓存失效（Cache MISS）。
   - **消融结果**：缓存失效引发 Actor 端回退走 HuggingFace PyTorch 重新计算 ViT，而 C++ xLLM 引擎与 PyTorch 计算结果的本底浮点误差，在已经产生畸变的 W8A8 标尺下会被成倍放大，导致 diff 瞬间尖峰。后续通过“软哈希”（四舍五入保留4位小数）和缓存特征散射（scatter）机制将 HIT 焊死在 100%。
3. **实验组 3（核心破案：KV Cache / Prefix Cache 跨请求污染）**：
   - 在检查 prefix cache 逻辑时，发现一个配置覆盖漏洞：请求级的 `enable_prefix_cache=true` 会覆盖 `runtime_env` 中的 `XLLM_DRIVER_ENABLE_PREFIX_CACHE=0`，使推理侧虽然在环境设置上关闭了 prefix cache，却仍然在局部被动启用了 page reuse。
   - 这导致 xLLM Rollout 端跨请求复用了旧的 KV prefix 块。因 VLM 的 block hash 未做图像 embedding 与请求上下文的强力隔离，导致复用了污染过的 stale KV 块。
4. **实验组 4（次要主因/放大器：thinking_parser 格式脱轨）**：
   - 早期 `run.sh` 中未显式打开 `thinking_parser` 时，模型仅受 system prompt 软约束。
   - 在 RL 策略更新阶段，没有强制的 `<think>` prefill 注入，模型很容易滑向不输出 `<think>`（no-think）或生成 `dekor/deklar` 前缀。极短的不规则文本没有足够的 token 去稀释量化底噪，导致 batch 级别的 `logprobs_diff` 经常突刺（Spike）。

## 3. 修复方案 (Alignment Fix)
### 3.1 核心修复：切断 KV Cache 跨请求污染
更新 xLLM 引擎，使 `enable_prefix_cache` 的默认值优先服从环境变量，并确保主仓的配置不再错误地覆盖其为 `True`：
- **xLLM 子仓修改** (`8ad321e896`):
  将 `GenerateParamsBuilder` 中 `enable_prefix_cache` 的默认行为从强制 `True` 改为尊重环境变量 `XLLM_DRIVER_ENABLE_PREFIX_CACHE`。
- **主仓运行配置修改**：
  确保环境变量 `XLLM_DRIVER_ENABLE_PREFIX_CACHE=0` 和 `XPU_DISABLE_PAGE_REUSE=1` 正确下发，使得请求级 `enable_prefix_cache` 完全置为 `False`，在 driver 侧的 `cached_length` 恒为 0，彻底根除 page reuse staled KV 污染。

### 3.2 格式规约修复：强制开启 thinking_parser Prefill 约束
在 `run.sh` 中显式添加 `thinking_parser` 的硬性约束配置：
```bash
++rollout.thinking_parser.enabled=True \
++rollout.thinking_parser.prefix="'<think>'" \
++rollout.thinking_parser.suffix="'</think>'" \
++rollout.thinking_parser.default_mode=enabled \
++rollout.thinking_parser.modes.enable.system_prompt="''" \
++rollout.thinking_parser.modes.enable.prefill_content="'<think>'" \
++rollout.thinking_parser.modes.disable.prefill_content="'<think></think>'" \
```
以此在 xLLM 推理侧硬性注入前缀，并在返回轨迹时由 `CustomModeReasoningParser` 准确分离并抽取 `reasoning_content`，防止无 parser 时模型策略向乱码/无 `<think>` 序列漂移。

## 4. 对齐验证 (Validation)
在 2450 节点上启动的 124 步完整长跑测试中：
- 采样了超过 128,000 条轨迹样本，全量请求的 `cached_length` 均为 0，无任何 prefix cache 命中。
- 格式表现：`reasoning_content` 的为空比例下降至 0（100% 非空），残留 think 标签及乱码从 ~54% 下降至 0.1% 左右。
- 精度指标：对比修复前后的 124 步结果，`logprobs_diff` 稳健保持在极高精度量级，未复现 70+ step 之后爬升至 0.1+ 的历史模式。

| 指标 | 修复前 `0525_095157` | 修复后 `0525_220913` |
|---|---:|---:|
| `logprobs_diff` 均值 | 0.086108 | **0.023021** (2%量级以内，完美对齐) |
| step 120 `logprobs_diff` | 0.148812 | **0.023386** |
| `logprobs_corr` 范围 | 0.197 - 0.949 | **0.909 - 0.949** |
| `grad_norm` 范围 | 0.069 - 5.003 | **0.038 - 0.166** |

## 5. 关联知识
- [[wiki/00_bugfix/vlm_thinking_parser_logpdiff_spike|Bugfix: thinking_parser 缺失导致的 logpdiff Spike]]
- [[wiki/02_precision/rl_precision_metrics_guide|RL 核心监控指标解析]]
- [[wiki/02_precision/vlm_rl_actor_rope_logprobs_diff|VLM RL 长序列 RoPE 偏移对齐]]
