---
title: ATB (Ascend Transformer Boost) 加速库
tags:
  - ATB
  - Transformer
  - acceleration
  - LLM
  - inference
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# ATB (Ascend Transformer Boost) 加速库

## 1. ATB 概述

ATB（Ascend Transformer Boost）是华为昇腾针对 Transformer 架构模型设计的高性能加速库，提供了一系列经过深度优化的融合算子，专门用于加速大语言模型（LLM）的推理与训练。ATB 位于 [[components/ascendcl]] 和 [[components/aclnn]] 之上，是更上层的加速库。

## 2. 核心算子

- **FlashAttention (V1-V3)**：高效 Attention 计算，显着降低显存带宽占用
- **PagedAttention**：vLLM 风格的显存管理 Attention 算子
- **SelfAttention**：标准自注意力算子
- **Rope**：旋转位置编码（Rotary Position Embedding）
- **RMSNorm / LayerNorm**：归一化层优化算子

## 3. 适用场景

- LLM 推理加速
- 训练中的 Attention 计算
- 需要高性能融合算子的 Transformer 相关场景

## 4. 在 OpPlugin 中的使用

OpPlugin 中 `atb/` 目录下的代码编译为独立的 `op_plugin_atb` 共享库，运行时动态链接 `libatb.so`，实现 ATB 算子的集成与调用。

## 5. 性能优势

- 融合算子设计减少了多次 kernel launch 的开销
- 通过算子融合降低了显存带宽消耗
- 针对昇腾 NPU 硬件架构进行了深度调优

## 6. 与 CANN 的关系

ATB 位于 CANN 软件栈的高层，是 AscendCL / aclnn 之上的高层加速库，直接服务于 LLM 推理与训练场景。参见 [[architecture/cann-overview]]。