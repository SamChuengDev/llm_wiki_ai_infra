---
title: DualPipe 重叠策略与显存峰值问题
tags:
  - DualPipe
  - pipeline-parallelism
  - overlap
  - memory-peak
last_updated: 2026-11-17
source: ["migrated from ascend-llm-wiki"]
---

# DualPipe 重叠策略与显存峰值问题

## 1. DualPipe 概述

DualPipe 是一种流水线并行（pipeline parallelism）中的计算和通信重叠策略，通过将前向（forward）和反向（backward）计算与通信操作进行调度重叠，从而隐藏 all-reduce 等通信延迟，提升整体训练吞吐。

## 2. 重叠机制

在流水线并行中，每个 micro-batch 的计算完成后需要发起梯度 all-reduce 通信。DualPipe 的核心思路是将当前 micro-batch 的通信操作与后续 micro-batch 的计算操作在时间上重叠执行：

- 前向计算完成后，立即发起该 micro-batch 的 all-reduce 通信
- 在通信完成之前，继续执行下一个 micro-batch 的前向或反向计算
- 通过精细的调度编排，使通信延迟被计算时间完全隐藏

## 3. 显存峰值问题

在重叠场景下，多个量化算子的输出 tensor 可能同时存活，导致瞬态显存（transient memory）累积，形成显存峰值。

### 3.1 问题描述

如文档第 6.5 节所述，该问题表现为 **2079 MB 的瞬态实例内存开销**。具体而言：

- 在重叠区间内，多个 micro-batch 的量化计算同时进行
- 每个量化算子的输出 tensor 在通信完成前无法释放
- 这些临时 tensor 在重叠窗口内同时存活，造成显存峰值显著上升

### 3.2 根源分析

问题的根本原因是 `npu_dynamic_mx_quant` 算子每次调用都分配新的输出 tensor。在 DualPipe 的重叠调度下：

1. 多个 micro-batch 的量化操作在时间上重叠
2. 每个调用都独立分配输出缓冲区
3. 这些缓冲区在重叠区间内同时存在，无法复用
4. 导致瞬态显存随重叠深度线性增长

## 4. 优化方案

通过引入 **预分配输出缓冲区**（out 版本）来复用显存：

- 将 `npu_dynamic_mx_quant` 替换为支持预分配输出 tensor 的版本（如 `npu_dynamic_mx_quant_out`）
- 在重叠区间开始前预先分配固定大小的输出缓冲区
- 多个 micro-batch 共享同一块显存，避免重复分配
- 将临时 tensor 的存活时间错开，消除重叠区间内的显存累积

该优化显著降低了 DualPipe 场景下的显存峰值，消除了 2079 MB 的瞬态开销。

## 5. 与 FP8 量化的关系

[[concepts/fp8-training]] 和 [[concepts/mx-quantization]] 中的量化算子是 DualPipe 场景下最主要的显存消耗者。FP8 混合精度训练中广泛使用 MX quant 进行张量量化，这些量化操作在 DualPipe 重叠调度下产生的临时 tensor 累积是显存峰值问题的核心来源。因此，DualPipe 的显存优化与 FP8 量化算子的实现密切相关。