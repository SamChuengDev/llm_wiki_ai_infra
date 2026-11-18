---
title: Microscaling (MX) 量化机制详解
tags:
  - MX
  - quantization
  - micro-scaling
  - col-only
  - row-only
  - dual-axis
last_updated: 2026-11-17
source: ["migrated from ascend-llm-wiki"]
---

# Microscaling (MX) 量化机制详解

## 1. MX 量化概述

Microscaling (MX) 是一种基于 block 的共享 scale 量化方法，属于 OCP (Open Compute Project) 标准。与传统的 per-tensor 或 per-channel 量化不同，MX 在更细粒度的 block 级别上共享缩放因子，从而在量化精度和计算开销之间取得更好的平衡。

## 2. 量化粒度

MX 按 **block**（通常 32 个元素）共享一个 scale 因子。每个 block 内的所有元素共用同一个缩放因子，大幅减少了 scale 参数的存储开销，同时保持了对数据分布局部变化的适应性。这种粒度设计在量化精度和显存/计算开销之间取得了有效平衡。

## 3. 三种量化轴

MX 量化支持三种不同的轴配置，分别适用于不同的计算场景：

### col-only (axis=-1)

按列分组量化。在前向传播（forward）中使用，与 activation 的布局匹配。每列元素共享一个 scale 因子，能够有效捕捉 activation 在特征维度上的分布差异。

### row-only (axis=-2)

按行分组量化。在反向传播（backward）中使用，与 gradient 的布局匹配。每行元素共享一个 scale 因子，与 gradient 张量的存储布局一致。

### dual-axis (双轴)

同时按行和列量化，产生 **4 个 tensor**（quant_x, scale_x, quant_w, scale_w）。这种方案虽然精度最高，但显存开销也最大，在训练过程中需要同时维护四个张量，对显存压力较大。

## 4. Path B 的策略

Path B 采用了一种更高效的策略：**前向使用 col-only + 反向使用 row-only**，避免 dual-axis 的 4 个 tensor 同时存在的显存峰值。通过在正向和反向传播中使用不同的量化轴，Path B 在保证精度的同时显著降低了显存占用，是一种实用的工程优化方案。

Path B 与 [[concepts/fp8-training]] 的训练框架结合紧密，为大规模模型训练提供了高效的量化支持。

## 5. 新算子

为支持 Path B 的 col-only 和 row-only 量化策略，专门实现了两个新的 thin wrapper 算子：

- **`npu_dynamic_mx_quant_col_only`**：实现 col-only 量化，用于前向计算
- **`npu_dynamic_mx_quant_row_only`**：实现 row-only 量化，用于反向计算

这两个算子封装了底层的 MX 量化逻辑，对外提供简洁的调用接口，便于集成到 [[concepts/dualpipe]] 等训练流水线中。

## 6. 优化方向

为了进一步优化显存效率，计划为 col-only 和 row-only 量化创建 **预分配输出缓冲区的 C++ 算子版本（out 版本）**。通过复用显存缓冲区，减少 DualPipe 场景下的峰值显存累积，从而支持更大规模的模型训练或更大的 batch size。