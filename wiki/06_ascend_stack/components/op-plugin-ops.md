---
title: OpPlugin 算子适配实现详解
tags:
  - OpPlugin
  - operator
  - adapter
  - aclops
  - opapi
  - ATB
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# OpPlugin 算子适配实现详解

## 1. 算子适配层概览

OpPlugin 是 TorchNPU 的核心组件之一，负责将 PyTorch ATen 算子映射到昇腾 NPU 的底层算子。适配代码位于 `op_plugin/ops/` 目录下，按照调用接口的不同分为三种实现方式。

关于 OpPlugin 的整体架构，请参见 [[architecture/op-plugin-architecture]]。

## 2. 三种实现方式对比

| 方式 | 目录 | 调用接口 | 算子数量 | 适用场景 |
|------|------|---------|---------|---------|
| **AscendCL** | `aclops/` | AscendCL API | 200+ | 基础算子（Abs, Add, Bmm, Conv2d） |
| **aclnn** | `opapi/` | aclnn API | 400+ | 复杂算子（量化、Attention、MoE） |
| **ATB** | `atb/` | ATB 加速库 | 20+ | 融合算子（FlashAttention, PagedAttention） |

### 2.1 AscendCL（aclops/）

AscendCL（Ascend Compute Language）是昇腾平台的基础计算 API 层。`aclops/` 目录下的算子直接调用 AscendCL API 完成计算，适用于 Abs、Add、Bmm、Conv2d 等基础算子。这类算子实现相对简单，直接映射到 NPU 硬件指令。

### 2.2 aclnn（opapi/）

aclnn 是昇腾提供的更高层算子库 API，封装了更复杂的计算逻辑。`opapi/` 目录包含 400+ 算子，涵盖量化、Attention、MoE 等复杂场景。aclnn 算子在内部可能组合多个 AscendCL 调用，提供更丰富的语义和更简洁的调用接口。

### 2.3 ATB（atb/）

ATB（Ascend Transformer Boost）是昇腾的 Transformer 加速库，专注于融合算子的优化。`atb/` 目录下包含 20+ 个融合算子，如 FlashAttention、PagedAttention 等，将多个计算步骤合并为单个 kernel 执行，减少显存访问和 kernel launch 开销。

## 3. DVM 机制

`dvm/` 目录实现动态虚拟内存（Dynamic Virtual Memory）延迟融合调度。DVM 的核心思想是将多个算子的内存分配请求延迟合并，减少显存碎片和分配开销。在 FP8 量化等需要大量临时缓冲区的场景中，DVM 机制可以有效降低显存峰值。

## 4. 代码生成器

`torchnpugen/` 目录下的代码生成器能够根据 YAML 配置文件自动生成算子适配代码。这种方式大幅减少了手写重复代码的工作量，确保算子接口的一致性和正确性。开发者只需在 YAML 中描述算子的输入输出和计算逻辑，生成器即可产出完整的 C++ 适配代码。

## 5. 关键 FP8 算子实现分析

关于 FP8 训练的背景知识，请参见 [[concepts/fp8-training]] 和 [[concepts/mx-quantization]]。

以下是 OpPlugin 中 FP8 量化相关的关键算子：

### 5.1 `npu_dynamic_mx_quant` — 通用 MX 量化

通用动态 MX 量化算子，通过 `axis` 参数指定量化轴。支持对指定维度进行 block 级别的共享 scale 量化，适用于灵活配置的量化场景。

### 5.2 `npu_dynamic_mx_quant_col_only` / `row_only` — 单轴专用量化

- **`npu_dynamic_mx_quant_col_only`**：仅对列进行量化，对应 Path B 的前向路径
- **`npu_dynamic_mx_quant_row_only`**：仅对行进行量化，对应 Path B 的反向路径

这两个算子相比通用版本省去了轴选择逻辑，性能更优，在 Path B 策略中分别用于前向和反向计算。

### 5.3 `npu_dynamic_mx_quant_with_dual_axis` — 双轴量化

同时对行和列进行量化，对应 Path A 策略。该算子包含 `out` 版本，允许用户预先分配输出缓冲区，避免算子内部动态分配带来的显存峰值。

## 6. 优化方向

- **预分配输出缓冲区版本**：为 `col_only` 和 `row_only` 算子添加预分配输出缓冲区版本，类似 `dual_axis` 已有的 `out` 版本，从而在 Path B 策略下也能避免运行时动态分配
- **算子融合**：将量化算子与相邻的矩阵乘算子融合，减少中间结果的显存驻留时间
- **DVM 调度优化**：进一步优化 DVM 机制的延迟融合策略，在更多场景下降低显存峰值