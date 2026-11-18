---
title: CANN 昇腾计算语言包详解
tags: [CANN, AscendCL, aclnn, ATB, GE]
last_updated: 2026-11-17
source: ["migrated from ascend-llm-wiki"]
---

# CANN 昇腾计算语言包详解

## 1. CANN 概述

CANN（Compute Architecture for Neural Networks），即华为昇腾计算语言包，是整个昇腾 AI 软件栈的核心中间层。它向下屏蔽底层硬件差异，向上提供统一的算子开发与调用接口，是连接昇腾硬件（Ascend NPU）与上层 AI 框架（PyTorch、TensorFlow 等）的桥梁。

## 2. 核心组件

### AscendCL (ACL)

底层计算接口，提供设备管理、上下文管理、内存管理、Stream/Event 管理等基础能力。适合简单算子的直接调用。AscendCL 是 CANN 中最基础的编程接口层，所有上层算子库最终都依赖 AscendCL 完成硬件交互。

### aclnn

高级 NN 接口，基于 AscendCL 封装，提供更便捷的算子调用方式。OpPlugin 中 `opapi/` 目录的算子主要调用此接口。aclnn 接口封装了算子执行的全流程（准备输入 → 执行 → 获取输出），开发者无需手动管理 Stream 和任务队列，大幅降低了开发复杂度。

### ATB (Ascend Transformer Boost)

面向 Transformer 模型的高性能融合加速库，提供 FlashAttention、PagedAttention、Rotary Position Embedding (RoPE)、Silu 激活函数等融合算子。ATB 将多个连续算子融合为单一 kernel，减少显存带宽消耗和 kernel launch 开销，特别适合 LLM 推理和训练场景。更多 ATB 算子详情请参见 [[components/atb]]。

### BlasLt

矩阵乘加速库，提供高性能 GEMM (General Matrix Multiply) 运算。BlasLt 针对昇腾硬件特性深度优化，支持混合精度矩阵乘（FP16、BF16、FP8 等），是 LLM 中线性层和注意力计算的核心加速引擎。

### GE 图引擎

图引擎/运行时，负责算子调度、资源管理和执行。GE 将算子构建为计算图，进行图级优化（算子融合、内存复用、数据排布优化等），生成高效的执行计划，并在 NPU 上调度执行。

### AscendC

算子编程语言，用于编写自定义 NPU kernel。AscendC 提供类 C++ 的编程模型，支持高效的数据搬运、向量计算和同步机制，开发者可用其实现标准算子库中未覆盖的自定义算子。

## 3. 各接口的适用场景与性能特点

| 接口 | 适用场景 | 性能特点 |
|------|---------|---------|
| AscendCL | 底层设备控制、简单算子调用 | 最接近硬件，灵活性最高，但开发成本大 |
| aclnn | 高级算子调用，OpPlugin 开发 | 封装性好，易用性高，性能接近 AscendCL 直接调用 |
| ATB | Transformer 模型融合算子 | 极致 kernel 融合，减少显存带宽消耗，适合长序列场景 |
| BlasLt | 矩阵乘运算 | 深度硬件优化，混合精度支持，GEMM 性能最优 |
| GE 图引擎 | 图级优化、算子调度 | 自动图优化，多算子融合，适合复杂模型 |
| AscendC | 自定义算子开发 | 灵活编写自定义 kernel，需深入硬件知识 |

## 4. 版本说明

CANN 9.2.0 是目前推荐版本，包含最新的 FP8 和 MX (Microscaling) 量化算子支持。该版本对 FlashAttention 的 FP8 实现做了显著优化，同时新增了对 FP8 矩阵乘和混合精度训练的支持，是昇腾大规模 LLM 训练和推理的首选 CANN 版本。

## 5. 与 OpPlugin 的关系

[[architecture/op-plugin-architecture]] 位于 CANN 之上，作为 PyTorch 与 CANN 之间的适配层。OpPlugin 将 PyTorch ATen 算子请求转换为 CANN API 调用（主要使用 aclnn 和 ATB 接口），从而在昇腾 NPU 上执行计算。简言之，OpPlugin 是 PyTorch 框架接入 CANN 软件栈的关键桥梁。