---
title: OpPlugin 算子适配插件层详解
tags:
  - OpPlugin
  - operator-adapter
  - NPU
  - aclops
  - opapi
  - ATB
last_updated: 2026-11-17
source: ["migrated from ascend-llm-wiki"]
---

# OpPlugin 算子适配插件层详解

## 1. OpPlugin 概述

昇腾 **OpPlugin** 是 TorchNPU 的配套算子适配仓库，代码位于 `D:\codes\dj-oom\op-plugin`。它将 PyTorch ATen 算子映射到昇腾 NPU 的底层算子实现，是衔接深度学习框架与昇腾硬件计算能力的关键桥梁。通过 OpPlugin，用户可以在 PyTorch 生态中无缝调用昇腾 NPU 的加速能力，无需感知底层硬件细节。

## 2. 核心目录结构

OpPlugin 按算子实现方式划分为以下几个核心目录（各目录的详细实现请参见 [[components/op-plugin-ops]]）：

| 目录 | 算子数量 | 接口类型 | 说明 |
|------|----------|----------|------|
| `ops/aclops/` | 200+ 基础算子 | AscendCL | 包含 Abs、Add、Bmm、Conv2d、Softmax 等基础算子，直接基于 AscendCL 接口封装，性能开销低 |
| `ops/opapi/` | 400+ 高级算子 | aclnn | 这是算子数量最多的实现方式，涵盖量化、Attention、MoE、RMSNorm 等复杂算子 |
| `ops/atb/` | 融合算子 | ATB | ATB 加速库融合算子，面向 LLM 推理场景，如 FlashAttention、PagedAttention、Rope |
| `ops/dvm/` | 动态虚拟内存 | — | 延迟融合（Lazy Fusion）支持，用于优化内存管理与算子融合调度 |

### 2.1 `ops/aclops/` — 基础算子层

基于昇腾 AscendCL（Ascend Computing Language）底层接口实现，提供最基础的算子能力。每个算子对应一个独立的文件，结构清晰，便于定位和调试。典型算子包括：

- 张量操作：`Abs`、`Add`、`Sub`、`Mul`、`Div`
- 神经网络：`Conv2d`、`Softmax`、`Relu`、`BatchNorm`
- 矩阵运算：`Bmm`、`Linear`、`Matmul`

### 2.2 `ops/opapi/` — 高级算子层

基于 aclnn 接口实现，是 OpPlugin 中规模最大的算子集合，覆盖了从基础到高级的各类算子。该层封装了更复杂的计算逻辑，支持：

- **量化算子**：FP8 量化、MX 量化等
- **注意力机制**：FlashAttention 系列、MLA 等
- **混合专家模型（MoE）**：MoE 系列算子
- **归一化**：RMSNorm、LayerNorm 等
- **通信融合**：AllGatherMatmul、ReduceScatterMatmul 等

### 2.3 `ops/atb/` — ATB 加速库融合算子

面向 LLM 推理场景，基于 ATB（Ascend Transformer Boost）加速库实现的高性能融合算子。这些算子将多个连续计算步骤融合为一个内核执行，显著减少访存开销和 kernel launch 延迟：

- **FlashAttention**：高效注意力计算
- **PagedAttention**：支持 Paged KV Cache 的注意力机制
- **Rope**：旋转位置编码

### 2.4 `ops/dvm/` — 动态虚拟内存

提供延迟融合（Lazy Fusion）支持，通过动态虚拟内存管理机制，允许算子在执行阶段再进行融合调度，提升算子执行效率。

## 3. 关键算子类别

### 3.1 FP8 量化算子

FP8 量化是 [[concepts/fp8-training]] 的核心技术路径，OpPlugin 中提供了丰富的量化算子支持：

- `npu_dynamic_mx_quant` — 动态 MX 量化
- `npu_dynamic_mx_quant_col_only` — 仅列维度动态 MX 量化
- `npu_dynamic_mx_quant_row_only` — 仅行维度动态 MX 量化
- `npu_dynamic_mx_quant_with_dual_axis` — 双轴动态 MX 量化

这些算子基于 [[concepts/mx-quantization]] 技术，能够根据张量统计数据动态调整量化参数，在保持模型精度的前提下显著降低内存占用和计算开销。

### 3.2 大模型训练算子

针对大语言模型训练场景，OpPlugin 提供了专用的高性能算子：

- **FlashAttention**：高效注意力机制，降低显存占用
- **MoE 系列**：混合专家模型相关算子，支持稀疏激活
- **MLA Prolog**：Multi-head Latent Attention 前置处理
- **RMSNorm / LayerNorm**：归一化算子，支持多种精度

### 3.3 通信融合算子

面向分布式训练场景，将通信与计算融合以减少通信开销：

- **AllGatherMatmul**：AllGather 通信与矩阵乘融合
- **ReduceScatterMatmul**：ReduceScatter 通信与矩阵乘融合
- **AlltoAllMatmul**：AlltoAll 通信与矩阵乘融合

## 4. 代码生成器 `torchnpugen/`

OpPlugin 提供了强大的代码生成能力，通过 `torchnpugen/` 目录下的代码生成器，可以根据 YAML 配置自动生成算子适配代码：

- 从 YAML 配置读取算子定义
- 自动生成算子的注册代码、forward/backward 模板
- 减少重复性编码工作，提升开发效率

## 5. 配置管理 `op_plugin/config/`

OpPlugin 的算子行为通过多个 YAML 配置文件进行管理：

| 配置文件 | 作用 |
|----------|------|
| `op_plugin_functions.yaml` | 定义算子函数列表及注册规则 |
| `derivatives.yaml` | 定义算子的反向传播（导数）规则 |
| `deprecated.yaml` | 标记已废弃的算子及其迁移路径 |

通过这些配置文件，可以实现算子行为的热更新，无需修改核心代码即可调整算子注册策略。

## 6. 构建系统

OpPlugin 使用 CMake 构建系统，编译输出为 `op_plugin_atb` 共享库，链接 TorchNPU 和 ATB 库。构建流程大致如下：

1. CMake 解析构建配置，确定目标平台和依赖
2. 编译 `ops/aclops/`、`ops/opapi/`、`ops/atb/` 等各层源码
3. 链接 `torch_npu`（TorchNPU 核心库）和 `libatb.so`（ATB 加速库）
4. 生成最终的 `op_plugin_atb` 共享库

## 7. 与本项目 FP8 优化的关联

OpPlugin 在 [[concepts/fp8-training]] 的 Path B 实现中扮演着核心角色。具体而言：

- **量化算子**是实现 Path B FP8 训练的关键组件，负责在前向传播和反向传播中对张量进行 FP8 量化
- **优化目标**：针对 `col-only` 和 `row-only` 模式下的量化算子，添加预分配输出缓冲区机制，减少动态内存分配带来的性能开销
- 通过预分配输出缓冲区，可以显著降低算子执行过程中的内存分配延迟，提升 FP8 训练的整体吞吐量

更多关于昇腾 CANN 软件栈的整体架构，请参考 [[architecture/cann-overview]]。