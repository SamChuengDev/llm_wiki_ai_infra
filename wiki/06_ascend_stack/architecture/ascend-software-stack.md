---
title: 昇腾全栈软件架构
tags:
  - ascend
  - architecture
  - software-stack
  - NPU
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# 昇腾全栈软件架构

## 概述

昇腾全栈软件体系涵盖了从顶层 PyTorch 训练脚本到底层 NPU 硬件的完整链路。用户编写的 AI 训练/推理脚本经过 AI 框架、算子适配、CANN 计算语言包、驱动等层层分发与转换，最终在昇腾 NPU 上高效执行。每一层各司其职，通过标准化的接口与协议进行通信，共同构成了一个开放、高性能的全栈计算平台。

## 分层架构

昇腾 AI 软件栈从上到下可分为以下层次：

```
┌─────────────────────────────────────┐
│          应用层                      │
│     PyTorch 训练 / 推理脚本           │
├─────────────────────────────────────┤
│      AI 框架层                       │
│  PyTorch (ATen) → TorchNPU 后端      │
│  (设备管理 / Stream管理 / ATen 后端分发)│
├─────────────────────────────────────┤
│      算子适配插件层                   │
│   OpPlugin                           │
│  (aclops / opapi / atb / dvm 适配)   │
├─────────────────────────────────────┤
│   CANN 昇腾计算语言包                 │
│  AscendCL / aclnn / ATB / GE 图引擎  │
├─────────────────────────────────────┤
│       驱动层                         │
│     NPU 驱动 & 固件                  │
├─────────────────────────────────────┤
│       硬件层                         │
│   昇腾 NPU（Da Vinci 架构）           │
└─────────────────────────────────────┘
```

## 各层职责说明

### 应用层

用户直接编写的 AI 训练或推理脚本，通常基于 PyTorch 框架。脚本中定义模型结构、数据加载、训练循环和推理逻辑，通过标准 PyTorch API 发起计算任务。

### AI 框架层

- **PyTorch (ATen)**: PyTorch 的原生张量计算库，提供丰富的算子定义和自动微分机制。当用户调用 `torch.add`、`torch.matmul` 等算子时，ATen 负责算子分发。
- **TorchNPU 后端**: 昇腾为 PyTorch 提供的设备后端插件，负责：
  - 设备管理（NPU 设备的初始化、查询、销毁）
  - Stream 管理（异步任务队列的创建与同步）
  - ATen 后端分发（将 ATen 算子请求分发到昇腾专属实现路径）

更多细节参见 [[architecture/torchnpu-architecture]]。

### 算子适配插件层

**OpPlugin** 是连接 PyTorch 框架与 CANN 计算语言包的桥梁。它针对不同的应用场景提供了多种适配路径：

- **aclops**: 基于 AscendCL 的算子封装，适用于标准算子的快速适配
- **opapi**: 基于 aclnn 算子 API 的封装，提供更贴近框架语义的接口
- **atb**: 基于 ATB (Ascend Transformer Boost) 的适配，专注于 Transformer 类模型的高性能执行
- **dvm**: 动态虚拟机的适配路径，支持动态 Shape 等灵活场景

详见 [[architecture/op-plugin-architecture]]。

### CANN 昇腾计算语言包

CANN（Compute Architecture for Neural Networks）是昇腾 AI 处理器的计算语言包，提供统一的编程接口：

- **AscendCL**: 昇腾计算语言（Ascend Computing Language），面向算子的底层编程接口，提供内存管理、模型加载、推理执行等基础能力
- **aclnn**: 昇腾算子 API 库，封装了常用算子的高性能实现，可直接被上层框架调用
- **ATB (Ascend Transformer Boost)**: 面向 Transformer 架构的高性能加速库，提供融合算子、KV Cache 管理等优化
- **GE (Graph Engine)**: 图引擎，负责计算图的编译、优化和执行调度，支持算子融合、内存复用等编译优化

更多信息参见 [[architecture/cann-overview]]。

### 驱动层

NPU 驱动与固件负责操作系统与 NPU 硬件之间的通信，包括：
- 设备枚举与初始化
- 任务提交与中断处理
- 显存管理
- 固件升级与健康管理

### 硬件层

昇腾 NPU 基于自研 Da Vinci 架构，提供强大的 AI 算力支持。Da Vinci 架构的核心包括：
- **AI Core**: 执行矩阵运算、向量运算和标量运算的计算单元
- **AI CPU**: 处理控制流和复杂逻辑的通用计算核心
- **存储系统**: 多级存储层次（L1 Buffer、L2 Buffer、Global Memory），优化数据存取效率

## 数据流

一条典型的训练数据流如下：

```
训练脚本 → torch.add(a, b)
         → ATen 算子分发
         → TorchNPU 后端 (NPUAdd)
         → OpPlugin 适配 (aclops 路径)
         → AscendCL API (aclrtLaunchKernel)
         → NPU 驱动 → 任务队列
         → NPU 硬件执行
         → 结果返回
```

1. 用户在训练脚本中调用 `torch.add(a, b)`，其中张量 `a`、`b` 位于 NPU 设备上
2. ATen 根据张量的设备类型，将算子请求分发到 TorchNPU 后端
3. TorchNPU 后端根据算子名称和参数，调用 OpPlugin 中对应的适配实现
4. OpPlugin 将 PyTorch 的 Tensor 格式转换为 CANN 数据结构，并调用 AscendCL 或 aclnn API
5. AscendCL 通过 IOCTL 将任务提交到 NPU 驱动
6. 驱动将任务放入 NPU 的硬件队列，由 NPU 执行单元完成计算
7. 执行完成后，结果通过中断或轮询机制返回给上层

## 关键特性

### 算子融合

通过 GE 图引擎和 ATB 加速库，昇腾软件栈支持将多个连续的小算子融合为一个大算子，减少 kernel launch 开销和显存带宽消耗。例如，将 `add` + `relu` + `matmul` 融合为单个高性能算子执行。

### 内存管理

TorchNPU 与 CANN 协同实现了高效的显存管理机制：
- **缓存分配器（Caching Allocator）**: 复用已释放的显存块，减少频繁的显存分配与释放开销
- **统一内存池**: 支持 PyTorch 的显存管理与 CANN 的显存管理共享同一内存池，避免显存碎片
- **P2P 通信**: 多卡场景下支持 Peer-to-Peer 显存访问，减少数据拷贝

### 数据格式

昇腾 NPU 支持 ND、NZ（Fractal）、5D 等多种数据排布格式，详见 [[concepts/ascend-data-types]]。

### 通信协同

在分布式训练场景中，昇腾软件栈提供以下通信能力：
- **HCCL (Huawei Collective Communication Library)**: 昇腾集合通信库，支持 AllReduce、AllGather 等常用通信原语
- **Overlap 计算与通信**: 通过 Stream 管理实现计算与通信的异步并发，掩盖通信延迟
- **拓扑感知**: 根据 NPU 的互联拓扑（如 HCCS、PCIe），自动选择最优的通信路径和算法