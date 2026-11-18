---
title: TorchNPU 架构详解
tags:
  - TorchNPU
  - PyTorch
  - ATen
  - NPU-backend
last_updated: 2026-11-17
source: ["migrated from ascend-llm-wiki"]
---

# TorchNPU 架构详解

## 1. TorchNPU 概述

TorchNPU 是华为昇腾为 PyTorch 框架提供的 NPU 后端适配层，通过扩展 PyTorch 的设备抽象接口，使 PyTorch 能够在昇腾 NPU（Neural Processing Unit）上高效执行深度学习计算任务。其核心设计思路是复用 PyTorch 的前端图构建与自动求导机制，仅在后端算子分发与设备管理层面进行替换，从而实现与原生 PyTorch 生态的高度兼容。

## 2. 核心功能

### ATen 后端分发

TorchNPU 通过注册自定义的 `BackendSelect` 与 `Autograd` 内核，将 PyTorch 的 ATen（A Tensor Library）算子调用分发到昇腾 NPU 上执行。具体而言，它实现了 `at::npu` 命名空间下的算子体系，覆盖了 `torch.nn.functional` 中的绝大多数计算原语，并利用 PyTorch 的 **DispatchKey** 机制（`DispatchKey::NPU`、`DispatchKey::AutogradNPU`）完成运行时派发。

### 设备管理

提供 `torch.npu` 模块下的设备管理接口，主要包括：

- **设备创建与销毁**：通过 `torch.npu.device_ctx` 上下文管理器及 `torch.npu.set_device` 控制当前默认设备。
- **设备查询**：`torch.npu.device_count()`、`torch.npu.current_device()`、`torch.npu.get_device_name()` 等。
- **设备属性**：获取显存总量、剩余显存、计算能力等硬件信息。

### Stream / Event 管理

实现对 NPU 上异步执行流的支持，与 CUDA 的 Stream/Event 模型对齐：

- **Stream**：`torch.npu.Stream` 类，支持创建命名流、设置优先级、查询完成状态。
- **Event**：`torch.npu.Event` 类，用于记录耗时、同步流之间的执行顺序。
- 通过 `torch.npu.synchronize()` 实现全局同步。

### Tensor 格式

TorchNPU 支持昇腾 NPU 特有的数据存储格式，以适配达芬奇（DaVinci）架构的矩阵计算单元：

| 格式 | 说明 |
|------|------|
| **ND** | 标准 N 维连续排布，与 PyTorch 默认 `contiguous` 格式一致 |
| **NZ** | 昇腾特有的分块格式（`NCHW` 中 `C` 维度按 `16` 对齐分块），适用于 `Conv`、`MatMul` 等算子 |
| **5D** | `NC1HWC0` 格式，`C1 = ceil(C / C0)`，`C0 = 16`（FP16）或 `32`（INT8），是昇腾内部最常用的物理格式 |
| **Fractal** | 适用于矩阵乘法的分形格式，如 `FRACTAL_NZ` |

TorchNPU 在算子执行前会根据算子属性自动进行格式转换（`format_convert`），或通过 `torch.npu.set_format` 手动控制张量的内存排布。

### 内存管理

TorchNPU 维护了一套高效的 NPU 显存管理机制：

- **分配与释放**：通过 `c10::npu::NPUCachingAllocator` 实现，复用 `c10::backend::CachingAllocator` 的缓存分配器架构。
- **缓存策略**：按块大小分级缓存已释放的显存块，减少频繁向驱动申请/归还显存的开销。
- **内存池**：维护 `small_pool` 与 `large_pool` 两级池，分别应对小张量与大张量的分配需求。
- **显存统计**：提供 `torch.npu.memory_stats()`、`torch.npu.memory_summary()`、`torch.npu.max_memory_allocated()` 等接口用于监控。

## 3. 版本信息

| 版本类型 | 版本号 | 环境 |
|----------|--------|------|
| **dev 版** | `2.9.0.post3.dev20260522` | `s00574452` conda 环境 |
| **release 版** | `2.9.0.post3` | `dsv4_llm` 环境 |

两个版本均基于 PyTorch 2.x 的 Python API 和 C++ 后端接口，release 版为稳定发布版本，dev 版包含最新的实验性功能与修复。

## 4. 与 OpPlugin 的关系

[[architecture/op-plugin-architecture]] 作为 TorchNPU 的算子插件模块，负责提供 TorchNPU 原生未覆盖的扩展算子实现。二者的分工如下：

- **TorchNPU**：提供标准 PyTorch 算子的 NPU 后端适配（如 `torch.add`、`torch.nn.Conv2d` 等），以及设备管理、内存管理等基础设施。
- **OpPlugin**：作为独立动态库（`torch_npu_plugin.so`）加载，注册自定义算子及自定义自动求导函数，用于补充 TorchNPU 尚未合入或需要迭代速度更快的算子。

在实际部署中，OpPlugin 与 TorchNPU 协同工作：TorchNPU 处理基础算子，OpPlugin 提供扩展算子，二者共享同一个 NPU 资源上下文。

## 5. 安装方式

### pip 安装

从华为昇腾官方 PyPI 源安装 release 版本：

```bash
pip install torch-npu -f https://www.hiascend.com/developer/ascendpytorch
```

### 源码编译

从源码编译 dev 版本，用于开发与调试：

```bash
git clone https://gitee.com/ascend/pytorch.git
cd pytorch
# 配置编译选项
python setup.py build
python setup.py install
```

源码编译需要预先安装昇腾 CANN 工具包及对应的 PyTorch 源码版本，确保 `ASCEND_HOME` 环境变量正确指向 CANN 安装路径。