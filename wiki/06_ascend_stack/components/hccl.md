---
title: HCCL 昇腾集合通信库
tags:
  - HCCL
  - communication
  - all-reduce
  - NPU
  - distributed
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# HCCL 昇腾集合通信库

## 1. 概述

HCCL（Huawei Collective Communication Library）是华为昇腾平台的集合通信库，类似于 NVIDIA NCCL。它提供多卡多机之间的高效通信能力，是昇腾分布式训练的基础组件。在 [[concepts/dualpipe]] 等流水线并行策略中，HCCL 是实现计算与通信重叠的关键基础设施。

## 2. 核心功能

HCCL 支持以下集合通信操作：

- **all-reduce** — 多卡数据归约并广播
- **all-gather** — 多卡数据收集并广播
- **reduce-scatter** — 多卡数据归约并分散
- **all-to-all** — 多卡间全交换
- broadcast、reduce、scatter 等基础操作

## 3. 关键环境变量最佳实践

推荐设置以下环境变量以保障大规模训练稳定性：

| 环境变量 | 推荐值 | 说明 |
|---------|--------|------|
| `HCCL_CONNECT_TIMEOUT` | `7200` | 连接超时时间（秒），多机场景下建议设置较大值 |
| `HCCL_BUFFSIZE` | `300` | 通信缓冲区大小（MB），影响通信吞吐 |
| `HCCL_EXEC_TIMEOUT` | `7200` | 执行超时时间（秒），防止大模型训练时超时中断 |
| `HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO` | `0.75` | 多维度拆分比例，优化多卡通信拓扑 |

### 应避免设置的变量

以下变量在 Ascend NPU 环境中不应手动设置，否则可能导致通信异常或性能下降：

- `HCCL_SOCKET_IFNAME` — 由系统自动检测，手动设置可能导致网卡绑定错误
- `HCCL_INTRA_ROCE_ENABLE` — 内部 RoCE 通信开关，不应手动干预
- `HCCL_DISABLE_NHR` — 禁止关闭 NHR（Neighbor Health Ratio）检测

## 4. 常见错误

### `hcclCommInitRootInfoConfig` 失败（error code 4, RootInfoDetect failed）

该错误通常由硬件异常事件引起，例如 NPU 卡出现异常后通信链路被隔离。

## 5. 通信融合算子

为了提升计算与通信效率，HCCL 提供了一系列将矩阵乘与集合通信融合的算子，减少显存搬运和 kernel launch 开销：

- `npu_mm_all_reduce_base` — 矩阵乘 + all-reduce 融合
- `npu_mm_reduce_scatter_base` — 矩阵乘 + reduce-scatter 融合
- `npu_all_gather_base_mm` — all-gather + 矩阵乘 融合

这些算子通常在自动并行或手动并行策略中通过通信拆分（communication splitting）自动插入。

## 6. 故障排查

更多 HCCL 相关错误及排查方法，请参见 [[troubleshooting/hccl-errors]]。