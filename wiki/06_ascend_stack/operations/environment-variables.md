---
title: 关键环境变量参考（HCCL / CANN）
tags: [environment-variables, HCCL, CANN, configuration]
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# 关键环境变量参考（HCCL / CANN）

## 环境变量列表

| 环境变量 | 推荐值 | 说明 |
|---|---|---|
| `HCCL_CONNECT_TIMEOUT` | `7200` | HCCL 连接超时时间（秒） |
| `HCCL_BUFFSIZE` | `300` | 通信缓冲区大小（MB） |
| `HCCL_EXEC_TIMEOUT` | `7200` | HCCL 执行超时时间（秒） |
| `HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO` | `0.75` | 多维度拆分比例 |
| `ASCEND_LAUNCH_BLOCKING` | `1` | 开启同步执行（调试用） |
| `COMBINED_ENABLE` | `1` | 开启通信融合 |

## 应避免的变量

以下变量可能干扰 HCCL 通信，**不建议手动设置**：

- `HCCL_SOCKET_IFNAME`
- `HCCL_INTRA_ROCE_ENABLE`
- `HCCL_DISABLE_NHR`

## 最小干扰原则

在配置环境变量时，应遵循 **最小干扰原则**：只设置必要的变量，避免引入不必要的环境变量。过多的环境变量可能会引入不确定的通信行为，增加问题排查难度。建议仅在调试或特定场景下按需添加变量，并在生产环境中保持最小化配置。

> 更多训练配置请参考 [[operations/training-setup]]。