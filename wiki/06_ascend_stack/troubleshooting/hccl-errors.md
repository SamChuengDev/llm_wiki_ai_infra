---
title: HCCL 通信错误排查指南
tags:
  - HCCL
  - communication-error
  - troubleshooting
  - RootInfoDetect
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# HCCL 通信错误排查指南

## 1. 常见 HCCL 错误及排查步骤

在使用昇腾 NPU 进行分布式训练时，HCCL（华为集合通信库）错误是常见问题。以下为系统化的排查思路：

1.  **检查硬件状态**：优先排查 NPU 硬件是否出现异常事件。
2.  **检查环境变量**：确认 HCCL 相关环境变量配置是否合理，避免使用不推荐的变量。
3.  **检查网络连通性**：使用 `probe_net.sh` 脚本测试节点间网络是否正常。
4.  **检查环境配置**：使用 `probe_env.sh` 脚本验证运行环境是否正确配置。

## 2. 错误: `hcclCommInitRootInfoConfig failed, error code: 4, RootInfoDetect failed`

### 根因

该错误通常由 **NPU 硬件 RAS 事件** 导致。当 NPU 硬件发生可修复或不可修复的错误时，RAS 模块会记录事件并上报，可能导致 HCCL 初始化 RootInfo 失败。

### 排查步骤

1.  **检查 RAS 事件**：
    ```bash
    dmesg | grep 'ras event'
    ```
    - 如果输出中有硬件错误信息，需进一步排查硬件状态。
2.  **检查 HCCL 环境变量**：
    - 如果无 RAS 事件，则检查 HCCL 环境变量配置是否合理。
    - 确保未设置不推荐的环境变量（见第 4 节）。
    - 确认推荐的 HCCL 环境变量已正确设置（见第 3 节）。

## 3. 推荐环境变量（基于实践验证）

以下环境变量经过了大规模训练实践的验证，可以有效提升 HCCL 通信的稳定性：

| 环境变量 | 推荐值 | 说明 |
|---------|--------|------|
| `HCCL_CONNECT_TIMEOUT` | `7200` | HCCL 建连超时时间（秒），训练任务多机规模大时建议设置较大值 |
| `HCCL_BUFFSIZE` | `300` | HCCL 通信缓冲区大小 |
| `HCCL_EXEC_TIMEOUT` | `7200` | HCCL 执行超时时间（秒），长耗时通信场景建议设置较大值 |
| `HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO` | `0.75` | 多维度拆分算法的比例因子，优化通信效率 |

## 4. 应避免的变量

以下环境变量可能干扰 HCCL 通信，**不建议设置**：

| 环境变量 | 原因 |
|---------|------|
| `HCCL_SOCKET_IFNAME` | 手动指定网络接口可能限制 HCCL 的选路能力，导致通信异常 |
| `HCCL_INTRA_ROCE_ENABLE` | 强制开启或关闭 RoCE 内部通信可能影响多节点通信稳定性 |
| `HCCL_DISABLE_NHR` | 禁用 NHR（Non-Homogeneous Ring）算法可能降低通信效率或引发兼容性问题 |

## 5. 网络诊断

使用 `probe_net.sh` 脚本测试节点间的网络连通性：

```bash
bash probe_net.sh
```

该脚本会检查：
- 节点间网络延迟和带宽
- RoCE 网卡状态
- 防火墙和端口连通性

## 6. 环境探测

使用 `probe_env.sh` 脚本检查运行环境配置：

```bash
bash probe_env.sh
```

该脚本会检查：
- NPU 驱动版本和固件版本
- HCCL 库版本
- 环境变量设置
- 系统内核参数配置