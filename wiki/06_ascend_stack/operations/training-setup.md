---
title: 昇腾 NPU 训练环境配置指南
tags:
  - training
  - setup
  - environment
  - NPU
  - CANN
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# 昇腾 NPU 训练环境配置指南

## 1. 环境要求概览

- 操作系统：Linux (推荐 openEuler / Ubuntu)
- NPU 驱动：已正确安装昇腾 NPU 驱动
- CANN：昇腾 AI 计算平台
- Python：3.10+
- PyTorch：2.9.0 系列
- torch_npu：与 PyTorch 版本匹配的 NPU 适配层

## 2. CANN 版本选择

集群默认安装的 CANN 版本为 **9.1.0**，但当前训练推荐使用 **CANN 9.2.0**，路径如下：

```
/home/pkg/MAIN_0727/cann-9.2.0
```

请在训练前通过 `source` 或环境变量切换至该版本，确保与 torch_npu 的兼容性。

## 3. Conda 环境

当前维护两套 conda 环境：

| 环境类型 | 环境名称 | PyTorch / torch_npu 版本 |
|---------|---------|------------------------|
| Dev（开发验证） | `s00574452` | torch_npu 2.9.0.post3.dev20260522 |
| Release（正式训练） | `dsv4_llm` | torch_npu 2.9.0.post3 |

- 开发环境用于日常调试和功能验证，包含最新的开发特性。
- Release 环境用于正式训练任务，版本稳定。

## 4. 推荐 HCCL 环境变量（最少干扰原则）

为避免过多环境变量干扰训练行为，仅设置以下核心 HCCL 参数：

```bash
HCCL_CONNECT_TIMEOUT=7200
HCCL_BUFFSIZE=300
HCCL_EXEC_TIMEOUT=7200
HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO=0.75
```

- `HCCL_CONNECT_TIMEOUT`：建连超时时间（秒），大集群训练建议保持 7200。
- `HCCL_BUFFSIZE`：通信缓冲区大小（MB），300 为推荐值。
- `HCCL_EXEC_TIMEOUT`：通信执行超时时间（秒），与建连超时保持一致。
- `HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO`：多维度拆分比例，0.75 为经验值。

## 5. 训练脚本启动流程

### 5.1 使用 wrapper 脚本

统一使用 wrapper 脚本管理环境变量和启动参数，避免手动拼写 `torchrun` 命令。示例：

```bash
bash run_fp8_train_83.sh
```

Wrapper 脚本通常包含：CANN 版本切换、conda 环境激活、HCCL 环境变量设置、分布式启动参数组装。

### 5.2 环境探测

在启动训练前，先执行环境探测脚本，确认 NPU 驱动和 CANN 版本正确：

```bash
bash probe_env.sh
```

### 5.3 网络测试

执行网络探测脚本，验证节点间 HCCL 通信是否正常：

```bash
bash probe_net.sh
```

确保所有节点通信无异常后再提交正式训练任务。

## 6. 日志管理

训练日志统一保存至共享存储目录：

```
/mnt/share/z00583051/codes/MindSpeed-LLM/logs/
```

日志命名建议包含任务名、时间戳和节点信息，便于后续排查。

## 7. SSH 远程管理

在多节点训练场景下，需要配置 SSH 免密登录。详细配置参考：[[operations/ssh-automation]]

## 8. 参考

环境变量相关说明见：[[operations/environment-variables]]

HCCL 通信库配置：[[components/hccl]]

版本配套信息：[[references/version-compatibility]]

常用命令速查：[[references/useful-commands]]