---
title: "Fact: 950PR UBX 机型硬件规格与组网架构（单机16卡、两级UB互连）"
tags: [status/stable, category/fact, topic/hardware, topic/topology, topic/bandwidth, topic/ubx]
last_updated: 2026-11-18
source: [A5字节相关机型.mhtml（机型对比表快照，2026-11-18 提取），含 1024 POD / 大禹DT / 950PR UBX / 950PR-PCIE 四列]
confidence: high
status: active
open_questions:
  - UBX 与大禹DT 的 X86 CPU 具体厂商（表内未标注，仅 PCIE 版明确 AMD 9A75）
  - "(4+1)×112GB/s=500GB/s" 算术应为 560，500 疑为扣开销后有效值
  - 425T 算力未标注精度口径
contradictions: []
---

# Fact: 950PR UBX 机型硬件规格与组网架构

## 1. 核心事实 (Core Fact)

### 1.1 整机形态

| 维度 | 规格 |
|---|---|
| 形态 | **单机 16 卡，非超节点**（UB 域封闭于单机内） |
| NPU | 16×NPU，总算力 425T（口径未标注，推断 FP16） |
| CPU | X86 × 2（厂商未标注；注意 x86 ≠ Intel，AMD 亦为 x86） |
| 跨机互联 | 1825 RoCE（与大禹 DT 同款 UB→RoCE 网卡） |

### 1.2 互连结构（两级）

```
4卡内:  fullmesh（每卡出 1 口直连）
16卡内: UB 交换互联（每卡出 4 个 UB 口接 UB Switch）
跨机:   1825 RoCE（集群网络，非 UB 域）
```

### 1.3 带宽明细

| 通道 | 带宽 | 备注 |
|---|---|---|
| 4 卡内 | 500 GB/s | 材料原文 (4+1)×112 GB/s；算术应为 560，疑扣开销 |
| 16 卡全互连 | 400 GB/s | 每 NPU 4×UB 口；4×112=448 裸速率，400 为标称 |
| 跨机 RoCE | 200 Gbps = 双向 50 GB/s / NPU | 4 NPU 共用 1×1825（800Gbps 单向） |
| H2D/D2H | 64 GB/s 双向 | PCIe 5.0 ×8 |

### 1.4 内存

| 配置 | 容量 | 带宽 |
|---|---|---|
| 128GB 版 | 128 GB | 1.6 TB/s |
| **112GB 主供版** | 112 GB | 1.4 TB/s |

## 2. 设计背景 (Background & Rationale)

### 2.1 与同表其他三机型对比（A5 字节相关机型）

| 维度 | 1024 标准POD | 大禹 DT (950DT) | **UBX (950PR)** | 950PR-PCIE |
|---|---|---|---|---|
| 形态 | 超节点 1024 卡 | 超节点 64 卡(可128) | **单机 16 卡** | 单机 |
| 组网 | 16计算柜+4总线柜 POD 间 5808 | 大平层 CLOS | **两级: 4卡fullmesh+16卡UB SW** | PCIe |
| 单卡互联 | — | 1.2 TB/s 双向(12口) | **400~500 GB/s(4~5口)** | — |
| CPU | 鲲鹏950×16 (ARM) | X86×16 | **X86×2** | AMD 9A75×2 |
| 内存 | — | 96GB@4TB/s | **128/112GB@1.6/1.4TB/s** | — |
| 跨机/NPU | — | 100 GB/s 双向 | **50 GB/s 双向** | — |

### 2.2 定位分析

- UBX = 950PR 的 **UB 互联版本**：单机 16 卡中型算力节点；
- 互连与内存规格明显低于 DT 超节点（400~500 vs 1200 GB/s；1.4/1.6 vs 4 TB/s）；
- 跨机收敛更紧（每 NPU 50 vs 100 GB/s 双向），适合中小规模部署或集群基本单元；
- **关键边界：UBX 不是超节点**——超节点的定义特征是 UB 域跨单机边界（如大禹 64/128 卡、POD 1024 卡）；UBX 跨机走 RoCE，UB 域上限即 16 卡。

## 3. 应用场景与约束 (Constraints)

- 通信估算可用值：机内 16 卡 UB 域 400 GB/s（口径推断单向/双向需与材料方确认，参照 950 系惯例 400G 口为单向标称则 4 口=200GB/s 单向，材料表 400 GB/s 口径待核）；跨机 25 GB/s 单向（50 双向均摊）；
- TP≤16 可留在机内 UB 域；跨机并行策略按 RoCE 收敛比约束（与 DT 相同方法论）；
- 4 NPU 共享 1 块 1825，单 rank 可占满网卡出口，均摊值非硬保证；
- 主供内存为 112GB 版，128GB 为可选配置。

## 4. 相关配置或引用 (References)

```
同源对比表四机型: 1024标准POD / 大禹DT / 950PR UBX / 950PR-PCIE
带宽口径方法论: 见 npu_supernode_bandwidth_methodology.md（单向/双向、busbw/algbw）
```

## 5. 沉淀与关联
- 关联知识点: [[wiki/05_fact/npu_supernode_bandwidth_methodology|NPU 超节点带宽计算口径与方法论]]
- 关联知识点: [[wiki/06_ascend_stack/architecture/dayu-supernode-topology|大禹超节点架构]]
- 关联知识点: [[wiki/06_ascend_stack/references/a5-hardware-specs|A5 硬件规格]]
