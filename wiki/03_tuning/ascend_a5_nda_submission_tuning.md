---
title: "Tuning: Ascend A5 NDA 提交模式选择与通信瓶颈调优"
tags: [status/verified, category/comm, hardware/a5]
last_updated: 2026-11-10
source: 华为内部社区《AIV 直驱 RDMA 的技术路径与实践》第 5/7/10 节，笔记归档 raw/docs/CANN_SHMEM_NDA_核心要点笔记.md
confidence: high
status: active
open_questions:
  - "各瓶颈类型的量化阈值（消息率/Doorbell 成本）需结合 Perftest 实测数据确认"
contradictions: []
---

# Profiling & Tuning Report: NDA 提交模式选择与通信瓶颈调优

前置知识：[[wiki/05_fact/ascend_a5_nda_nrd_architecture|Fact: Ascend A5 NDA/NDR 设备直驱 RDMA 通信架构]]

## 1. 瓶颈分析 (Bottleneck Analysis)

NDA 场景下五类通信瓶颈的特征识别：

| 瓶颈类型 | 常见表现 | 有效方向 |
|---|---|---|
| Host/Control-bound | 设备结果频繁返回 Host，下发进关键路径 | NDA 让 AIV 直接发起通信 |
| Issue/Queue-bound | 多 AIV 单队列串行提交 | Multi-QP |
| Submission-bound | 小消息多，Doorbell/提交成本高 | 数据连续化 + Aggregate |
| Bandwidth-bound | 大消息接近链路上限 | 优化消息组织、拓扑、计算重叠（**不盲目加 QP/聚合深度**） |
| Dependency-bound | Quiet/Barrier 过早，通信未被计算覆盖 | 重新放置完成点，缩短暴露等待 |

## 2. 优化方案 (Optimization Strategy)

### 方案 A: 异步三段式（NBI + 依赖点 Quiet，最核心使用方式）

- **原理**: Issue Early → Compute in Between → Complete on Dependency。Quiet 位置跟随数据流而非机械跟随每次 Put/Get：正确性要求依赖该数据的下一阶段前必须完成；性能上 Quiet 越靠近真实依赖，可重叠窗口越完整。
- **实施步骤**:
  ```cpp
  aclshmemx_roce_put_nbi(dst, src, elem_size, pe);  // 尽早发起
  compute_independent_tile();                        // 重叠无依赖计算
  aclshmemx_roce_quiet(pe);                          // 真实依赖点完成
  consume_or_reuse_buffer();
  ```
- **Quiet 场景化放置**: MoE → 目标分组发送结束后；Tile Pipeline → 下一 Tile 消费前；KV Cache → Ready 状态发布前。

### 方案 B: Multi-QP（解决 Issue/Queue-bound）

- **原理**: 通信流映射到多个独立 QP 并行提交；QP 内有序、QP 间无序。
- **实施步骤**: 按 AIV / Channel / 数据类型 / 目标 PE 分配 QP，`qp_idx = aiv_idx % qp_num`；稳定映射保序；按 QP 分别 Quiet（队列级完成）。
- **接口**: `aclshmemx_roce_qp_put_nbi / qp_quiet` + `qp_idx`

### 方案 C: Aggregate（解决 Submission-bound）

- **原理**: 单 QP 内多个 WQE 聚合（defer 累积），最后一次 submit 统一提交，一组小操作共享一次 Doorbell。
- **实施步骤**: `put_nbi(..., defer) × N` + 末次 `submit`；聚合深度 = 提交效率与等待时间的平衡；大消息不适用。
- **与 Multi-QP 关系**: 维度不同可组合——Aggregate 优化单队列内提交频率（纵向），Multi-QP 扩展排序域并行度（横向）。

### 完成语义四原语的选择

| 原语 | 语义 | 典型位置 |
|---|---|---|
| Fence | 建立前后远端操作的**顺序** | 数据与后续控制操作需有序时 |
| Quiet | 等待发起域中**前序通信完成** | Buffer 复用、Get 消费、阶段切换前 |
| Sync | 参与 PE **到达同一同步点**（会合） | 各 PE 完成本地通信阶段后 |
| Barrier | 前序更新完成 **+ 参与者会合** | 全体 PE 共享同一数据依赖时 |

## 3. 优化效果 (Results)

- **渐进式调优路径**（官方推荐顺序）:
  1. 通用语义跑通数据流：对称内存、PE、Put/Get NBI、Quiet，先定义清远端对象、Buffer 生命周期、消费依赖
  2. 把 Quiet 移到真实依赖点：识别可与通信重叠的计算，形成基本通信-计算流水
  3. 按 Profiling 增强提交能力：单 QP 串行化 → QP-specific；小操作提交频繁 → Aggregate；带宽受限 → 优先数据连续化与流水覆盖
- **评估口径**: 同时观察消息率、带宽、完成时延、端到端算子阶段时间；微基准找提交/网络上限，真实场景（MoE/Tile Pipeline/KV Cache）阶段时间判断优化是否进入业务关键路径。
- **注意事项**: 上层算法不因性能层次变化而重写通信模型——统一语义层保持稳定。QP 数量增加带来资源占用；Aggregate 不适用于大消息；HNS_1825 不支持 Aggregate 与 RDMA Atomic（Backend 能力差异见前置知识页矩阵）。

### Profiling 现象 → 优先动作速查

| Profiling 现象 | 优先考虑 |
|---|---|
| 通信可与计算重叠，但路径仍逐操作等待 | NBI + 依赖点 Quiet |
| 多 AIV/数据流集中单队列，Issue 串行化 | Multi-QP |
| 连续小操作多，Doorbell/提交频率高 | Aggregate |
| 大消息已接近链路带宽 | 优化数据组织与流水，不盲目加 QP/聚合深度 |

## 4. 关联知识

- [[wiki/05_fact/ascend_a5_nda_nrd_architecture|Fact: Ascend A5 NDA/NDR 设备直驱 RDMA 通信架构]]
- [[wiki/03_tuning/ep_zero_sync|Tuning: EP 零冗余权重同步与 Resharding 优化 (ZeroSync)]]
