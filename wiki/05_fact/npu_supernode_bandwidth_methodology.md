---
title: "Fact: NPU 超节点带宽计算口径与方法论（单向/双向、Scale-Up/Out、busbw/algbw）"
tags: [status/stable, category/fact, topic/bandwidth, topic/supernode, topic/hccl]
last_updated: 2026-11-18
source: [大禹 64/128 卡方案材料推导（2026-11-17/18 会话），沉淀于 raw/docs/2026-11-17-950dt-dayu-64-128card.md]
confidence: high
status: active
open_questions:
  - 384 个 800G 光模块是单柜还是整系统 BOM 口径，影响跨柜带宽精确值
  - 1825 网卡与 NPU 的具体绑定关系（每卡服务 2 NPU 为图面推断）
contradictions: []
---

# Fact: NPU 超节点带宽计算口径与方法论

## 1. 核心事实 (Core Fact)

超节点带宽计算必须先统一**三层口径**，否则所有数字相差 2 倍以上：

### 1.1 端口速率口径：400G 是单向

- 400G 端口 = 4 lane × 112G SerDes = 448 Gbps 裸速率，扣编码/FEC 后 400 Gbps **单方向**额定值；
- 全双工 = TX 400G + RX 400G 独立并行；
- **判据**（强度递减）：
  1. 材料自证：`18 ports × 56 GB/s = 1.008 TB/s`（单向）/ `× 112 GB/s = 2.016 TB/s`（双向）→ 单口单向 56 GB/s = 448 Gbps；
  2. 算术自洽：12 口 × 400G × 2 ÷ 8 = 1.2 TB/s 与材料 "D-D 1.2TB/s" 吻合，若 400G 为双向则差一倍；
  3. 行业惯例：端口/网卡/光模块线速标称均为单向。

### 1.2 领域惯例：互连用双向，端口用单向

| 领域 | 口径 | 例子 |
|---|---|---|
| 芯片间互连（NVLink/UB/xGMI） | **双向聚合**（营销惯例） | H100 900 GB/s = 450×2；UB 1.2 TB/s = 600×2 |
| 网络端口/网卡/光模块（Ethernet/IB/RoCE） | **单向** | 400G 网卡、800G 光模块 |
| PCIe | 单向 | PCIe5 x16 = 64 GB/s/方向 |
| 内存（HBM/DDR） | 单向（读写分时共享总线） | HBM3 815 GB/s |

**对比规则**：跨域对比先换算成单向；拿不准找规格书脚注 `bidirectional aggregate` / `per direction`。

### 1.3 通信测试口径：algbw/busbw 均为单向

```
algbw = S ÷ t                    # 应用层吞吐（算子视角）
busbw = algbw × 2(N-1)/N         # AllReduce 专用（链路视角，归一化为每链路单向）
```

- busbw 虽在 ring 收发并行时测得，但公式已归一化到**单向口径**，可与硬件单向量直接对比；
- 不同算子归一化因子不同：AllGather/ReduceScatter 为 `(N-1)/N`，P2P 为 1；
- **常见误判**：用测试单向带宽去比硬件双向标称（如 300 vs 1200），误以为只跑出 25%，实际已打满。

## 2. 设计背景 (Background & Rationale)

### 2.1 双向带宽的物理本质与利用率

- 双向聚合 = TX 独立通道 + RX 独立通道；**单方向负载（只读/只写）永远只能用一半**（1.2 TB/s 链路单向发送上限 600 GB/s）；
- 双向数字只在**收发并行的算法**里兑现：
  - Ring AllReduce / ring 实现的 AllGather/ReduceScatter：每 rank 同时收上一环+发下一环，★★★；
  - AllToAll（EP/MoE）：成对并发，★★~★★★；
  - Broadcast、单向 P2P：只用单方向，★；
  - 成对 batch send/recv（如 PP 前传激活+回传梯度同调度）：也能打满双向，关键在**链路上是否同时存在双向数据流**，不在算子名字。

### 2.2 大禹超节点三层带宽实例（推导值）

| 通道 | 路径 | 单向带宽 | 双向带宽 |
|---|---|---|---|
| 柜内 Scale-Up（UB CLOS） | NPU 12 口 × 400G → 总线节点 | 600 GB/s/卡 | 1.2 TB/s/卡 |
| 跨柜（128 卡双柜） | 总线节点间剩余 768 口 | 300~600 GB/s/卡（BOM 口径待确认） | 600 GB/s~1.2 TB/s/卡 |
| 跨超节点 Scale-Out | 每计算节点 2×1825 UB→RoCE 400G 网卡 | 100 GB/s/节点（25 GB/s/NPU 均摊） | 200 GB/s/节点（50 GB/s/NPU） |

- **收敛比 Scale-Up:Scale-Out ≈ 12:1**（600:50 单向均摊），决定并行策略：TP/EP 大流量关在 UB 域内，跨超节点只走 PP/DP/数据加载；
- 1825（推断为 Hi1825 系 UB→RoCE 400G 网卡）是 **UB 域 → 以太网域的边界转换点**，同时经 PCIe 5.0 服务主机侧，主机流量会挤占同一出口；
- 跨柜带宽来自总线端口差额：128 卡方案 12×128=1536 口 vs NPU 768 口，差额 768 口即柜间通道（64 卡方案 6×128=768 恰好用完无跨柜余量）。

### 2.3 端口数交叉校验法

无收敛 CLOS 的验证：**NPU 侧端口总数 = 总线节点侧端口总数**（64×12 = 6×128 = 768）。带宽推导后务必做此校验，防止 BOM 表理解偏差。

## 3. 应用场景与约束 (Constraints)

- 通信时间估算必须用**单向**值：柜内 600 GB/s、跨超节点 25 GB/s（NPU 均摊）或 100 GB/s（节点聚合）；
- per-NPU 均摊值（如 50 GB/s 双向）不是硬件保证：NPU 通过 UB 域共享网卡出口，单 rank 可占满整个 200 GB/s；
- Ring AR 耗时公式 `2(N-1)/N × S ÷ B` 中 B 取单向 600 GB/s，因子已隐含双向并行收益，**不要再乘 2**；
- UB 域内 ring 相邻 rank 应落同一总线节点平面，跨平面跳步实际带宽低于线速；
- 双向打满时瓶颈可能转移到 HBM 读写或协议处理，以 benchmark 为最终裁判；
- 所有 1.2 TB/s 等数字为名义线速之和，非实测。

## 4. 相关配置或引用 (References)

```
大禹 64 卡: 16 计算节点 + 6 总线节点(2OU)；每 NPU 12×400G UB 口；D-D 1.2TB/s
大禹 128 卡: 双柜，每柜 16 节点 + 12 总线节点(1OU)；BOM 384×800G 光模块
计算节点: 4×NPU 模组(950DT+冷板) + 2×UB→RoCE 400G 网卡(1825) + AMC
```

## 5. 沉淀与关联
- 关联知识点: [[wiki/06_ascend_stack/architecture/dayu-supernode-topology|大禹超节点架构]]（含 D2D 带宽口径章节）
- 关联知识点: [[wiki/06_ascend_stack/references/a5-hardware-specs|A5 硬件规格]]
- 关联知识点: [[wiki/06_ascend_stack/operations/a5-communication-tuning|A5 通信调优]]
