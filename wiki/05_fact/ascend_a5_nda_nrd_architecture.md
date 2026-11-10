---
title: "Fact: Ascend A5 NDA/NDR 设备直驱 RDMA 通信架构 (CANN SHMEM)"
tags: [status/stable, category/fact, domain/comm, hardware/a5]
last_updated: 2026-11-10
source: 华为内部社区《Ascend 950 代际，从数据直达到设备直驱》+《AIV 直驱 RDMA 的技术路径与实践》(jx.huawei.com)，笔记归档 raw/docs/CANN_SHMEM_NDA_核心要点笔记.md，代码仓 https://gitcode.com/cann/shmem
confidence: high
status: active
open_questions:
  - "性能数据需查询 https://gitcode.com/cann/shmem/wiki/Perftest.md 实测归档"
contradictions: []
---

# Fact: Ascend A5 NDA/NDR 设备直驱 RDMA 通信架构 (CANN SHMEM)

## 1. 核心事实 (Core Fact)

**NDR（NPU Direct RDMA）**：NIC 直接访问 NPU Memory（数据直达），但通信任务、队列提交与完成推进由 CPU / Host Runtime / 代理线程组织。

**NDA（NPU Direct Async）**：Host 只完成初始化、建链和上下文准备；进入稳态后，**AIV 在 Kernel 内完成目标选择、远端访问发起、WQE 构造、QP 选择、Doorbell 提交与完成管理（Quiet/Sync/Barrier）**。CPU 退出高频逐操作提交链路，专注控制面。

| 观察点 | NDR | NDA |
|---|---|---|
| 数据路径 | NIC ↔ NPU Memory 直达 | NIC ↔ NPU Memory 直达（不变） |
| 通信目标 | Host 任务预先表达 | AIV 根据 Kernel 运行结果决定 |
| 提交主体 | CPU / Runtime / Proxy | AIV |
| 队列控制 | Host 组织 WQE 并触发提交 | AIV 构造 WQE、选择 QP 并触发 Doorbell |
| 完成管理 | Host 或代理协调 | AIV 在设备侧执行 Quiet/Sync/Barrier |
| 通算编排 | 受 Host 与 Kernel 边界约束 | 可嵌入同一 Kernel 的数据依赖流水 |

与 NVIDIA 体系类比：NDR ≈ GPUDirect RDMA + NVSHMEM IBRC proxy-based transport；NDA ≈ GDAKI（GPUDirect Async Kernel-Initiated，经 IBGDA/GPUNetIO）。双方接口、协议、队列格式、硬件实现各自独立。

### 设计动机（三个结构性矛盾）

| 结构性矛盾 | AI 负载表现 | NDA 处理方式 |
|---|---|---|
| 决策位置与发起位置分离 | 路由/分块/调度结果在 AIV 中产生，网络任务由 Host 组织 | AIV 直接把运行时结果转化为 RMA/AMO 操作 |
| 通信颗粒与 Host 任务颗粒不匹配 | Token、索引、计数器、状态标志频繁出现 | 设备侧 NBI、Multi-QP 与 Aggregate 提高高频提交效率 |
| 上层算法与网卡实现耦合 | 代际/网卡/队列变化向算子扩散 | 统一 SHMEM 语义承接通信意图，Backend 完成适配 |

关键背景：MoE 门控后 Token 目标 Expert、分块计算的就绪状态、Prefill 后 KV Block 的目标实例与 Ready 状态都在设备侧产生。若先返回 Host 再组织网络任务，控制链路被切成多段，动态目标、细粒度消息、通算融合会受损（消息率、流水深度、尾时延）。

### RDMA / RoCE 基础

- **RDMA**：网卡硬件直接在本端/远端应用内存间搬数据——Kernel Bypass（绕过协议栈）+ Zero-copy（0 次数据面拷贝，对比 TCP/IP 4 次）+ CPU 只负责发起（WQE）与完成（CQE/Quiet）。同集群 RoCE 单程 ~1–2 μs，接近线缆速率带宽。
- 前提：通信双方注册内存区域（MR）获得访问凭据（Key）；远端凭 Key + 目标地址直接读写。
- RDMA ≠ 单纯 DMA：还需 MR/Key + VA→PA 硬件翻译（指定写对端应用内存第 X 字节）、可靠传输引擎（排序/重传/流控硬件卸载）、WQ/Doorbell/CQ 用户态映射、NIC DMA 可直接寻址 NPU HBM（NPU Direct 硬件前提）。反证：SoftRoCE 纯软件模拟，时延从 ~1 μs 退化到几十 μs。
- **RoCE v2**：RDMA 语义封装进 UDP/IP（端口 4791）跑标准以太网，可三层路由，当前主流；依赖无损以太网（PFC 优先级流控 + ECN 显式拥塞通知等 DCB 特性）。A5 XSCALE（云脉）、HNS_1825 网卡经 RoCE fabric 实现本端 NPU HBM ↔ 远端 NPU HBM 直写。

### QP（Queue Pair）

QP 是 RDMA 两节点间专用通信通道 = SQ（发送队列）+ RQ（接收队列）。本端把操作组织成 WQE 放入 SQ，网卡硬件直接消费执行，绕过 Host。**关键语义：一个 QP = 一个独立排序域——QP 内严格按序，QP 之间无顺序保证**（这是 Single/Multi-QP/Aggregate 三种提交模式的根基）。类比：QP 之于 RDMA 如同 TCP 连接之于 socket，但由网卡硬件直驱。

### 一次 AIV Put 端到端六步

1. 定位远端对象：对称地址偏移 + 目标 PE + 远端内存信息 → 目标地址
2. 选择通信队列：默认路径，或 QP-specific 映射（`qp_idx = aiv_idx % qp_num`）
3. 构造 WQE：源/目标地址、长度、访问凭据
4. 提交到 NIC：WQE 进 SQ，更新队列状态并触发 Doorbell
5. 执行 RDMA：NIC 从本端 NPU HBM 读数据，经 Fabric 直写远端 NPU HBM
6. 建立完成：NIC 更新完成状态，AIV 在依赖点调 Quiet 后复用 Buffer

前提：Host 已一次性建立对称内存/连接/队列并发布通信状态（Device Communication State），AIV 稳态反复复用。

## 2. 设计背景 (Background & Rationale)

分层架构（IR-like，把"通信意图"和"网卡机制"拆开）：

| 层 | 职责 | 运行特征 |
|---|---|---|
| Host 控制面 | 进程/PE 初始化、对称内存、网络资源、连接建立、通信状态发布 | Kernel 启动前一次性完成 |
| AIV 数据面 | 远端地址计算、QP 选择、WQE 构造、队列提交、Doorbell、完成等待 | 随计算流水持续执行（稳态） |
| Backend | 统一操作转换为具体网卡的 WQE/SQ/Doorbell 动作 | 按网卡适配 |
| 完成原语 | Quiet / Sync / Barrier 把完成关系带回 Kernel 数据依赖 | 依赖点执行 |

两层 Device API：通用 SHMEM Device API（`aclshmem_putmem_nbi / getmem_nbi / quiet`，跨 Transport）+ RoCE 性能扩展（`aclshmemx_roce_*`，显式控制 QP 与提交）。

编程模型四对象：Symmetric Memory（`aclshmem_malloc` 定位远端对象）、PE/Team（参与者）、RMA/AMO（Put/Get/Atomic/Signal）、Completion/Sync（Quiet/Fence/Sync/Barrier）。映射：Tensor/Token/KV Block → 对称对象；目标卡 → PE；通信动作 → RMA/AMO；消费关系 → 完成与同步。

## 3. 应用场景与约束 (Constraints)

### Backend 能力矩阵

| 能力 | A5 XSCALE 云脉 | A5 HNS_1825 | A2/A3 IN_DIE | 作用 |
|---|:-:|:-:|:-:|---|
| Device Put/Get 与 NBI | ✓ | ✓ | ✓ | 设备侧远端读写与异步发起 |
| Quiet/Sync/Barrier | ✓ | ✓ | ✓ | 完成管理与跨 PE 协同 |
| QP-specific RMA | ✓ | ✓ | — | 多 AIV、多逻辑流并行提交 |
| RDMA Atomic | ✓ | — | — | 远端计数与状态更新 |
| Aggregate 提交 | ✓ | — | ✓ | 摊薄连续小操作提交成本 |
| 统一 SHMEM Device 语义 | ✓ | ✓ | ✓ | 屏蔽代际与 Backend 差异 |

### 三大 AI 场景

| 场景 | 设备侧编排 | 关键能力 |
|---|---|---|
| MoE Dispatch/Combine | 门控路由 → 按目标组织 Token/Metadata/Gate Weight → Put NBI → Multi-QP 并发 → Atomic/Ready 衔接远端 Expert FFN | RMA、Atomic、Multi-QP、Aggregate |
| Tile Pipeline（通算融合） | 计算 Tile N → 预取 Tile N+1（NBI 提前发起）→ 独立计算 → 依赖点 Quiet → 消费 | NBI、Quiet |
| KV Cache 传输（P/D 解耦） | Prefill 直发远端写 KV Block → 更新 Metadata → Fence/Quiet 后发布 Ready（Signal/Atomic）→ Decode 按完成协议消费 | Put、Fence/Quiet、Signal/Atomic |

约束：
- Aggregate 仅 XSCALE 与 IN_DIE 支持，HNS_1825 不支持；RDMA Atomic 仅 XSCALE 支持
- QP 数量增加会带来资源占用；Multi-QP 需稳定映射（如 `qp_idx = aiv_idx % qp_num`）保持队列顺序
- Aggregate 聚合深度是提交效率与等待时延的权衡；大消息不适用
- RoCE 依赖无损以太网（PFC/ECN），网络配置是前提条件

## 4. 相关配置或引用 (References)

```cpp
// Level 1 通用异步 RMA（默认路径）
aclshmemx_roce_put_nbi(remote_dst, local_src, elem_size, pe);
compute_independent_work();
aclshmemx_roce_quiet(pe);

// Level 2 QP-specific 扩展
aclshmemx_roce_qp_put_nbi(..., qp_idx);  // qp_idx = aiv_idx % qp_num
aclshmemx_roce_qp_quiet(qp_idx);

// Level 3 Aggregate 扩展
put_nbi(..., defer) x N;  // defer 累积
submit();                 // 最后一次统一提交
```

- 代码仓：https://gitcode.com/cann/shmem
- 性能数据归档：https://gitcode.com/cann/shmem/wiki/Perftest.md
- 架构图：`raw/docs/rdma-vs-tcpip-diagram.svg`、`raw/docs/rdma-vs-roce-diagram.svg`
- 参考资料：NVIDIA GPUDirect RDMA、NVSHMEM Device APIs（IBRC / GDAKI Transport）、NCCL Device-Initiated Communication 与 GIN、DOCA GPUNetIO

## 5. 沉淀与关联

- 关联知识点: [[wiki/03_tuning/ascend_a5_nda_submission_tuning|Tuning: NDA 提交模式选择与瓶颈调优]]
- 关联知识点: [[wiki/03_tuning/ep_zero_sync|Tuning: EP 零冗余权重同步与 Resharding 优化 (ZeroSync)]]
