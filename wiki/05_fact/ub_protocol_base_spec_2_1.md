---
title: "Fact: 灵衢 UB (UnifiedBus) 基础规范 2.1 协议要点（面向 950 训练性能优化与调试）"
tags: [status/stable, category/fact, ub, unifiedbus, supernode, hccl, performance, debug]
last_updated: 2026-11-18
source: "UB-Base-Specification-2.1.0-zh-clean.pdf（华为，2026-09 发布 2.1 版）；协议官网 https://www.unifiedbus.com/zh"
confidence: high
status: active
open_questions: []
contradictions: []
---

# Fact: 灵衢 UB (UnifiedBus) 基础规范 2.1 协议要点

## 1. 核心事实 (Core Fact)

UB（灵衢，UnifiedBus）是华为定义的**面向超节点的互联协议**（Scale-Up 域内统一互连），将 IO、内存访问、消息传递、过程调用统一在单一协议栈，避免协议转换开销。规范版本 2.1（2026-09），协议来源：https://www.unifiedbus.com/zh。

**协议栈分层**（自底向上）：
- **物理层**：自定义速率（不受标准速率限制，支持 224Gbps）、多 FEC 模式（RS(128,120,T=2/4)、RS(256,240,T=8)）及动态切换、故障降速/降lane 恢复升速/升lane、支持非对称上下行链路宽度（1/2/4/8 Lane）。
- **数据链路层**：点到点可靠传送。CRC30/CRC62 校验、GoBackN 重传（Retry Buffer）、**信用流控**（按 VL 粒度）、**虚通道 VL**（每链路最多 16 个）、包率控制（NOB 填充）。
- **网络层**：Domain 内/跨 Domain 路由。标准 IP 地址 + 简短网络地址 CNA（16/24-bit）、**逐包/逐流多路径负载均衡**（RT 域段）、SL-VL 映射 QoS、拥塞标记（FECN/CAQM）、网络隔离 NPI、死锁避免、ICRC 端到端校验、可靠重路由。
- **传输层**：三种模式 + Bypass。**RTP**（面向连接端到端可靠，PSN+应答+重传）、**CTP**（轻量级，可靠传输卸载给链路层，无传输层应答/重传）、**UTP**（无连接不可靠）；**TP Bypass**（事务层直调网络层，降协议开销）；TPG 传输通道组多通道负载均衡；拥塞控制（窗口/速率/CAQM）。
- **事务层**：内存/消息/维护/管理四类事务（Write、Read、Atomic×9、Send、Write_with_remote_sync 等）；四种事务服务模式 **ROI/ROT/ROL/UNO**；事务切片；TAACK 聚合。
- **功能层**：Load/Store 同步访问、**URMA 异步访问**（Jetty + SQ/RQ/CQ/EQ 队列模型，类 RDMA）、**URPC**（Client/Server/Worker 平等远程函数调用、引用传参）、多 Entity 协同（融合操作、集合通信操作）。
- **横切组件**：UBFM（UB Domain 管理者：计算/互连/通信资源管理）、UMMU（内存地址映射+权限校验：TECT→TCT→MATT/MAPT 四级查表）、安全（设备认证/分区隔离/访问控制/数据通路保护/TEE）。

**关键模型**：
- Home-User 访问模型：Home 为内存拥有者，User 为访问者；User 携带 UBMD（EID + TokenID + UBA）访问 Home 内存，Home 侧 UMMU 完成地址转换（MATT）与权限校验（MAPT）；User 侧 UB Decoder 将本地 PA 查表转为 UBMD。
- Entity 是资源分配与通信对象的基本单元（EID 唯一标识）；UBPU 是支持 UB 协议栈的处理单元，协议地位平等，任意 UBPU 互访无需第三方介入；UBoE 通过以太/IP 承载 UB 事务实现跨 Domain 互通。

## 2. 设计背景 (Background & Rationale)

- **为什么协议归一**：超节点内 NPU/SSU/CPU 等处理单元间同时存在内存语义（权重/激活值搬运）、消息语义（控制消息）、过程调用（存储卸载），分立协议栈（PCIe+以太+私有 DMA）存在转换开销；UB 用单一事务层屏蔽编程模型差异。
- **为什么多层重传**：链路层点到点 GoBackN 重传把误码消灭在单跳内（不扩散到上层），传输层端到端重传兜底路由丢包/乱序，两级配合使"直连高可靠链路走 CTP 免端到端重传"成为可能——性能与可靠按链路质量弹性伸缩。
- **为什么信用流控而非反压**：信用证机制使发送端实时掌握接收端 Buffer 余量，原理上保证不溢出、无需提前反压；但逐级反压可能扩散拥塞（FECN 拥塞扩散检测）甚至形成 VL 循环依赖死锁（网络层提供三种解死锁机制）。
- **为什么四种事务服务模式**：保序位置决定性能——ROI 在发送端保序（等响应再发，多一个 RTT）、ROT 在目的端保序（省一个 RTT，需 Target 缓存乱序包 + SC 资源）、ROL 卸载给下层（RTP 下靠同 TP Channel 按序递交，CTP/Bypass 下被迫单路径）；无保序需求的可靠事务四种模式无差异，可全乱序+多路径。

## 3. 应用场景与约束（950 训练性能优化视角）(Constraints)

### 3.1 传输模式选择：开销 vs 可靠性
| 模式 | 开销 | 适用 |
|---|---|---|
| RTP | 大（PSN/应答/端到端重传） | 跨多跳交换机、链路质量不确定 |
| CTP | 小（可靠传输卸载给链路层重传） | **UBPU 直连或高可靠链路（超节点内 NPU 间首选）** |
| UTP | 最小 | 丢包不敏感（带内建连） |
| TP Bypass | 无传输层 | Load/Store 同步访问 + 精简 BTAH，极致降协议开销 |

950 训练中 HCCL 集合通信若走 UB 域内直连高可靠链路，选 CTP 免端到端重传开销；跨柜/经 Switch 的路径用 RTP。

### 3.2 多路径带宽利用率（大集合通信打满带宽的关键）
- 网络层 RT 域段四种组合：逐流/逐包 × 全部可达路径/最短路径集。**保序需求 → 逐流（单路径）；无保序 → 逐包（多路径）**。
- 传输层 TPG：一对 UBPU 间建多条 TP Channel（绑定不同物理端口/不同 UDP source port 或 LBF），调度算法（RoundRobin/按队列深度/按拥塞程度）分发；TP Channel 与逐包负载均衡可叠加。
- 乱序接收：PSN 乱序区间可配 [128,256,512,1024,2048]（TP Packet 个数）；配合 TP Sender 最大发送窗口限制在途包数。
- 逐包负载均衡下避免不必要重传：不使用快速重传、TPSACK 累计触发；拥塞控制建议把网络当整体（避免单路径拥塞引发全通道降速欠吞吐）。

### 3.3 拥塞控制
- 反馈通道：TPACK-CC/TPNAK-CC/TPSACK-CC/CNP（CETPH 头承载）；支持 LDCP（窗口）、DCQCN（CNP）、CAQM（交换机主动队列管理：随包携带窗口增长请求，逐 Switch 决定是否批准）、自定义。
- 网络层 FECN（2bit 拥塞程度 + Loc 拥塞位置）、FECN_RTT（时间戳测 RTT）；拥塞扩散检测只在源头打标（可配）。
- CC Context 按 TP Channel 粒度维护（cw/inflight/aval_win）；逐流负载均衡下每 TP Channel 独立拥塞控制。

### 3.4 链路层流控与包率控制
- 信用流控：1 Cell = n×Flit（n∈{1..128} 协商），独占/共享两种模式；**共享模式**（SHARE_CRD + 各 VL 少量独占保底防饥饿）提高 Receive Buffer 利用率、可按 VL 流量动态调整——高并发训练通信的关键。
- 信用证返还粒度（DATA_CREDIT_GRAIN_SIZE）需保证满足线速，否则 Crd_Ack Block 影响发送性能；待返还信用证达阈值需反压网络层并强制发 Crd_Ack Block。
- 包率控制：接收端小包处理能力有限，发送端按静态/动态模式插 NOB 控制包率——训练中大量小控制消息场景注意 PACKET_MIN_INTERVAL 等参数。

### 3.5 事务层性能机制
- **事务切片**：大事务（如 1MB Write）切片分散到不同 TP Channel，防止单通道长时间阻塞；切片粒度可配（如 64KB）。
- **TAACK 后向聚合**：连续 TASSN 的正常 TAACK 聚合为一个返回（最多 10 个）；ROL 模式下 TAACK 由传输层应答捎带（无独立 TAACK 包）。
- CTP + BTAH.No_TAACK=1：目的端不回 TAACK，极致省响应开销（仅网络状态极佳场景）。
- 保序模式与下层组合约束（规范表 7-15/7-16）：ROI/ROT → RTP 指定 TPG/多 TP Channel + 网络层可逐包；ROL → RTP 单 TP Channel 或 CTP/Bypass+逐流单路径；无保序事务默认可多路径乱序。
- 重传算法选择：GoBackN+快速重传（逐流、丢包极低）/ GoBackN 无快速重传（逐包）/ 选择性重传（丢包较高，需乱序接收+BitMap）；逐流负载均衡下 MarkPSN 机制加速非首次丢包识别，免等超时。
- 超时重传：静态 [512μs,16ms,128ms,4s] 或动态指数退避 RTO = Base_time×2^(N×Times)（建议 Base_time 与 RTT 正相关，UB 支持测 RTT）；超时间隔过小可能在路由收敛期误判路径不可达。

### 3.6 URMA/URPC 与集合通信
- URMA：Jetty 通信单元 + SQ/RQ/CQ/EQ/JFC 队列模型（异步提交、完成通知、异常事件），支持 ROI/ROT/ROL/UNO；训练通信库（HCCL 底层）编程模型对应 URMA。
- URPC 引用传参：参数数据地址传给 Worker，由 Worker 控制拉取时机，**规范明确适用"AI 训练/推理场景，数据传输和 NPU 运算需相互掩盖执行时间"**。
- 多 Entity 协同定义了**集合通信操作**框架：单次功能调用分解为多个 UB 事务，动态映射至各 UBPU 协同执行，减少数据移动次数、提升同步效率——即 UB 原生集合通信的协议基础（HCCL over UB 的依托）。

## 4. Debug 排障要点（训练 Debug 视角）(References)

### 4.1 分层错误检测链路（按"哪里坏→哪里报"定位）
| 层 | 检测/恢复机制 | 上报 |
|---|---|---|
| 物理层 | FEC 纠错、Retrain（降速/降lane 恢复）、链路训练状态机 | LinkUp==0 断链指示 |
| 链路层 | CRC30/CRC62、GoBackN 重传、Retry Buffer | Receive Buffer Overflow / Flow Control Overflow / DL Protocol Error（信用证返还超时）/ DL Retry ACK Timeout / DL Retry Rollover / DL Retry Error |
| 网络层 | ICRC 端到端校验（失败丢包）、死锁避免（路由防死锁/VL 主动切换/超时丢包）、断链可靠重路由 | 错误事件 |
| 传输层 | PSN 重复/乱序/无效区间判定、GoBackN/选择性重传、超时重传 | 重传超最大次数 → CQE 上报路径不可达 |
| 事务层 | RNR/Page Fault 重传；长度/权限类异常不可重传上抛 | TAACK.RSPST + CQE Status/Sub-Status |

### 4.2 RAS 错误三级分类（定位错误影响范围）
- **A 类（单事务）**：通过 CQE 上报（Status/Sub-Status 编码）：Unsupported Opcode(0x01)、Local Length(0x02/0x1)、Local Access 权限(0x02/0x2)、Remote Length/Access/Poison(0x03)、事务重传超次(0x04)、**事务应答超时(0x05)**；UMMU 侧经 EQE 上报：E_BAD_TECT、E_BAD_TOKENID、E_BAD_TCT、E_A_TRANSLATION、E_A_ADDR_SIZE、E_P_CFG_ERROR、E_P_PERMISSION（权限校验未通过）、E_EBIT_DENY（排他校验失败）。
- **B 类（Entity/TP Channel/Jetty 级）**：JFS/JFC Check Error、**Transport Retry Counter Exceeded（传输层重传超次）**、Transport Unexpected Opcode、**PSN 与事务 MSN 不匹配**——上报异步事件，驱动处理。
- **C 类（设备/端口级公共资源）**：拓扑损坏等，UBFM 处理；细分可纠正错误与不可纠正错误（NFUE/FUE）。
- 复位粒度三级：设备级（全 Entity+Port，仅 Entity 0 有复位寄存器）、Entity 级（不影响网络/链路/物理层公共配置）、Port 级（路由表标记不可用）。

### 4.3 典型训练故障的协议层线索
- **训练 hang/通信卡死**：信用证返还超时（DL Protocol Error）、VL 循环依赖死锁（查网络层死锁三机制是否启用）、Retry Buffer 互锁（Crd_Ack Block 优先发送阈值）、事务层 RNR 重试风暴。
- **性能劣化不报错**：链路 Retrain 降速（查 FEC 译码失败统计/重训计数 NUM_PHY_REINIT）、重传计数升高（NUM_RETRY 阈值 15 建议值）、PSN 乱序区间不足引发频繁 TPSACK/重传、拥塞窗口震荡（FECN 标记、CNP 风暴）、包率控制 NOB 插入过多（小包场景）、LBF/路由哈希不均（多路径带宽未打满）。
- **静默数据问题**：ICRC 校验失败静默丢包靠传输层重传恢复；链路层不攒齐发送的 Packet 用 ERROR_FLAG/Poison 标错，在攒齐点丢弃——误码不会引入上层。
- **Atomic 不可重传约束**：Atomic Response 阶段 RNR/Page Fault 重传会导致重复执行（如 fetch_add 两次加 2）破坏一致性，只能上报；Request 阶段异常可重传。
- **超时参数**：链路层 WAIT 超时建议 >2 倍链路时延（1μs~10s）；传输层静态超时 [512μs..4s]、动态指数退避防路由收敛期误判。

### 4.4 关键状态查询入口
- 重传状态机状态：Data Link State Machine Status 寄存器 Retry Req Status / Retry Ack Status 域段（规范附录 D.6.2.3）。
- Retry Buffer 管理：NumFreeBuf/WrPtr/TailPtr/RdPtr（发端维护）、RcvPtr（收端维护）；ACK 溢出 → DL Protocol Error。
- 配置空间（CFG0/CFG1 寄存器组）+ Resource Space 提供 Entity 管理接口；UBFM 统一拓扑/路由/地址管理。

## 5. 沉淀与关联
- 关联知识点: [[wiki/05_fact/ubx_950pr_hardware_specs|950PR UBX 机型硬件规格与组网架构]]（UB 协议在 950PR 上的物理承载）
- 关联知识点: [[wiki/05_fact/npu_supernode_bandwidth_methodology|NPU 超节点带宽计算口径方法论]]（UB 链路带宽标称/实测口径）
- 关联知识点: [[wiki/06_ascend_stack/architecture/dayu-supernode-topology|大禹超节点架构与 D2D 带宽口径]]（UB 域拓扑与跨柜互连）
- 关联知识点: [[wiki/03_tuning/ascend_a5_nda_submission_tuning|Ascend A5 NDA 提交模式与通信瓶颈调优]]（HCCL/UB 通信调优实践）
