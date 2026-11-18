---
title: A5 通信配置与性能调优
tags:
  - A5
  - HCCL
  - CCU
  - communication
  - tuning
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# A5 通信配置与性能调优

## 一、三种通信模式对比

A5 提供三种 HCCL 通信模式，差异如下：

| 特性 | AICPU+URMA | AIV+UBMEM | CCU+URMA |
|------|------------|-----------|----------|
| 基础时延 | 大 | 小 | 小 |
| 计算核占用 | 不占用 | **占用** | **不占用**（硬件卸载） |
| 编程复杂度 | - | 简单易用，支持完备指令 | 复杂，仅静态编排，有限指令 |
| 动态编排(dispatch/combine) | 不支持 | **支持** | 不支持 |
| 通信域数量限制 | 无 | 无 | **最多 64 个** |
| 单卡多进程 | 支持 | 支持 | **暂不支持** |
| 默认模式 | **是** | 需配置 | 需配置 |
| 协议带宽利用率 | 85% | 60% | 85% |
| all2all@64P 性能 | 106 GBps | **204 GBps** | - |

## 二、模式选择配置

### 2.1 配置方式（优先级从高到低）

1. **通信域粒度配置**：`HcclCommConfig.HCCL_OP_EXPANSION_MODE`（代码中设置）
2. **环境变量**：`export HCCL_OP_EXPANSION_MODE="CCU_MS"`

### 2.2 可选值

| 值 | 说明 |
|----|------|
| `AI_CPU` | 默认模式 |
| `AIV` | AIV+UBMEM 模式 |
| `CCU_MS` | CCU 模式 |
| `CCU_SCHED` | CCU 调度模式 |

### 2.3 回退机制

AIV/CCU 模式开启条件不满足时 **自动回退** 到 AI_CPU。

### 2.4 环境变量速查

```bash
export HCCL_OP_EXPANSION_MODE="AI_CPU"   # 默认
export HCCL_OP_EXPANSION_MODE="AIV"       # AIV+UBMEM
export HCCL_OP_EXPANSION_MODE="CCU_MS"    # CCU 模式
export HCCL_OP_EXPANSION_MODE="CCU_SCHED" # CCU 调度模式
```

## 三、CCU 配置策略（重点）

CCU 资源有限，需按通信域优先级分配。CCU 不支持 >64 通信域。

### 3.1 预训练场景推荐策略

| 切分 | 通信算子 | 通信域大小 | CCU 配置策略 | 备注 |
|------|---------|-----------|------------|------|
| **TP** | allgather / reducescatter | 1/2/4/8 | **高优使用** | TP 通信，对时延敏感 |
| **EP** | alltoall / alltoallv | 4/8/16/64/128 | **EP<64 时高优使用** | alltoallv 在 AICPU 展开慢，CCU 收益明显 |
| FSDP | Allgather / reducescatter | 同 DP 域（可达千卡） | 无明显收益 | 通信量大 |
| CP | alltoall / allgather | 4/8/16/32/64 | 无明显收益 | 长序列 CP 通信，通信量大 |
| EP_DP | allgather / reducescatter | 千卡万卡 | 无明显收益 | MoE 梯度同步 |
| DP_CP | allgather / reducescatter | 千卡万卡 | 无明显收益 | 非 MoE 层梯度同步 |
| PP0 | send | 2 | 优先级低 | 通信量小可掩盖 |
| PP1 | recv | 2 | 优先级低 | 与 send 分 group 实现并发 |
| TP_CP | allreduce | 1/4/8/16/64 | 优先级低 | auxloss 计算，通信量小 |
| PP | allreduce | 2/4/8/16 | 优先级低 | auxloss 统计打印 |
| MP | allreduce | 8-256 | 优先级低 | 全局信息统计打印 |
| TP_EP_MP | tp_ep_mp | 4-128 | 优先级低 | moe token 分数 |
| ALL(default) | allreduce | 16-256 | 优先级低 | gradnorm 聚合 |

### 3.2 RL 场景推荐策略

**重要限制**：CCU 暂不支持单卡多进程！

1. **推理优先**使用 dispatch/combine（不占用 CCU 资源）
2. CCU 资源优先给训练使用，配置策略同预训练
3. dispatch/combine 不支持的场景：回退 alltoall 算子
   - CCU 资源优先给推理优化下发
   - 训练使用 AICPU

### 3.3 推理场景推荐策略

| 阶段 | 切分 | 通信算子 | 通信域 | CCU 配置策略 |
|------|------|---------|--------|------------|
| **Prefill** | TP | allgather / reducescatter / allreduce | 1/2/4/8/16 | 高优使用 |
| Prefill | EP | alltoall / alltoallv | 4/8/16/32(64/128) | EP<64 高优 |
| Prefill | CP | allgather | 2/4/8/16/32 | 无明显收益 |
| Prefill | PP | send / recv | 2 | 优先级低 |
| Prefill | DP | Gloo allreduce(cpu 侧) | 2-128 | 不涉及 |
| **Decoding** | EP | mc2 / fused mc2 | 4/8/16/32(64/128) | **走 AIV，不涉及 CCU** |
| Decoding | CP | allgather / all2all | 2-32 | 无明显收益 |
| Decoding | DP | Gloo allreduce(cpu 侧) | 2-128 | 不涉及 |
| Decoding | TP 复用 DP | all2all / reducescatter / allgather | 2/4/8/16 | 高优使用 |

## 四、双轴并发算法（A5 HCCL 流量编排）

AllReduce 采用双轴 X-Y 并发：

1. **Step1**：沿 Y 轴 fullmesh reducescatter
2. **Step2**：沿 X 轴 fullmesh reducescatter + allgather
3. **Step3**：沿 Y 轴 fullmesh allgather
4. **Step4**：沿 X 轴 fullmesh allgather（具体步骤因算法而异）

## 五、性能调优指南

### 5.1 带宽基准

- **网络总带宽**：18 × 56 × 0.85 / 1024 = **0.83 TBps**（URMA 利用率 85%）
- **GM 访存带宽**：
  - 950PR：1.6 / 2 = **0.8 TB/s**
  - 950DT：4 / 2 = **2 TB/s**
- **L2→UB 带宽**：5~6 TB/s

**关键结论**：当网络带宽（0.83 TBps）接近 GM 访存带宽（950PR 为 0.8 TB/s）时，容易出现 **访存 bound**。

### 5.2 性能诊断工具与方法

#### op_statistic.csv 分析
- 将所有卡上同个算子的耗时统计出来
- 对比最大和最小耗时差异
- **差异大**说明存在负载不均或进程不同步
- 可分析低精算子加速比（MXFP8 典型 1.6x）

#### kernel_details 分析
- 分析具体算子性能
- 关注算子泛化性、分核 tiling 等影响整网性能的因素

#### Profiling 堆栈分析
- 打印堆栈分析模型 linear 层是否为 TELinear 层
- 确认是否走低精算子分支（影响算子替换比例，新模型适配尤其重要）

### 5.3 常见 Bound 类型与优化

| Bound 类型 | 表现 | 优化方向 |
|-----------|------|---------|
| **AICPU 通信展开 bound** | AICPU 占比高 | 使用 CCU/AIV 模式卸载 |
| **CPU host 下发 bound** | host 侧耗时高 | 检查下发逻辑，批量下发 |
| **AIV/CCU/dispatch 负载不均** | 各卡算子耗时差异大 | op_statistic 定位不均衡算子 |
| **访存 bound** | GM/MTE2 占比高，接近带宽上限 | CCU 卸载、QoS 调优、访存消减 |

### 5.4 访存 Bound 优化

访存 bound 来源：SDMA 本地拷贝 + localReduce + 计算算子访存

**优化手段**：
1. **CCU 卸载**：减少访存压力
2. **QoS 分配**：
   - SDMA outstanding：**30%**
   - UBDMA QoS：**7**
3. **访存消减术**：减少不必要的数据搬运
4. **重计算注意**：MXFP8 前向为单轴量化，重计算为双轴量化（影响重计算性能）

### 5.5 调优参数速查表

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| SDMA outstanding | 30% | 访存 QoS 控制 |
| UBDMA QoS | 7 | UB DMA QoS 等级 |
| URMA 带宽利用率 | 85% | 理论利用率 |
| UBMEM 带宽利用率 | 60% | 理论利用率 |
| MXFP8 典型加速比 | ~1.6x | 算子级加速比 |

### 5.6 性能看板

- 性能看板地址：http://board.ascend.huawei.com/pages/specs.html

## 六、端口配置速查

| 卡型 | PCIe 端口 | UB 端口 | 备注 |
|------|---------|--------|------|
| 4P 卡 | UDIEO P0-P3 | UDIEO P4-P6（3 口） | 支持四卡互联，双向 300GB/s |
| 2P 卡 | UDIEO P0-P3 | UDIEO P4-P8（5 口） | P4-P6 连 UB SW，P7-P8 组成 4 卡 fullmesh，双向 400GB/s |
| UBX 液冷卡 | PCIe 5.0 X8（2 个 UB 端口） | 4 个连 UB SW + 3 个 4 卡 fullmesh | 16 卡全互连 400GB/s |
| 大禹 DT 卡 | PCIe 5.0 X16（4 个 UB 端口） | 12 个 UB 端口连 UB 总线 | 64 卡全互连 1.2TB/s |

## 七、常见问题排查

### 7.1 CCU 相关问题

1. **CCU 不生效？**
   - 检查通信域数量是否超过 **64**
   - 检查是否为 **单卡多进程**（CCU 暂不支持）
   - 检查模式配置优先级（通信域配置 > 环境变量）
   - 确认是否自动回退到 AICPU（查看日志）

2. **EP 通信慢？**
   - EP 通信域 <64 时优先配置 CCU
   - alltoallv 在 AICPU 展开慢，CCU 收益明显
   - EP≥64 时考虑 AIV 模式

3. **CCU 开启后性能没提升？**
   - 检查是否通信域过大（CCU 对大通信域收益不明显，如 FSDP/CP）
   - 检查是否访存 bound，CCU 无法解决访存瓶颈
   - 确认算子是否支持 CCU 卸载

### 7.2 性能问题排查清单

1. 先确认当前使用的通信模式（AICPU/AIV/CCU）
2. 用 `op_statistic.csv` 检查各卡算子耗时差异（判断负载不均）
3. 计算网络带宽和访存带宽，判断是否访存 bound
4. 检查 CCU 资源分配是否符合推荐优先级
5. 调整 SDMA outstanding（30%）和 UBDMA QoS（7）参数
6. Profiling 确认算子替换比例（Topk/LMHead 除外应均走低精）
7. 重计算场景确认 MXFP8 量化轴配置

## 参考文档

- A5 集合通信技术 260706.pptx（作者：程祥乐）
- 训练全场景方案.pptx（作者：王铭震）