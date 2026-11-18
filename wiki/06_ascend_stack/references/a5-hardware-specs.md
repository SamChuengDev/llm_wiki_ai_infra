---
title: A5 硬件规格参考
tags:
  - A5
  - hardware
  - specs
  - 950
  - topology
created: 2026-08-05
updated: 2026-11-17
last_updated: 2026-11-17
source: ["migrated from ascend-llm-wiki"]
---

# A5 硬件规格参考

## 一、A2 / A3 / A5 代际对比

| 维度 | A2 (910B) | A3 (910C) | A5 (950DT/950PR) |
|------|-----------|-----------|------------------|
| FP16 矩阵算力 | 354T | - | 432T(DT)/378T(PR) |
| FP16 向量算力 | 22T (1:16) | - | 54T(DT)/47T(PR) (1:8) |
| Cube FP32 算力 | 基准 | - | 仅 A2 的 1/3（需用 HF32） |
| 显存容量 | 64GB | 1024GB HBM(8卡) | 96GB(DT)/128GB(PR) |
| 显存带宽 | 1.6TB/s | - | 4TB/s(DT)/1.6TB/s(PR) |
| L2 Cache 容量 | 192MB | - | 128MB |
| L2 Cache 带宽 | 7.86TB/s | - | 5.2TB/s |
| 卡间互联 | HCCS 392GB/s | HCCS | UB 总线 |
| 单卡出口带宽 | 机内196GB/s / 机间25GB/s | 机内392GB/s / 机间50GB/s | 18×56GBps=1.008TBps |
| 内存语义范围 | 单机 8 卡 | 超节点 768die | K 级规模 |
| 通信加速 | SDMA/RDMA/AIV | +AICPU | +CCU 硬件卸载 |
| Sector 大小 | 512 Byte | - | 128 Byte |
| 编程模式 | SIMD | SIMD | SIMD+SIMT 混合 |
| 低精度格式 | FP16/BF16 | - | HiF8/MXFP8/FP8/MXFP4/FP4 |

## 二、A5 芯片关键升级

### 2.1 AI Core 升级
- **Vector 算力配比翻倍**：1:16 → 1:8
- **SIMD-SIMT 混合编程**模式
- **Cube-Vector 融合通路**：MoE 算子融合效率更高
- **Cube 算力比**：FP16 / FP8 / FP4 = 1:2:4
- **指令双发**：2 份硬件单元可并行执行计算指令
- **RegBase**：基于寄存器的编程模型，通过 MicroAPI 直接操作寄存器
- **NDDMA**：多维数据搬运，支持搬运时 padding / transpose / broadcast / slice

### 2.2 内存升级
- **950PR**：大容量 Bailu 内存（128GB）
- **950DT**：高带宽 Zhuque/HBM 内存（4TB/s）
- **128 Byte Sector**（A2 为 512 Byte），提升离散小内存访问效率
- L2→UB 带宽维持 5~6TB/s

### 2.3 通信升级
- **CCU（集合通信加速器）**：通信卸载，硬件加速
- **UB 总线**：新协议，支持 1K 规模内存语义
- **双轴 X-Y 拓扑**：带宽并发
- **1825 DPU 网卡**：数控分离
- **新增 UB 通路**：L1 与 UB、FixPipe 到 UB 的通路，减少 CV 融合算子搬运

## 三、A5 机型规格

### 3.1 Atlas 350 标卡（950PR）

| 属性 | 规格 |
|------|------|
| 尺寸 | 扩高(137mm) × 全长(312mm) × 双宽(39.04mm) |
| 算力 | 425T @ FP16 |
| 精度 | FP32 / HF32 / FP16 / BF16 / FP8 / FP4 / INT8 |
| 显存 | 112GB，带宽 1.4TB/s |
| PCIe | PCIe 5.0 x16，双向 128GB/s |
| 卡间互联 | 4P 卡：3×UB Port (P4-P6)，双向 300GB/s；2P 卡：4×UB Port (P4-P8)，双向 400GB/s |
| 功耗 | ≤600W，被动散热 |
| 上市 | 2026Q2 |

### 3.2 G8550 风冷机型（16 卡）

| 属性 | 规格 |
|------|------|
| 形态 | 10U 风冷 AI 服务器 |
| CPU | Turin 9A75 × 2 |
| 内存 | DDR5 64G-6400MHz × 24 |
| 存储 | 系统盘 / 数据盘：4×NVMe |
| NPU | 16 × Atlas 350（950PR 风冷） |
| Switch 出网卡 | 4 × NIC（云脉网卡） |
| CPU 出网卡 | 裸金属 BF3 + lan switch（B3220+0302Y728） |
| 管理 | BMC 出 GE |
| 拓扑 | 双上行 + 环形拓扑 |
| 工作温度 | 35℃ @ 1500m |

### 3.3 UBX 液冷高密机型（16 卡）

| 属性 | 规格 |
|------|------|
| 形态 | 10U 液冷，搭配 19 寸液冷柜 |
| NPU | 16 × HIAC 高密卡（950PR），PCIe 5.0 × 8 |
| 拓扑 | 16 卡全互连（400GB/s），4×4 卡 fullmesh（300GB/s） |
| Scale-Out | 4 × NPU 共用 1 × 1825，每 NPU 200Gb/s |
| Host CPU | 2 × AMD 9A75 |
| 内存 | 24 × DDR5（64GB/96GB，6400MHz） |
| IO 扩展 | 1× PCIe 5.0 X16 DPU，4× PCIe 5.0 X4 NVMe SSD |
| NPU 端口 | PCIe 5.0 X8（2 个 UB 端口）；UB 端口：4 个连 UB SW，3 个组成 4 卡 fullmesh |

### 3.4 大禹 950DT 定制机型（64 卡）

| 属性 | 规格 |
|------|------|
| NPU | 64 × 950DT 液冷模组 |
| 算力 | 31.1P @ FP16，显存 6.2TB |
| Scale-Up | 64 模组全互联，带宽 1.2TB/s |
| Scale-Out | 400Gb/s RoCE |
| Host CPU | 16 × X86 CPU (AMD 9A75) |
| Host 内存 | 192 × DDR5（64GB/96GB，6400MHz） |
| IO 扩展 | 16 × DPU 卡，32 × NVMe SSD |
| 组成 | 16 个计算节点（各含 X86+1DPU+2NVMe+4NPU）+ 6 个 2OU UB 总线节点（64 卡单柜）；128 卡双柜方案为每柜 12 个 1OU 总线节点，详见 [[architecture/dayu-supernode-topology]] |
| NPU 端口 | PCIe 5.0 X16（4 个 UB 端口）；UB：12 个 UB 端口连接总线节点（CLOS 组网，D-D 带宽 1.2TB/s） |
| 扩展 | 可扩展支持 K 级超节点 |

> 2026-11-17 更新：依据《950DT大禹配置-64卡与128卡方案》（proposal 快照），64 卡单柜为 **6 个总线节点**；本页早期版本的"12 个 UB 总线节点"对应 128 卡方案的单柜形态。两方案计算节点配置相同。

### 3.5 A3 384 超节点（对比参考）

| 属性 | 规格 |
|------|------|
| 形态 | 12 × 47U 计算柜 + 4 × 47U 总线设备柜 |
| NPU | 384 × 昇腾 910C |
| AI 算力 | FP16 307.2 PFLOPS，FP32 76.8 PFLOPS |
| 内存 | 48T HBM，最大 3072 个 DDR 插槽 |
| 交换 | 336 × 灵衢交换芯片 + 56 × 灵衢总线设备 |
| 散热 | 液冷(计算柜) + 风冷(总线设备柜) |
| 功耗 | ~716KW（计算柜 54KW/柜，网络柜 20KW/柜） |

## 四、硬件子系统参数

### 4.1 RegBase / SIMD / SIMT 硬件参数

| 规格名称 | Reg Based SIMD | SIMT | H100 参考 |
|---------|----------------|------|---------|
| 算力 | 128 FP32/cycle | 128 FP32/cycle | 128 FP32/cycle |
| 指令发射 | 2ld+2exe+1scalar+1st | 1/cycle/sched | 1/cycle/sched |
| UB 带宽 | 512R/256W | 128R/W | 128R/W |
| UB latency | ~10 cycle | ~20 cycle | 20~30 cycle |
| GM 直接访问 | No | Yes | Yes |
| HardwareLoop | Yes | No | No |
| Register 容量 | 18.5KB | 128KB | 256KB |
| Instruction size | 32bit | 64~128bit | 128bit |
| Icache size | 8KB | 8KB | 128KB L1+128KB L0 |
| L2→UB 带宽 | 5~6TB/s | - | - |

### 4.2 SIMD vs SIMT 对比

| 特性 | SIMD | SIMT |
|------|------|------|
| 编程范式 | 矢量编程（需感知硬件并行度） | 标量编程（无需感知硬件并行度） |
| 数据要求 | 连续、对齐 | 每个线程独立控制流 |
| GM 访问 | Scalar 支持直接访问 GM；Vector 不支持 | 支持直接访问 GM，有核内 DCache |
| 指令发射 | 同时发射 ld/exe/st | 每个 Cycle 发射单指令 |
| Warp Reduce 指令 VL | - | 是 SIMD 的一半 |
| UB 访存效率 | R512B/s + W256B/s | R/W 128B/s |
| 擅长场景 | 连续规整计算 | 随机访存、离散访问、小包搬运 |

### 4.3 单卡端口分配（A5 HCCL 双轴拓扑）

- **Adie**：P0 出框，P1-P2 H2D
- **Bdie**：P0 X 轴，P1 出框，P2-P8 X 轴 FM（fullmesh）

## 五、A5 低精度算力

| 格式 | 算力 |
|------|------|
| FP8 / MXFP8 / HiF8 | 1 PFLOPS |
| MXFP4 | 2 PFLOPS |

## 参考文档

- A2&A3&A5 硬件介绍_v1.0.pptx
- A5 算子新特性分享.pptx（作者：黄源）