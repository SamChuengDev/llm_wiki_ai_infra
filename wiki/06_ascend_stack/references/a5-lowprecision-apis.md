---
title: A5 低精度算子 API 与性能数据
tags:
  - A5
  - low-precision
  - API
  - operator
  - performance
  - AMCT
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# A5 低精度算子 API 与性能数据

## 一、完整低精度算子支持矩阵（50+ 算子）

### 1.1 Quant 量化类算子（7 个）

| 算子 | FP8 | MXFP8 | HiFP8 | MXFP4 | 备注 |
|------|-----|-------|-------|-------|------|
| DynamicMxQuant | - | ✓ | - | ✓ | MX 量化核心算子（单轴 1×32） |
| DynamicMxQuantWithDualAxis | - | ✓ | - | ✓ | 双轴 MX 量化（训练场景 32×32） |
| GroupedDynamicMxQuant | - | ✓ | - | ✓ | 分组 MX 量化（GMM 专用） |
| DynamicBlockMxQuant | - | ✓ | - | ✓ | Block 粒度 MX（训练 + kv 量化） |
| DynamicDualLevelMxQuant | - | - | - | ✓ | 二级量化（提升 MXFP4 精度） |
| DynamicQuant | ✓ | - | ✓ | - | 普通动态量化（HiF8/FP8） |
| AscendAntiQuant | ✓ | - | ✓ | - | 反量化（一般融合调用） |

### 1.2 Matmul 类算子（7 个）

| 算子 | FP8 | MXFP8 | HiFP8 | MXFP4 | 备注 |
|------|-----|-------|-------|-------|------|
| QuantBatchMatmulV4 | ✓ | ✓ | ✓ | ✓ | 低 bit matmul 全量化计算（线性层通用） |
| WeightQuantBatchMatmulV2 | ✓ | ✓ | ✓ | ✓ | 权重量化 |
| TransposeQuantBatchMatMul | ✓ | ✓ | — | — | Transpose + qbmm 融合 |
| QuantBatchMatmulInplaceAdd | ✓ | ✓ | ✓ | — | 训练场景梯度更新 in-place add |
| DualLevelQuantBatchMatmul | — | — | — | ✓ | MXFP4 二级量化 + matmul 融合（精度补偿） |
| RotateQuant | — | — | — | ✓ | Hadamard 旋转 + 量化融合（精度补偿） |
| FlatQuant | — | — | — | ✓ | FlatQuant + 量化融合 |

### 1.3 GMM 分组矩阵乘（MoE 核心，5 个）

| 算子 | FP8 | MXFP8 | HiFP8 | MXFP4 | 备注 |
|------|-----|-------|-------|-------|------|
| GroupedMatmul | ✓ | ✓ | ✓ | ✓ | MoE 核心专家计算 |
| GroupedMatmulActivationQuant | — | ✓ | — | — | 仅支持 gelu+quant |
| GroupedMatmulSwigluQuant | — | ✓ | ✓ | ✓ | MoE 典型 SwiGLU+Quant 融合 |
| GroupedMatmulFinalizeRouting | — | ✓ | ✓ | ✓ | MoE 定制路由+计算融合 |
| QuantGroupedMatmulInplaceAdd | ✓ | — | ✓ | — | MoE 训练梯度更新 in-place add |

> **注意**：低精 GMM 场景仅支持"单单单"（x/weight/scale 均为单个 dtype），weight 为 list。

### 1.4 Norm/Activation 融合类算子

**Norm 类**：

| 算子 | FP8 | HiFP8 | MXFP4 | 说明 |
|------|-----|-------|-------|------|
| RmsNormQuant | ✓ | ✓ | — | RMSNorm + 量化融合 |
| AddRmsNormQuant | ✓ | ✓ | — | Add + RMSNorm + 量化融合 |
| AddRmsNormDynamicQuant | ✓ | ✓ | — | Add + RMSNorm + 动态量化 |
| RmsNormDynamicMxQuant | ✓(mxfp8) | — | ✓ | RMSNorm + MX 量化 |
| AddRmsNormDynamicMxQuant | ✓(mxfp8) | — | ✓ | Add + RMSNorm + MX 量化融合 |

**Activation 类**：

| 算子 | FP8 | MXFP8 | HiFP8 | MXFP4 |
|------|-----|-------|-------|-------|
| GeluQuant | ✓ | - | ✓ | - |
| SwigluMxQuant | - | ✓ | - | ✓ |

### 1.5 Attention 类算子（5 个）

| 算子 | FP8 | MXFP8 | HiFP8 | MXFP4 | 备注 |
|------|-----|-------|-------|-------|------|
| FusedInferAttentionScore | ✓ | ✓ | ✓ | 仅伪量化 | 推理 GQA/MQA/MHA 全量化 |
| FlashAttentionScore | ✓ | — | ✓ | — | 训练场景 GQA/MQA/MHA |
| MLA Prolog V3 | ✓ | ✓ | ✓ | — | MLA（MLA Attention）前处理 |
| QuantLightning Indexer | ✓ | — | ✓ | — | Sparse attention 特定算子 |
| MixedQuant Sparse Flash MLA | ✓(仅伪量化) | — | — | — | KV 支持 FP8 伪量化（DSv4 专用） |

### 1.6 通信融合类（MC2）

参见下方 MC2 部分。

## 二、低精度算子 API（PyTorch 接口）

### 2.1 量化算子

```python
# ===== MXFP8/MXFP4 量化 =====

# MX 动态量化（单轴 1×32）
data, scale = torch_npu.npu_dynamic_mx_quant(
    input,
    axis=-1,                          # 量化轴（K 维）
    dst_type=torch.float8_e4m3fn,     # 目标: float8_e4m3fn / float8_e5m2 / float4_e2m1
    scale_alg=0                       # 0=OCP 标准(默认), 1=cuBLAS 兼容
)

# MX 双轴量化（训练场景 32×32，weight 前后向复用）
data, scale_row, scale_col = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
    input, dst_type=torch.float8_e4m3fn, ...
)

# MX 分组量化（GMM 专家量化）
data_list, scale_list = torch_npu.npu_grouped_dynamic_mx_quant(
    input_list, ...                   # list[tensor]，每个 expert 一组
)

# Block 粒度 MX 量化（训练 + KV cache 量化）
data, scale = torch_npu.npu_dynamic_block_mx_quant(input, block_size=32, ...)

# MXFP4 二级量化（精度补偿 PTS）
data, scale1, scale2 = torch_npu.npu_dynamic_dual_level_mx_quant(input, ...)

# ===== FP8/HiF8 量化 =====

# 动态量化（FP8/HiF8 通用）
data, scale = torch_npu.npu_dynamic_quant(
    input,
    dst_type=torch.hifloat8,          # 目标: torch.float8_e4m3fn / torch.hifloat8
    dst_type_max=0,                   # HiF8 专用: 0(自动)/15/56/224/32768
    quant_mode="pertensor"            # pertensor(默认)/pertoken/perblock
)

# 静态量化（支持 FP8/HiFloat8 delayed scaling 策略）
torch_npu.npu_quantize(input, scale, zero_point, dtype, ...)

# 反量化
output = torch_npu.npu_ascend_anti_quant(data, scale, dst_type=torch.bfloat16, ...)
```

### 2.2 计算算子

```python
# 低精度 Matmul（QuantBatchMatmulV4）
output = torch_npu.npu_quant_matmul(
    x, weight, scale,
    # scale: 单个 scale(tensor) 或元组 (scale_x, scale_w) 或 list
)

# 低精度 Grouped Matmul（MoE 专家计算）
output_list = torch_npu.npu_grouped_matmul(
    x_list, weight_list, scale_list,
    group_list=...,                   # 每个 expert 的 token 数量
    ...
)
```

### 2.3 通算融合算子（MC2，计算+通信低 bit 闭环）

```python
# ReduceScatter + Matmul 融合
output = torch_npu.npu_quant_mm_reduce_scatter(x, weight, scale, op="sum", group=...)

# Matmul + AllGather 融合
output = torch_npu.npu_all_gather_quant_mm(x, weight, scale, group=...)

# Matmul + AllReduce 融合（算子支持，模型侧未完全适配）
output = torch_npu.npu_mm_all_reduce_base(x, weight, scale, group=..., reduce_op="sum")

# GMM + AlltoAllv 融合（算子支持，模型侧未完全适配）
output = torch_npu.npu_quant_gmm_alltoallv(x_list, weight_list, ...)
output = torch_npu.npu_alltoallv_quant_gmm(x_list, weight_list, ...)
```

### 2.4 Norm/Activation 融合算子

```python
# FP8/HiF8
out, norm_out = torch_npu.npu_rms_norm_quant(x, gamma, quant_scale, dst_type=...)
out, norm_out = torch_npu.npu_add_rms_norm_quant(x, residual, gamma, ...)
out, norm_out = torch_npu.npu_add_rms_norm_dynamic_quant(x, residual, gamma, ...)

# MXFP8/MXFP4
out, scale = torch_npu.npu_rms_norm_dynamic_mx_quant(x, gamma, ...)
out, scale = torch_npu.npu_add_rms_norm_dynamic_mx_quant(x, residual, gamma, ...)

# 激活+量化
out, scale = torch_npu.npu_gelu_quant(x, scale, dst_type=...)
out, scale = torch_npu.npu_swiglu_mx_quant(x, gate, ...)
```

### 2.5 Add 融合类

```python
# Add + QuantMatmul in-place 融合
torch_npu.npu_add_quant_matmul_(bias, x, weight, scale, ...)

# Add + QuantGMM in-place 融合
torch_npu.npu_add_quant_gmm_(bias, x_list, weight_list, scale_list, ...)
```

## 三、实测性能数据

| 算子分类 | 量化模式 | 目标加速比(vs BF16) | >40us 大算子平均 | 整体平均 |
|---------|---------|-------------------|----------------|---------|
| **MM（线性层）** | FP8-PerToken | 1.6x | 1.747x | **1.62x** |
| MM | FP8-PerTile | 1.2x | 1.330x | 1.25x |
| **MM** | **MXFP8 MxA8W8** | 1.6x | **1.815x** | **1.61x** |
| MM | MXFP4 MxA4W4 | 3.0x | 3.161x | 2.55x |
| **GMM（MoE 专家）** | FP8-PerToken | 1.6x | 1.62x | 1.47x |
| GMM | FP8-PerTile | 1.2x | 1.54x | 1.42x |
| **GMM** | **MXFP8 MxA8W8** | 1.6x | **1.80x** | **1.64x** |
| GMM | MXFP4 MxA4W4 | 3.0x | 3.19x | 2.71x |
| IFA（推理 FlashAttn） | MXFP8 | 1.6x | 1.20x | 1.20x |
| **PFA（训练 FlashAttn）** | **MXFP8** | 1.6x | **1.62x** | **1.62x** |

**关键结论**：
- MXFP8 MxA8W8（MM/GMM）整体平均加速比稳定在 ~1.6x
- MXFP4 MxA4W4 加速比约 2.5-2.7x（受限于访存/scale 读取开销）
- 大算子（>40us）加速比明显高于小算子（访存 bound）
- 推理 FA（IFA）加速比偏低（1.2x），训练 FA（PFA）达 1.6x

## 四、DeepSeek W4A8 方案

| 方案 | Weight 量化路径 | 计算方式 | 说明 |
|------|--------------|---------|------|
| DeepSeekv4 标准（MoE W4A8） | bf16 → 1×32 MXFP4 → 128×128 MXFP8 | E8M0 1×128 MXFP8 input @ 128×128 MXFP8 weight | 前反向用相同 weight tensor |
| A5A6-MXFP4 | bf16 → 1×32 MXFP4 → 32×32 MXFP8 | 1×32 MXFP8 input @ 32×32 MXFP8 weight | 当前 quant_matmul 支持的粒度 |
| A6-HiF4 | bf16 → 1×64 HiF4（推理）/ 1×32 MXFP8（反向） | 1×32 MXFP8 input @ 1×64 HiF4 weight | 推理用 HiF4，反量化回 MXFP8 |

**反向梯度计算**：保存前向 weight → E8M0 1×128/1×32 MXFP8 grad @ 对应 weight。

## 五、AMCT 量化工具（PTQ 流程）

### 5.1 AMCT 定位

AMCT（Ascend Model Compression Toolkit）提供统一 PTQ 量化工作流：
**Adapt（模型适配）→ Evaluate（精度评估）→ Quantize（PTQ 优化）→ Deploy（部署导出）**

解决前期量化痛点：
1. 每个模型一个权重转换脚本 → 标准化流程
2. 精度有损，每个模型重复适配算法 → 算法注册式组合
3. 不同算法效果无统一评估 → 统一评估标准

### 5.2 AMCT 能力概览

- **20+ 网络支持**：DeepSeek v3.2/v4、Qwen3/Qwen3-Next、GLM 5.1/5.2、Longcat Flash/Next 等
- **4 种低比特格式**：FP8、MXFP4/8、HiF8
- **10+ 量化算法**：LAC、LWC、AutoRound、SVDQuant、GPTQ、Hadamard、Flatquant、OmniQuant、SmoothQuant
- **多种量化粒度**：per-token / per-group(32/64/128) / per-block / per-tensor / per-channel / MX

### 5.3 精度补偿算法分类

| 适用对象 | 算法 | 用途 |
|---------|------|------|
| **激活** | LAC | Learned Activation Clipping，8bit 轻微损失补偿 |
| **权重** | LWC → AutoRound → GPTQ → SVDQuant | 权重 outlier 优化，精度依次提升 |
| **激活&权重（结构变换）** | Hadamard(可学习)、FlatQuant、OmniQuant、SmoothQuant | 块内分布均衡化/舍入误差优化 |

> **经验**：对 A4W4 精度损失超过 1% 的网络，基本都需要使用 rotation 类算法（Hadamard/FlatQuant）。

### 5.4 量化实践效果示例

**Qwen3-235B**：

| 方案 | AIME-24 | GPQA | GSM8K | MATH500 |
|------|---------|------|-------|---------|
| BF16 | 83.33 | 72.22 | 96.44 | 87.0 |
| MoE-W4A4 + Attn-w4a4 + Hadamard(可学习) | 83.33 | 71.21 | 97.12 | 86.6 |

**DeepSeek-R1**：

| 方案 | WikeText2(PPL↓) | MMLU | GPQA-Diamond | MATH-500 | AIME-2024 |
|------|----------------|------|-------------|---------|----------|
| BF16 | 3.33 | 90.8 | 72.73 | 97.6 | 80 |
| MoE-W4A4 直转（无补偿） | 3.95 | 90.4 | 65.85 | 96.4 | 80 |
| MoE-W4A4+LWC+LAC+Hadamard | 3.55 | 90.7 | 70.7 | 97 | 83.3 |

## 六、性能分析与调试

### 6.1 算子替换范围

- **可替换**：大部分 linear 层、Attention 中的 QKV/Out 投影、MoE 专家层
- **不可替换**：Topk 路由部分、LM head 层（保持高精度 bf16）、第一层/Embedding 层视情况保留

### 6.2 重计算注意事项

| 格式 | 前向量化 | 重计算量化 | 注意事项 |
|------|---------|----------|---------|
| MXFP8 | 单轴 1×32 | 双轴 32×32 | 重计算需重新做双轴量化，影响性能 |
| FP8 | E4M3 per-token | E5M2 per-token | 格式不同需重新量化 |
| **HiF8** | per-tensor | per-tensor | **无此问题，前后向量化一致直接复用** |

### 6.3 性能分析工具链

| 工具 | 分析内容 | 关注点 |
|------|---------|-------|
| **Profiling 堆栈** | linear 层是否为 TELinear、是否走低精算子分支 | **算子替换比例是新模型适配第一关键指标** |
| **op_statistic.csv** | 各低精算子加速比、各卡耗时差异 | 加速比是否达标（MxA8W8 ~1.6x）、负载是否均衡 |
| **kernel_details** | 具体算子的 tiling/分核/泛化性问题 | 大算子性能是否正常，是否有 split 过小 |
| **性能看板** | http://board.ascend.huawei.com/pages/specs.html | 算子理论峰值对比 |

### 6.4 性能分析 Checklist

1. **Profiling**：打印堆栈，确认 linear 层替换比例
2. **op_statistic**：分析低精算子加速比是否达标（MxA8W8 ~1.6x，MxA4W4 ~2.5x）
3. **kernel_details**：分析具体算子是否有 tiling/泛化性问题
4. **重计算检查**：MXFP8 场景确认双轴量化是否正确配置
5. **MC2 融合**：确认通信算子是否使用 MC2 融合版本
6. **访存 bound 排查**：小算子加速比低通常是访存 bound，尝试算子融合、增大 batch
7. **各卡差异**：op_statistic 中各卡同一算子耗时差异 >10% → 负载不均问题

### 6.5 常见问题排查

| 现象 | 可能原因 | 排查方向 |
|------|---------|---------|
| 低精算子加速比 <1.2x | 访存 bound / 算子过小 / 融合缺失 | 增大 batch、启用 NormQuant 融合、检查 MC2 |
| 训练 loss 不收敛/NaN | 梯度溢出 / 量化精度不够 | 切换 dst_type_max 更大档位、LM head 保留 bf16 |
| 精度下降 >1% | outlier 过多 / 4bit 无补偿 | 加 SmoothQuant、MXFP4 加 Hadamard/FlatQuant |
| 各卡速度差异大 | 通信不均衡 / 负载不均 | 参考 A5 通信配置 |
| MXFP4 精度严重下降 | 未用精度补偿 | 必须加 rotation 类算法（Hadamard/FlatQuant） |
| HiF8 性能不如 MXFP8 | 正常（小算子场景） | 计算 bound 大 batch 切 MXFP8，精度优先用 HiF8 |

## 参考文档

- Ascend950 低 bit 能力汇总（外发）.pptx（作者：刘波、李艇，2026.07.05）
- CANN 社区：https://gitcode.com/cann
- OCP MicroScaling Formats 标准：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- TransformerEngineNPU / MindSpeed / AMCT 相关文档