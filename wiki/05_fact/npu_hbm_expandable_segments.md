---
title: "Fact: NPU HBM 虚拟显存 expandable_segments 特性"
tags: [status/stable, category/fact, npu, hbm, oom, memory]
last_updated: 2026-06-04
source:
  - "用户会话: NPU HBM 虚拟显存特性作用及用法, 2026-06-04"
  - "runtime_env_xg.yaml"
  - "run_xg_vlm_2b5.sh"
  - "run_xg_vlm_2b5_before_merge.sh"
  - "outputs/precision_cmp/torch_infer_23b_128p_baseline_0510_212035.log"
confidence: high
status: active
open_questions: []
contradictions: []
---

# Fact: NPU HBM 虚拟显存 expandable_segments 特性

## 1. 核心事实 (Core Fact)

`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 是 NPU PyTorch allocator 的虚拟显存 / HBM 管理特性开关。开启后通常可以降低 NPU HBM 显存使用和碎片化压力，一定程度缓解 NPU device OOM。

在 VLM RL xpu_gpt 与 23B torch inference 验证口径中，该开关应视为内存稳定性相关配置，而不是无关环境噪声。对齐合并前后实验时，如果 baseline worker 环境中存在该变量，合并后脚本也应保持开启，除非有明确实测证据要求关闭。

## 2. 设计背景 (Background & Rationale)

VLM RL 训练和 rollout 混合部署时，Actor、Ref、Rollout / xLLM worker 会在同一组 NPU 上经历权重同步、KV cache、MoE、ViT embedding、optimizer step 等高峰 HBM 分配。`expandable_segments:True` 通过 PyTorch NPU allocator 的扩展段 / 虚拟显存机制缓解大块分配与碎片化问题，因此常用于降低 device OOM 风险。

23B torch inference baseline 日志 `outputs/precision_cmp/torch_infer_23b_128p_baseline_0510_212035.log` 的 worker 环境中可见：

```text
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

当前 xpu_gpt 相关脚本也已开启该开关。

## 3. 应用场景与约束 (Constraints)

- 对 NPU VLM RL 长跑、23B torch inference、xpu_gpt W8A8 rollout 等 HBM 压力较高的任务，默认保留 `expandable_segments:True`。
- 对齐 baseline 精度 / 性能时，该变量属于需要纳入环境差异检查的内存配置。
- 该开关只能缓解 HBM 分配和碎片化压力，不能证明 OOM 已修复，也不能替代长跑验证。
- `expandable_segments:True` 不等价于释放跨进程物理 HBM；如果需要把当前进程缓存的物理显存归还给驱动，仍需在合适位置调用 `torch.npu.empty_cache()`。
- 若某条路径因兼容性或实测问题需要关闭该特性，必须在脚本或验证记录中说明原因，不能把关闭动作当作默认调参。

## 4. 相关配置或引用 (References)

```bash
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

```yaml
# runtime_env_xg.yaml
"PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"
```

```bash
# run_torch_infer_vlm_23b.sh
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
```

## 5. 沉淀与关联

- 关联知识点: [[wiki/00_bugfix/npu_empty_virt_addr_cache_oom|Bugfix: empty_virt_addr_cache 导致 NPU 物理显存泄露引发 OOM]]
- 关联知识点: [[wiki/05_fact/torch_infer_merge_debug_checklist|Fact: xpu_gpt + torch infer 合并验证排查清单]]
