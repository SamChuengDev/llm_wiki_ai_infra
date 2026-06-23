---
title: "性能调优 (Tuning)"
tags: [category/tuning]
last_updated: 2026-06-17
source: []
confidence: high
status: active
open_questions: []
---

# 03 性能调优 (Tuning)

本章节记录 HCCL 通信、显存优化及算子融合等性能调优实践。

## 📖 核心文档
- [ ] [[hccl_optimization|HCCL 通信优化]]
- [ ] [[memory_management|显存复用策略]]
- [x] [[vlm_23b_rl_torch_infer_vs_xg_performance|Tuning: VLM 23B RL torch infer 与 XG 推理后端性能对比及 ep128 if144_b8/b12、kv80 与 ViT 透传调参经验]]
- [ ] [[xpu_gpt_w8a8_rl_scheduler_memory_tuning|Tuning: xpu-gpt W8A8 RL scheduler batch 与显存水位调优]]
- [ ] [[ep_zero_sync|Tuning: EP 零冗余权重同步与 Resharding 优化 (ZeroSync)]]
