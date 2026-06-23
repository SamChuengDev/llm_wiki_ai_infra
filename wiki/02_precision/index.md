---
title: "精度对齐 (Precision)"
tags: [category/precision]
last_updated: 2026-04-11
source: []
confidence: high
status: active
open_questions: []
---

# 02 精度对齐 (Precision)

本章节汇总在模型迁移过程中的精度溢出、NaN 排查及 Loss 对齐经验。

## 📖 核心文档
- [[vlm_23b_xg_logpdiff_spike_replay_handoff|VLM 23B XG logprobs_diff spike replay 定位现场]]
- [[vlm_torch_infer_response_vit_embeds_unified_cache|torch infer VLM rollout response 直传 vit_embeds 统一 ViT cache 方案]]
- [[vlm_torch_infer_frozen_vlm_weight_sync_reward|torch infer frozen VLM 权重未同步导致 reward 低于合并前]]
- [[vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移对齐]]
- [[vlm_rl_actor_blind_diff|VLM RL Actor 概率不一致 (The Missing -100)]]
- [ ] [[overflow_handling|FP16/BF16 溢出处理]]
- [ ] [[alignment_tools|精度对齐工具集]]
