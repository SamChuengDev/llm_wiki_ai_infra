---
title: "常规排错 (Bugfix)"
tags: [category/bugfix]
last_updated: 2026-06-02
source: []
confidence: high
status: active
open_questions: []
---

# 00 常规排错 (Bugfix)

本章节记录日常开发中的非特定领域报错（如环境配置问题、框架启动异常、通用业务代码崩溃）的复盘与防坑指南。

## 📖 核心文档

> 暂无已沉淀文档，待 Ingest 时补充。

- [[wiki/00_bugfix/xpu_gpt_decode_graph_cache_maxlen_mismatch|xpu_gpt decode graph cache 未按 max_length 隔离导致 12k->32k 图模式断言]]
- [[wiki/00_bugfix/vlm_template_reuse_rollout_vit_embeds|VLM RL Template 未复用 rollout vit_embeds 导致 Actor 侧重复图片编码]]
- [[wiki/00_bugfix/vlm_torch_infer_scheduler_batch_env_override|torch infer VLM 默认 env 覆盖 scheduler/model batch 导致 RL step 劣化]]
- [[wiki/00_bugfix/vlm_rl_xg_hash_image_size_check|VLM 2B5 xg RL hash 路径未关闭 xLLM 图片尺寸检查导致 InvalidImageURL]]
- [[wiki/00_bugfix/vlm_2b5_vit_checkpoint_fallback|M8 2B5 VLM Companion ViT 权重路径 fallback 缺失导致启动失败]]
- [[wiki/00_bugfix/vlm_torch_infer_vit_cache_sp_slice_logpdiff|torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸]]
- [[wiki/00_bugfix/encoder_cache_eviction_assertion|xLLM 并发压测下 EncoderResultCache 驱逐触发 AssertionError]]
- [[wiki/00_bugfix/npu_empty_virt_addr_cache_oom|empty_virt_addr_cache 导致 NPU 物理显存未归还引发 OOM]]
- [[wiki/00_bugfix/ep16_decode_deadlock_pd_count_fix|EP16 MoE Decode Deadlock pd_count 修复]]
- [[wiki/00_bugfix/companion_vit_init_deadlock_and_encode_rejected|Companion ViT 死锁 + EncodeRejected (Unix Socket URI 格式错误)]]
