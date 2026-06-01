---
title: "常规排错 (Bugfix)"
tags: [category/bugfix]
last_updated: 2026-06-01
source: []
confidence: high
status: active
---

# 00 常规排错 (Bugfix)

本章节记录日常开发中的非特定领域报错（如环境配置问题、框架启动异常、通用业务代码崩溃）的复盘与防坑指南。

## 📖 核心文档

> 暂无已沉淀文档，待 Ingest 时补充。

- [[wiki/00_bugfix/xpu_gpt_decode_graph_cache_maxlen_mismatch|xpu_gpt decode graph cache 未按 max_length 隔离导致 12k->32k 图模式断言]]
- [[wiki/00_bugfix/vlm_torch_infer_vit_cache_sp_slice_logpdiff|torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸]]
- [[wiki/00_bugfix/companion_vit_init_deadlock_and_encode_rejected|Companion ViT 死锁 + EncodeRejected (Unix Socket URI 格式错误)]]
