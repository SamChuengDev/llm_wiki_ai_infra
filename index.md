# Map of Content (知识入口)

欢迎来到 LLM-WIKI。这里是异构计算 Infra 知识的索引核心。
LLM 在执行 Query 时应首先读取本文件定位相关页面。

---

## 🏛️ 知识目录

### 0. [[wiki/00_bugfix/index|常规排错 (Bugfix)]]
- [[wiki/00_bugfix/xpu_gpt_vlm_2b5_dual_node_a3_rollout_adaptation|Bugfix: xpu-gpt 2B5 VLM 双节点 A3 rollout 适配]]
- [[wiki/00_bugfix/vlm_torch_infer_scheduler_batch_env_override|Bugfix: torch infer VLM 默认 env 覆盖 scheduler/model batch 导致 RL step 劣化]]
- [[wiki/00_bugfix/vlm_template_reuse_rollout_vit_embeds|Bugfix: VLM RL Template 未复用 rollout vit_embeds 导致 Actor 侧重复图片编码]]
- [[wiki/00_bugfix/xpu_gpt_decode_graph_cache_maxlen_mismatch|Bugfix: xpu_gpt decode graph cache 未按 max_length 隔离导致 12k->32k 图模式断言]]
- [[wiki/00_bugfix/vlm_rl_xg_hash_image_size_check|Bugfix: VLM 2B5 xg RL hash 路径未关闭 xLLM 图片尺寸检查导致 InvalidImageURL]]
- [[wiki/00_bugfix/vlm_2b5_vit_checkpoint_fallback|Bugfix: M8 2B5 VLM Companion ViT 权重路径 fallback 缺失导致启动失败]]
- [[wiki/00_bugfix/vlm_cache_hash_mismatch|Bugfix: VLM Cache Miss Due to Cross-Boundary Hash Mismatch]]
- [[wiki/00_bugfix/vit_cache_miss_cityhash|Bugfix: ViT Cache 100% Miss & Race Condition Mismatch]]
- [[wiki/00_bugfix/vlm_torch_infer_vit_cache_sp_slice_logpdiff|Bugfix: torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸]]
- [[wiki/00_bugfix/encoder_cache_eviction_assertion|Bugfix: xLLM 并发压测下 EncoderResultCache 驱逐触发 AssertionError]]
- [[wiki/00_bugfix/ep16_decode_deadlock_pd_count_fix|Bugfix: EP16 MoE Decode Deadlock pd_count 修复]]
- [[wiki/00_bugfix/vlm_thinking_parser_logpdiff_spike|Bugfix: VLM RL 训练中 thinking_parser 缺失导致 logpdiff 极端 Spike (0.3+)]]
- [[wiki/00_bugfix/host_oom_glibc_arena|Bugfix: 宿主机 OOM 与 SYS 物理内存持续增长 (glibc malloc arena 碎片化)]]
- [[wiki/00_bugfix/pytorch_to_empty_libc_leak_and_precision|Bugfix: PyTorch to_empty() 精度丢失与 libc 碎片化泄漏连环坑]]
- [[wiki/00_bugfix/npu_empty_virt_addr_cache_oom|Bugfix: empty_virt_addr_cache 导致 NPU 物理显存泄露引发 OOM]]
- [[wiki/00_bugfix/oom_moe_dispatch|Bugfix: OOM 导致 MoeDistributeDispatchV2 算子死锁与集群崩溃]]
- [[wiki/00_bugfix/vlm_companion_vit_rpc_crash|Bugfix: VLM Companion ViT RPC 连接失败与环境变量断层]]
- [[wiki/00_bugfix/cann_aclnngather_zero_length_crash|CANN aclnnGather 算子零长维报错与规避]]
- [[wiki/00_bugfix/vlm_grpc_companion_routing_deadlock|VLM Companion Encoder gRPC 路由死锁与地址错配]]
- [[wiki/00_bugfix/veturborpc_env_override|强行禁用 veturborpc_ext 及 xLLM 环境映射机制]]
- 日常环境异常与框架启动报错
- 业务逻辑层的 Bug 修复复盘与 Workaround

### 1. [[wiki/01_operators/index|算子迁移 (Operators)]]
- GPU (CUDA) -> NPU (CANN) 算子映射表
- 算子规避与自定义算子实现
- `torch_npu` 特有算子库

### 2. [[wiki/02_precision/index|精度对齐 (Precision)]]

**📚 诊断方法论与工具向导 (Methodology & Tools)**
- [[wiki/02_precision/rl_precision_metrics_guide|RL 核心监控指标解析]]
- [[wiki/02_precision/rl_alignment_checklist|RL 精度对齐排查前置清单 (Golden Checklist)]]
- [[wiki/02_precision/rl_troubleshooting_workflow|五阶段大盘定位排查主流程 (Workflow)]]
- [[wiki/02_precision/rl_off_policy_inconsistency|训推架构 Off-Policy 异构发散诱因与对策]]
- [[wiki/02_precision/rl_debug_tools_guide|NPU 框架级调试工具与探针指南 (Tools)]]

**🐛 实战排查案例 (Case Studies)**
- [[wiki/02_precision/vlm_torch_infer_frozen_vlm_weight_sync_reward|torch infer frozen VLM 权重未同步导致 reward 低于合并前]]
- [[wiki/02_precision/tokenizer_config_mismatch|tokenizer_config 未对齐导致 response 异常]]
- [[wiki/02_precision/rmsnorm_eps_mismatch_reward|RMSNorm 超参 eps 未对齐导致 reward 下降]]
- [[wiki/02_precision/verl_randomness_fixing|Ray 多进程下 VeRL 确定性无法透传问题]]
- [[wiki/02_precision/moe_shared_expert_overlap_grad|MOE Shared Expert Overlap 多流内存踩踏与梯度爆炸]]
- [[wiki/02_precision/vllm_virtual_memory_eviction|VLLM 显存不一致导致驱逐机制分叉]]
- [[wiki/02_precision/dsa_uniform_exponential_diff|NPU 硬件随机数实现异构导致的分布偏差]]
- [[wiki/02_precision/deepseek_sharding_dummy_load|Deepseek dummy 初始化的 Sharding 脱序]]
- [[wiki/02_precision/ring_mini_nz_format_diff|NZ 数据排布导致的 Off-Policy 激化]]
- [[wiki/02_precision/tensordict_async_memory_stomp|Tensordict缺原生NPU支持导致异步显存重叠复制]]
- [[wiki/02_precision/vlm_rl_actor_blind_diff|VLM RL 架构级概率不一致发散 (XLLM -100 Token 丢失)]]
- [[wiki/02_precision/vlm_2b5_w8a8_gibberish|VLM 2.5B W8A8 推理高熵乱码问题 (RoPE 冲突)]]
- [[wiki/02_precision/vlm_rl_actor_rope_logprobs_diff|VLM RL 长序列 RoPE 偏移导致 logprobs_diff 发散 (400x vs 20x)]]
- [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
### 3. [[wiki/03_tuning/index|性能调优 (Tuning)]]
- [[wiki/03_tuning/xpu_gpt_w8a8_rl_scheduler_memory_tuning|Tuning: xpu-gpt W8A8 RL scheduler batch 与显存水位调优]]
- [[wiki/03_tuning/ep_zero_sync|Tuning: EP 零冗余权重同步与 Resharding 优化 (ZeroSync)]]
- HCCL 并发与通信优化
- 显存复用与 OOM 解决
- 算子融合 (Operator Fusion) 策略
- Profiling 结果分析

### 4. [[wiki/04_frameworks/index|框架适配 (Frameworks)]]
- **vLLM**: NPU Backend 改造点
- **Megatron-LM**: 昇腾适配版分析
- **DeepSpeed**: 加速库适配

### 5. [[wiki/05_fact/index|事实知识 (Facts)]]
- [[wiki/05_fact/vlm_2b5_32rank_actor_rollout_deployment|Fact: VLM 2B5 xpu_gpt 32rank Actor 与 Rollout 部署形态]]
- [[wiki/05_fact/xpu_gpt_torch_infer_merge_xg_comparison|Fact: xpu_gpt + torch inference 合并后 xg 回归对比结论]]
- [[wiki/05_fact/vlm_2b5_vit_llm_deployment_architecture|Fact: VLM 2B5 VIT与LLM的混合部署架构约束]]
- [[wiki/05_fact/vlm_2b5_inference_service_golden_config|VLM 2B5 推理服务化部署黄金配置 (DP16/EP16)]]
- [[wiki/05_fact/vlm_2b5_rl_rollout_quantization|VLM 2.5B RL 训推转换低精度 Rollout 量化实现 (D_W8A8C8)]]
- [[wiki/05_fact/vlm_2b5_rl_rollout_timing|Fact: VLM 2.5B RL NPU Rollout 阶段的真实耗时约束]]
- [[wiki/05_fact/torch_infer_merge_debug_checklist|Fact: xpu_gpt + torch infer 合并验证排查清单]]
- 约束与设计事实 (架构隔离、Rank 配置等)

---

## ⚙️ 操作指南

本 Wiki 遵循 [Karpathy llm-wiki 模式](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)，核心操作有三种：

| 操作 | 触发方式 | 说明 |
|------|---------|------|
| **Ingest** | `/ingest-bugfix`, `/ingest-precision`, `/ingest-tuning`, `/ingest-fact` | 吸收新的原始资料，提炼为 wiki 页面 |
| **Query** | `/query-wiki` 或直接提问 | 基于 wiki 知识回答问题，有价值的回答回写 |
| **Lint** | `/lint-wiki` | 健康检查：死链、孤立页、矛盾、陈旧、缺页 |

---

## 📋 日志与索引

- [[log|操作时间线]] — 按时间倒序记录所有 Ingest/Query/Lint 操作
- 快速查看最近操作: `grep "^## \[" log.md | tail -5`

---

## 🕒 最近更新

- `[2026-06-03]` 排错: [[wiki/00_bugfix/xpu_gpt_vlm_2b5_dual_node_a3_rollout_adaptation|xpu-gpt 2B5 VLM 双节点 A3 rollout 适配]]
- `[2026-06-03]` 事实: [[wiki/05_fact/vlm_2b5_32rank_actor_rollout_deployment|VLM 2B5 xpu_gpt 32rank Actor 与 Rollout 部署形态]]
- `[2026-06-03]` 性能调优: [[wiki/03_tuning/xpu_gpt_w8a8_rl_scheduler_memory_tuning|xpu-gpt W8A8 RL scheduler batch 与显存水位调优]]
- `[2026-06-02]` 精度对齐: [[wiki/02_precision/vlm_torch_infer_frozen_vlm_weight_sync_reward|torch infer frozen VLM 权重未同步导致 reward 低于合并前]]
- `[2026-06-02]` 排错: [[wiki/00_bugfix/vlm_template_reuse_rollout_vit_embeds|VLM RL Template 未复用 rollout vit_embeds 导致 Actor 侧重复图片编码]]
- `[2026-06-02]` 事实: [[wiki/05_fact/xpu_gpt_torch_infer_merge_xg_comparison|xpu_gpt + torch inference 合并后 xg 回归对比结论]]
- `[2026-06-02]` 排错: [[wiki/00_bugfix/vlm_torch_infer_scheduler_batch_env_override|torch infer VLM 默认 env 覆盖 scheduler/model batch 导致 RL step 劣化]]
- `[2026-06-01]` 排错: [[wiki/00_bugfix/vlm_rl_xg_hash_image_size_check|VLM 2B5 xg RL hash 路径未关闭 xLLM 图片尺寸检查导致 InvalidImageURL]]
- `[2026-06-01]` 排错: [[wiki/00_bugfix/xpu_gpt_decode_graph_cache_maxlen_mismatch|xpu_gpt decode graph cache 未按 max_length 隔离导致 12k->32k 图模式断言]]
- `[2026-06-01]` 排错: [[wiki/00_bugfix/vlm_2b5_vit_checkpoint_fallback|M8 2B5 VLM Companion ViT 权重路径 fallback 缺失导致启动失败]]
- `[2026-05-30]` 排错: [[wiki/00_bugfix/vlm_torch_infer_vit_cache_sp_slice_logpdiff|torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸]]
- `[2026-05-29]` 事实: [[wiki/05_fact/torch_infer_merge_debug_checklist|xpu_gpt + torch infer 合并验证排查清单]]
- `[2026-05-27]` 性能调优: [[wiki/03_tuning/ep_zero_sync|Tuning: EP 零冗余权重同步与 Resharding 优化 (ZeroSync)]]
- `[2026-05-26]` 精度对齐: [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
- `[2026-05-20]` 排错: [[wiki/00_bugfix/encoder_cache_eviction_assertion|Bugfix: xLLM 并发压测下 EncoderResultCache 驱逐触发 AssertionError]]
- `[2026-05-19]` 排错: [[wiki/00_bugfix/vlm_cache_hash_mismatch|Bugfix: VLM Cache Miss Due to Cross-Boundary Hash Mismatch]]
- `[2026-05-19]` 排错: [[wiki/00_bugfix/vit_cache_miss_cityhash|Bugfix: ViT Cache 100% Miss & Race Condition Mismatch]]
- `[2026-04-28]` 排错: [[wiki/00_bugfix/ep16_decode_deadlock_pd_count_fix|EP16 MoE Decode Deadlock pd_count 修复]]
- `[2026-05-16]` 排错: [[wiki/00_bugfix/vlm_thinking_parser_logpdiff_spike|Bugfix: VLM RL 训练中 thinking_parser 缺失导致 logpdiff 极端 Spike (0.3+)]]
- `[2026-05-09]` 排错: [[wiki/00_bugfix/host_oom_glibc_arena|宿主机 OOM 与 SYS 物理内存持续增长 (glibc malloc arena 碎片化)]]
- `[2026-04-14]` 排错: [[wiki/00_bugfix/cann_aclnngather_zero_length_crash|CANN aclnnGather 算子零长维报错与规避]]
- `[2026-04-12]` 事实: [[wiki/05_fact/vlm_2b5_inference_service_golden_config|VLM 2B5 推理服务化部署黄金配置]]
- `[2026-04-12]` 排错: [[wiki/00_bugfix/vlm_grpc_companion_routing_deadlock|VLM Companion Encoder gRPC 路由死锁与地址错配]]
- `[2026-04-11]` 排错: [[wiki/00_bugfix/veturborpc_env_override|Pydantic 拦截与 xLLM 环境映射机制]]
- `[2026-04-11]` 精度对齐: [[wiki/02_precision/vlm_rl_actor_blind_diff|VLM RL logprobs_diff 发散根因]]
- `[2026-04-11]` 初始架构构建
