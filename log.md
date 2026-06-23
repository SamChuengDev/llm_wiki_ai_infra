## [2026-06-23] ingest-precision | VLM 23B XG logprobs_diff spike replay 定位现场
- **来源**: step83 replay 日志 `outputs/logs/vlm_0622_124100_replay_step83_xg_23b_replay_step83_ep128_kv64_tok6144_if128_b8_chunk6144_debug_bs64.log`、strict replay 日志 `outputs/logs/vlm_0622_160204_replay_step83_strict_xg_23b_replay_step83_strict_ep128_kv64_tok6144_if128_b8_chunk6144_debug_bs64.log`、guard run 日志 `outputs/logs/vlm_0622_173707_guard_garbled_xg_23b_guard_ep128_kv64_tok6144_if128_b8_chunk6144_dyn32_safe_bs64.log`、full-batch dump `outputs/DEBUG/logprobs_diff_0623_000610_replay_step83_xpusampler_budget/logprobs_diff_step1_top8.pt` 与 `outputs/DEBUG/logprobs_diff_0623_030200_replay_step83_forced17mask/logprobs_diff_step1_top16.pt`。
- **提取点 1**: replay 能复现 step83 高 diff：普通/strict replay step1 分别为 `logprobs_diff=0.0436147/0.0436931`；guard run step1 仍为 `0.0625356`，step2 回到 `0.0288409`，因此过滤只能作为隔离手段，不能定性为根因修复。
- **提取点 2**: full-batch dump 中 `token_id=17 && rollout_logprob=0` 对 corr 破坏显著：两轮 dump 去掉该类 token 后 corr 从 `0.752793/0.760122` 回到 `0.96662/0.975799`，但 strict replay 没有 token17 zero，说明这不是唯一来源。
- **提取点 3**: 当前高疑点是 thinking forced end token / `thinking_budget` fallback 与 rollout logprob 注入不一致；no-budget replay 尚未完成，原因是验证环境 snapshot 只在 head 容器存在且 Ray working_dir/PYTHONPATH 容易混入旧仓路径。

## [2026-06-17] ingest-tuning | VLM 23B XG if144_b12 负收益与 ViT RECOMPUTE
- **来源**: XG ep128 if144_b12 日志 `outputs/logs/vlm_0617_121607_xg_23b_rollout_ep128_kv64_tok2048_if144_b12_kvprobe_hccl400c1024_200step_bs64.log`、HBM 监控 `outputs/DEBUG/hbm_8node_0617_121607_ep128_if144_b12.jsonl`、torch infer response-level `vit_embeds` 统一方案 `wiki/02_precision/vlm_torch_infer_response_vit_embeds_unified_cache.md`。
- **提取点 1**: step2 同口径下，`if144_b12` 的 `train/timing/step=1987.91s`，相比 `if144_b8` 的 `1796.14s` 慢 `191.77s / 10.7%`；`train/timing/data=1453.85s`，相比 `1322.53s` 慢 `9.9%`。
- **提取点 2**: b12 日志在 Actor 侧大量出现 `[ViT-Probe] path=RECOMPUTE`，例如日志 33116 行附近聚合为 `repeated 256x across cluster`；同一日志没有看到 `forwarded final/current vit_embeds`、`VanillaRollout forwarding vit_embeds` 或 `attached response vision_embeddings` 成功信号。
- **提取点 3**: `/dev/shm/vit_cache` 只能作为节点本地 fallback，不适合作为 8 节点全局 cache。下一组应回到 `if144_b8`，单独验证 `XPU_KV_NUM_BLOCKS=80` 与 `MAAS_VLM_USE_ROLLOUT_VIT_EMBEDS=1` 的 response-level ViT 透传；成功信号是 Actor 侧出现 `path=BYPASS` 且不再大规模 `RECOMPUTE`。

## [2026-06-17] ingest-tuning | VLM 23B XG ep128 if160_b8 持平/负收益
- **来源**: XG ep128 if160_b8 日志 `outputs/logs/vlm_0617_095100_xg_23b_rollout_ep128_kv64_tok2048_if160_b8_kvprobe_hccl400c1024_200step_bs64.log`、HBM 监控 `outputs/DEBUG/hbm_8node_0617_095100_ep128_if160_b8.jsonl`。
- **提取点 1**: step2 同口径下，`if160_b8` 的 `train/timing/step=1790.48s`，相比 `if144_b8` 的 `1796.14s` 仅快 `5.66s / 0.3%`，不构成有效性能收益。
- **提取点 2**: `if160_b8` 的 `speed_tokens=595.94` 低于 `if144_b8` 的 `603.30`，`logprobs_diff=0.0617` 明显劣于 `if144_b8` 的 `0.0300`。
- **提取点 3**: 完整 run HBM 峰值仍为 `65471 MB`，接近卡上限；当前最优候选保持 `ep128 + if144 + kv64 + ctx8/b8/model8`，下一组优先在 `if144` 上小步试 `scheduler/model/context batch=12`，不要继续单独加 in-flight。

## [2026-06-17] ingest-tuning | VLM 23B XG ep128 if144_b8 正收益
- **来源**: XG ep128 if144_b8 日志 `outputs/logs/vlm_0617_073951_xg_23b_rollout_ep128_kv64_tok2048_if144_b8_kvprobe_hccl400c1024_200step_bs64.log`、HBM 监控 `outputs/DEBUG/hbm_8node_0617_073951_ep128_if144_b8.jsonl`。
- **提取点 1**: step2 同口径下，`if144_b8` 的 `train/timing/step=1796.14s`，相比历史 `if128_b8` 的 `1974.33s` 快 `178.19s / 9.0%`；`train/timing/data=1322.53s`，相比 `1488.46s` 快 `11.1%`。
- **提取点 2**: `if144_b8` 的 `speed_samples=0.2851`、`speed_tokens=603.30`，均高于 `if128_b8` 的 `0.2593`、`548.72`；response length 与 num_tokens 基本持平，说明收益主要来自 rollout/data 阶段。
- **提取点 3**: 完整 run HBM 峰值仍为 `65496 MB`，与 if128 基本持平；当前最优候选更新为 `ep128 + if144 + kv64 + ctx8/b8/model8`，下一组只小步验证 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS=160`，不直接增加 KV blocks。

## [2026-06-17] ingest-tuning | VLM 23B XG ep128 if96_b8 负收益
- **来源**: XG ep128 if96_b8 日志 `outputs/logs/vlm_0617_015800_xg_23b_rollout_ep128_kv64_tok2048_if96_b8_kvprobe_hccl400c1024_200step_bs64.log`、HBM 监控 `outputs/DEBUG/hbm_8node_0617_015800_ep128_if96_b8.jsonl`。
- **提取点 1**: step2 同口径下，`if96_b8` 的 `train/timing/step=2279.70s`，相比历史 `if128_b8` 的 `1974.33s` 慢 `305.37s / 15.5%`；`train/timing/data=1801.77s`，相比 `1488.46s` 慢 `21.0%`。
- **提取点 2**: `if96_b8` 的 `speed_samples=0.2246`、`speed_tokens=482.26`，均明显低于 `if128_b8` 的 `0.2593`、`548.72`；response length 只长约 `1.9%`，不能解释主要退化。
- **提取点 3**: 完整 run HBM 峰值仍为 `65501 MB`，降低 in-flight 没有释放峰值窗口；当前最优候选仍应回到 `ep128 + if128 + kv64 + ctx8/b8/model8`，不要在无 OOM 余量证据时增加 `XPU_KV_NUM_BLOCKS`。

## [2026-06-17] ingest-tuning | VLM 23B XG ep128 ctx16_b8 持平/负收益
- **来源**: XG ep128 ctx16_b8 日志 `outputs/logs/vlm_0616_234100_xg_23b_rollout_ep128_kv64_tok2048_if128_ctx16_b8_kvprobe_hccl400c1024_200step_bs64.log`、HBM 监控 `outputs/DEBUG/hbm_8node_0616_234100_ep128_ctx16_b8.jsonl`。
- **提取点 1**: step2 同口径下，`ctx16_b8` 的 `train/timing/step=1979.38s`，相比历史 `ctx8_b8` 的 `1974.33s` 略慢 `0.3%`；`train/timing/data=1499.35s`，相比 `1488.46s` 慢 `0.7%`。
- **提取点 2**: `ctx16_b8` 的 `speed_samples=0.2587`、`speed_tokens=546.59`，均略低于 `ctx8_b8` 的 `0.2593`、`548.72`；response length 与 num_tokens 基本持平，不能解释为样本分布收益。
- **提取点 3**: step2 HBM 时间窗峰值为 `62860 MB`，但 full run 仍出现 `65499 MB` 峰值；单独增大 context batch 没有性能收益，下一轮应回退 `ctx8/b8/model8`，单独验证 `MAAS_ROLLOUT_MAX_INFLIGHT_REQUESTS=96` 是否缓解 decode 长尾和 HBM 峰值。

## [2026-06-16] ingest-tuning | VLM 23B XG ep128 b8/b16 调参对比
- **来源**: XG ep128 b8 日志 `outputs/logs/vlm_0616_191446_xg_23b_rollout_ep128_kv64_tok2048_if128_b8_kvprobe_hccl400c1024_200step_bs64.log`、HBM 监控 `outputs/DEBUG/hbm_8node_0616_191446_ep128_sync0_hccl300.jsonl`、XG ep128 b16 日志 `outputs/logs/vlm_0616_210520_xg_23b_rollout_ep128_kv64_tok2048_if128_b16_kvprobe_hccl400c1024_200step_bs64.log`、HBM 监控 `outputs/DEBUG/hbm_8node_0616_210507.jsonl`。
- **提取点 1**: step2 同口径下，`scheduler/model batch=8` 的 `train/timing/step=1974.33s`，而 `batch=16` 为 `2074.29s`，慢 `99.96s / 5.1%`；`train/timing/data` 从 `1488.46s` 变为 `1586.49s`，几乎解释全部回退。
- **提取点 2**: b16 的 `speed_samples=0.2468`、`speed_tokens=527.69`，低于 b8 的 `0.2593`、`548.72`；response length 只长约 `1.4%`，不能解释 5% 的 step 回退。
- **提取点 3**: 两轮 HBM 完整 step 峰值都接近 `65536 MB`，不能用 rollout 稳态低水位判断可无风险继续加 KV 或 decode batch。下一轮应回退 `model_max_batch_size=8`，拆开验证 context/prefill batch，而不是继续整体放大 b16。

## [2026-06-16] ingest-tuning | VLM 23B RL torch infer 与 XG 推理后端性能对比
- **来源**: torch infer 启动脚本 `run_torch_infer_vlm_23b.sh`、XG 启动脚本 `run_xg_vlm_23b.sh`、torch infer 日志 `outputs/logs/vlm_0606_070312_torch_infer_23b_full_8nodes_vit_responsefallback_bs256_long_bs256_57steps.log`、XG 日志 `outputs/logs/vlm_0616_065614_xg_23b_rollout_ep64_kv64_tok2048_hccl400c1024_200step_bs64.log` 及同时间戳 runtime env。
- **提取点 1**: torch infer 0606 历史 run 的实跑口径是 `data.batch_size=256`、`rollout.generation.n=8`、`train/batch_size=2048`、`rollout.tp_size=8/dp_size=2`、`enable_async=True`、`scheduler/model batch=512`、chunk `10240`；当前脚本默认已是 `384/8192`，不能用当前脚本默认值倒推该历史日志。
- **提取点 2**: XG 0616 当前实跑口径是 `data.batch_size=64`、`rollout.generation.n=8`、`train/batch_size=512`、`rollout.ep_size=64`、`scheduler/model batch=8`、chunk `2048`、`XPU_KV_NUM_BLOCKS=64`、W8A8；前 3 step 平均 `train/timing/step=2611.06s`、`speed_samples=0.1984`，显著慢于 torch infer 57 step 平均 `1902.80s`、`1.0773`。
- **提取点 3**: XG 关键瓶颈是 rollout 有效并发只有 64（512 总请求中 448 pending，每个 client 8 running）、scheduler/model batch 只有 8、`sync_weight` 平均 169.69s、`collect_quant_amax` 平均 16.85s、以及低 HBM 工作区 empty_cache 信号；torch infer 完成 57 step 后在 step58 前后因 torch_npu stream timeout/raylet dead 终止，不能称为长跑稳定通过。

## [2026-06-06] lint | health check
- **扫描**: 59 个 Markdown 文件（根 `index.md`、`log.md` 与 `wiki/**/*.md` 页面）。
- **发现**: 18 个断链、0 个孤立页面、2 个根文件 frontmatter 缺失提示、0 个 `status: active` 超 90 天页面；未发现本轮新增 precision 页面引入的新断链。
- **自动修复**: 追加本次 lint 日志；未创建占位页面，避免把规划主题误写成事实页面。
- **待人工处理**: 剩余断链仍主要是规划/占位概念或待独立成文主题，包括 `vllm_npu`、`mapping_table`、`custom_operators`、`overflow_handling`、`alignment_tools`、`hccl_optimization`、`memory_management`、`megatron_npu`、`xllm_engine_scheduling` 等；根 `index.md` / `log.md` 是否补 frontmatter 需先统一 wiki 约定。

## [2026-06-06] ingest-precision | torch infer VLM rollout response 直传 vit_embeds 统一 ViT cache 方案
- **来源**: 当前 23B full 8 节点 response fallback 验证日志 `outputs/logs/vlm_0606_070312_torch_infer_23b_full_8nodes_vit_responsefallback_bs256_long_bs256.log`，2B5 单节点长跑日志 `outputs/logs/vlm_0602_113333_24189_torchinfer_align_ypq_bs128.log`，以及 torch infer / RL 侧 ViT cache 相关代码。
- **提取点 1**: 当前存在三层 ViT cache/复用机制：xLLM torch infer 内部 `LRUDict` vision cache、RL 侧 `/dev/shm/vit_cache` 文件 cache、以及 rollout final response 直传 `vit_embeds`。其中 `/dev/shm` 适合单节点本地共享，不适合作为多节点全局 cache。
- **提取点 2**: 单机/多机推荐统一到 `MAAS_VLM_USE_ROLLOUT_VIT_EMBEDS=1` 的 response-level `vit_embeds` 直传，让 embedding 跟随 sample 数据流进入 Actor/ref；`/dev/shm/vit_cache` 仅保留为 `MAAS_VLM_USE_ROLLOUT_VIT_EMBEDS=0 && ENABLE_VIT_CACHE=1` 的旧 fallback。
- **提取点 3**: 23B full bsz=256 早期 step1-7 的 `logprobs_diff` 稳定在 `0.0171~0.0176`，且 `forwarded final response vit_embeds` 与 `attached response vision_embeddings` 均无 missing/token mismatch；该证据不能替代 70+ / 200 step 长线漂移验证。xpu-gpt + Companion ViT 路径尚未适配 response-level attach，不应强行统一。

## [2026-06-04] ingest-fact | NPU HBM 虚拟显存 expandable_segments 特性
- **来源**: 用户会话说明，当前仓 `runtime_env_xg.yaml` / `run_xg_vlm_2b5.sh` / `run_xg_vlm_2b5_before_merge.sh`，以及 23B torch inference baseline 日志 `outputs/precision_cmp/torch_infer_23b_128p_baseline_0510_212035.log`。
- **提取点 1**: `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 是 NPU PyTorch allocator 的虚拟显存 / HBM 管理特性开关，通常可降低 HBM 使用与碎片化压力，缓解 NPU device OOM 风险。
- **提取点 2**: 该变量属于合并前后实验对齐时需要检查的内存环境配置；若 baseline worker 环境中开启，合并后 23B torch inference 脚本也应保留开启，除非有实测证据要求关闭。
- **提取点 3**: 该特性不能替代 `torch.npu.empty_cache()` 对物理 HBM 的释放，也不能单独作为 OOM 修复成功证据；仍需日志和长跑验证闭环。

## [2026-06-03] ingest-bugfix | xpu-gpt 2B5 VLM 双节点 A3 rollout 适配
- **来源**: 双节点 2450 + 26100 验证会话，失败日志 `outputs/logs/vlm_0603_115505_2614_xg_graph_perfopt_bs128.log`，bsz=128 验证日志 `outputs/logs/vlm_0603_130806_xg_2nodes_2b5_bs128.log`，bsz=256 验证日志 `outputs/logs/vlm_0603_153936_xg_2nodes_2b5_bsz256_bs256.log`。
- **提取点 1**: 双节点初始失败包含 Ray 版本不一致以及 `assert replicas_per_node != 0`；后者与 `ray.nodes()` 中 `Alive=False` / 无 NPU 资源的残留节点相关，训练资源池和 rollout replica 枚举必须过滤 non-alive / no-device nodes。
- **提取点 2**: 多节点 xLLM rollout replica 需要使用 Ray node 本机 IP 作为 rendezvous 地址；修复后日志显示 head replica 使用 `10.101.24.50`，worker replica 使用 `10.101.26.100`，不再依赖单独 runtime env 开关。
- **提取点 3**: 统一 `runtime_env_xg.yaml` 后，bsz=256 已验证到 `global_step=4`，reward 非全零，`logprobs_diff` 为 `0.0388/0.0287/0.0296/0.0293`，fatal 扫描未命中 Traceback/OOM/RPC；该结论不替代 70+ / 200 step 长线漂移验证。

## [2026-06-03] ingest-fact | VLM 2B5 xpu_gpt 32rank Actor 与 Rollout 部署形态
- **来源**: 双节点 2450 + 26100 验证会话，配置 `run_xg_vlm_2b5.sh` / `runtime_env_xg_32rank.yaml` / `ray_xg_32rank.sh`，代码公式 `maas_finetune/trainer/rl/dist_utils.py` 与 `maas_finetune/trainer/rl/rollout/ray_driver.py`，验证日志 `outputs/logs/vlm_0603_130806_xg_2nodes_2b5_bs128.log`。
- **提取点 1**: Rollout 口径为 `rollout.tp_size=1`、`rollout.ep_size=16`，每个 xLLM rollout replica 占 16 NPU；双节点各 16 NPU 时形成 2 个本地 16-rank replica，即 rollout 层面 DP=2，每节点 1 个副本。
- **提取点 2**: Actor 口径为 `actor.fsdp_size=-1`、`actor.ep_size=2`、`actor.sp_size=4`；32rank 下 `fsdp_size=-1` 使模型副本维度 `dp_replicate_size=1`，但数据 dispatch 维度为 `world_size / sp_size = 8`，不能把 actor 的数据 DP 误解为 1。
- **提取点 3**: `MAAS_XG_ROLLOUT_USE_NODE_IP=1` 是多节点 rollout 的 rendezvous 地址修正开关，不直接决定 DP 数；扩到更多节点时仍需保留，但 3/4 节点以上规模需要对应实测确认。

## [2026-06-03] lint | health check
- **扫描**: 55 个 Markdown 文件（根 `index.md`、`log.md` 与 `wiki/**/*.md` 页面），并确认旧根目录 `00_bugfix/` 已不存在，唯一 Bugfix 目录为 `wiki/00_bugfix/`。
- **发现**: 18 个断链、0 个孤立页面、2 个根文件 frontmatter 缺失提示、0 个 `status: active` 超 90 天页面、未发现明确矛盾标记；另有 41 个非强制反链候选。
- **自动修复**: 本轮未做新的自动修复；剩余断链均是占位/规划概念或需要单独成文的主题，不适合猜测补页。
- **待人工处理**: 建议后续按优先级补齐 `vllm_npu`、`hccl_optimization`、`memory_management`、`overflow_handling`、`mapping_table`、`xllm_engine_scheduling` 等缺页；是否给根 `index.md` / `log.md` 增加 YAML frontmatter 需要先统一 wiki 约定。

## [2026-06-03] lint | health check
- **扫描**: 55 个 Markdown 文件（根 `index.md`、`log.md` 与 `wiki/**/*.md` 页面）。
- **发现**: 初扫发现 22 个断链、0 个孤立页面、2 个根文件 frontmatter 缺失提示、0 个 `status: active` 超 90 天页面、未发现明确矛盾标记；复扫后剩余 18 个断链与 41 个非强制反链候选。
- **自动修复**: 将旧目录已有内容补齐为正式 wiki 页面 `wiki/00_bugfix/encoder_cache_eviction_assertion.md` 与 `wiki/00_bugfix/npu_empty_virt_addr_cache_oom.md`，并接入 `wiki/00_bugfix/index.md`；断链从 22 降至 18。
- **待人工处理**: 剩余断链多为规划/占位概念或需要单独成文的主题，包括 `vllm_npu`、`hccl_optimization`、`memory_management`、`overflow_handling`、`mapping_table`、`xllm_engine_scheduling` 等；反链候选多为“关联知识”的弱关联，未批量自动补齐。

## [2026-06-03] ingest-tuning | xpu-gpt W8A8 RL scheduler batch 与显存水位调优
- **来源**: 调整前日志 `outputs/logs/vlm_0602_165949_2614_xg_graph_bs128.log` / `_filter.log`，调整后日志 `outputs/logs/vlm_0602_210957_2614_xg_graph_perfopt_bs128.log` / `_filter.log`。
- **提取点 1**: 配置差异为 `gpu_memory_utilization=0.7 -> 0.85`，`scheduler_max_schedule_batch_size=512 -> 768`，`scheduler_max_schedule_context_batch_size=512 -> 768`，`model_max_batch_size=512 -> 768`；`inferencer_async_runtime_chunk_size=10240` 未变化。
- **提取点 2**: 同口径统计 step 3-5 与 11-15 共 8 个 step，端到端 `train/timing/step` 从 `427.329s` 降到 `422.394s`，提升约 `1.15%`；rollout 从 `203.945s` 降到 `198.072s`，提升约 `2.88%`。
- **提取点 3**: 主要收益来自 rollout 与 `sync_weight` 小幅下降，`update_actor`、`collect_quant_amax` 略变慢。该优化可作为小幅正收益配置记录，但 768/0.85 不是已证明最优值，仍需长线 OOM 与吞吐稳定性观察。

## [2026-06-02] ingest-precision | torch infer frozen VLM 权重未同步导致 reward 低于合并前
- **来源**: 合并前 torch infer 日志 `/Users/bytedance/posttrain/codes/maas_engine_torch_infer/outputs/logs/vlm_0528_002344.log`，合并后坏日志 `outputs/logs/vlm_0601_213829_24189_torchinfer_align_ypq_bs128_filter.log`，修复验证日志 `outputs/logs/vlm_0602_113333_24189_torchinfer_align_ypq_bs128.log`。
- **提取点 1**: 合并后 reward mean 明显低于合并前，但 `logprobs_diff` 仍在 `~0.016-0.018`，说明 actor/rollout 概率对齐健康，问题更可能在共同使用的视觉 embedding 质量上。
- **提取点 2**: 同一 image hash 的 VLM embedding 从合并前 mean `0.006879` 变为坏 run mean `-40.2794`；坏日志缺少 `vision_encoder` / `multi_modal_projector` 的 SHM 同步行，torch infer DirectEngine 初始化 mock weights 后未收到真实 frozen VLM 权重。
- **提取点 3**: 修复为 `XLLM_MODEL_CLS=torch` 时不跳过 `vision_encoder.`、`ln_vision.`、`multi_modal_projector.` 权重同步；`xpu` 或未设置时继续跳过，保持 xpu_gpt/Companion 路径不变。验证到 step14，ViT-Probe mean 恢复到 `~0.005-0.008`，`logprobs_diff~0.016`，但仍需 70+ / 200 step 长线验证。

## [2026-06-02] ingest-bugfix | VLM RL Template 未复用 rollout vit_embeds 导致 Actor 侧重复图片编码
- **来源**: 提交 `58187c25f99f83db9f212e79538ee928c7a47c86`，xg 验证日志 `outputs/logs/vlm_0601_165516_2450_after_merge_bs128_filter.log`，torch inference 验证日志 `outputs/logs/vlm_0601_213829_24189_torchinfer_align_ypq_bs128_filter.log`。
- **提取点 1**: 旧模板在样本已有 rollout `vit_embeds` 时仍逐图调用 `_encode_image()`，最后再把 `vit_embeds` 塞回输出，造成 Actor/ref 侧重复图片编码与像素输入/embedding 语义混用风险。
- **提取点 2**: 修复为有 `vit_embeds` 时跳过 `_encode_image()`，用 `input_ids` 中 image token 数作为 `total_image_tokens`，并新增 `vit_embeds.shape[0] == image_token_count` 校验；无 `vit_embeds` 时保留原图片编码路径。
- **提取点 3**: xg 后端日志已到 120 step，尾部 `logprobs_diff` 约 `0.0236~0.0259`；torch inference 后端日志已到 62 step，尾部 `logprobs_diff` 约 `0.0171~0.0182`；两份日志均未见 Traceback 或新增 token mismatch discard。该结论不替代 200 step 长线精度漂移验证。

## [2026-06-02] ingest-fact | xpu_gpt + torch inference 合并后 xg 回归对比结论
- **来源**: 合并前完整日志 `24167:/data01/zcs/maas_engine_vlm2b5/outputs/logs/vlm_0525_220913_24167_thnk_parser_enabled_fix_prefix_cache_bs128.log`，合并后完整日志 `outputs/logs/vlm_0601_165516_2450_after_merge_bs128.log`，两侧 filter/metric 日志与 runtime env。
- **提取点 1**: 完整日志 Hydra dict 结构化 diff 只有 `trainer.trajectory_logger.stream` 时间戳不同；Data、Actor、Ref、Rollout、trainer 主配置一致，包括 `data.batch_size=128`、`rollout.generation.n=8`、`rollout.model.path=/share_data/2b5_saved_model_for_xg/`、`thinking_parser.enabled=True`。
- **提取点 2**: LLM target 路径文本从 `saved_model/global_step_0/model.safetensors` 变为 `saved_model/W8A8/model.safetensors`，但二者解析到同一 W8A8 权重且 `cmp_llm_exit=0`；Companion ViT 路径文本不同但 `cmp_vit_exit=0`，不能把权重不同作为已证实差异原因。
- **提取点 3**: 环境变量存在真实差异：`CPU_AFFINITY_CONF=2 -> 1,npu_affine:1`、host 地址变化、0525 额外 logprobs debug 与 draft/spec 文本变量；投机推理未实锤生效，不能用 draft/eagle 文本字段解释精度差异。指标上 126 step 对齐显示 `logprobs_diff` 后者平均高约 `+0.00134`，step time 后者平均低约 `63s`，但这不是 200 step 长线精度结论。
- **提取点 4**: 性能上合并后单个 RL step 平均减少约 `50~60s`，126 step 对齐均值为 `-63.1755s`；主要分项包括 `sync_weight` 约 `-31s`、`update_actor` 约 `-27s`、`offload_weights/data` 小幅下降。`CPU_AFFINITY_CONF` 的绑核变化是候选解释，但由于日志提示 `npu_affine` 不被支持并被禁用，仍需单独 A/B 才能定量归因。

## [2026-06-02] ingest-bugfix | torch infer VLM 默认 env 覆盖 scheduler/model batch 导致 RL step 劣化
- **来源**: merge 慢日志 `outputs/logs/vlm_0601_205541_24189_torchinfer_align_ypq_bs128.log`，源目录健康对照 `/Users/bytedance/posttrain/codes/maas_engine_torch_infer/outputs/logs/vlm_0528_002344.log`，修复文件 `maas_finetune/trainer/rl/rollout/ray_driver.py`。
- **提取点 1**: 失败签名不是 Traceback，而是 `train/timing/data=966.465s`、step `1131.7309s`；Hydra 配置为 `scheduler/model batch=512`，但 xLLM 实际启动为 `max_schedule_batch_size=64` 与 `model max_batch=64`。
- **提取点 2**: 调度证据显示 merge 日志只活跃 `DECODING:64` 且 `PREFILL:134` 堆积，async `first_token/enqueue/generate_waiting_time` 可达百秒级；源目录同口径日志可见 `DECODING:256`，first token/enqueue 多为 `3~5s`。
- **提取点 3**: 根因定位到 `ray_driver._setup_envs()` 先写入 rollout 512，随后 `VLM_DEFAULT_ENVS` 覆盖 `XLLM_SCHEDULER_MAX_SCHEDULE_BATCH_SIZE` 与 `XLLM_MODEL_MAX_BATCH_SIZE` 为 64；修复为 VLM 默认合并后重新应用显式 rollout 配置。已通过 py_compile、`_setup_envs` 与 xLLM Launcher 烟测，尚需新 RL 日志确认 step time 恢复。

## [2026-06-01] ingest-bugfix | VLM 2B5 xg RL hash 路径未关闭 xLLM 图片尺寸检查导致 InvalidImageURL
- **来源**: 2450 xg 失败日志 `outputs/logs/vlm_0530_204441_2450_xg_bs128.log`，修复验证日志 `outputs/logs/vlm_0601_093232_2450_xg_bs128.log`，提交 `039c20a3420c0e4f0870396b595d3bbb9a9b92cf`。
- **提取点 1**: 失败签名为 DataProcessor 阶段 `InvalidImageURL: width = 6840, height = 3549, Maximum allowed dimension: 6000 pixels`，发生在 xg 运行到约 `global_step=19` 后处理下一批样本时。
- **提取点 2**: 证据指向 Actor/hash 路径手动构造 `VLMDialogProcessor` 时未显式传 `check_image_size`；仅设置 `XLLM_PROCESSOR_CHECK_IMAGE_SIZE=0` 不会改变构造参数默认值 `True`。
- **提取点 3**: 修复为 `RLTemplate._encode_image()` 显式传入 `check_image_size=_env_bool("XLLM_PROCESSOR_CHECK_IMAGE_SIZE", False)`；2450 xg 验证到 `global_step=20`，未再出现 `InvalidImageURL` / Traceback。该结论不替代 70+ / 200 step 长线精度漂移验证。

## [2026-06-01] ingest-bugfix | M8 2B5 VLM Companion ViT 权重路径 fallback 缺失导致启动失败
- **来源**: 2450 xpu_gpt + Companion ViT 启动失败日志 `vlm_0601_161331_2450_after_merge_ge_graph_bs128.log`，修复验证日志 `vlm_0601_165516_2450_after_merge_bs128.log`，代码路径 `3rdparty/xpu_gpt/xpu_gpt/adapter/xllm_adapter/xpu_loader.py`。
- **提取点 1**: 失败签名为 `[Companion] ArkLauncher launch failed` 后，`_resolve_vit_checkpoint()` 只查 `/share_data/2b5_saved_model_for_xg/saved_model/global_step_0` 并报 `ValueError: ViT model not found`。
- **提取点 2**: 修复为 ViT checkpoint 解析增加 fallback：优先 `saved_model/global_step_*/vit.*`，随后查 `saved_model/vit.*` 与模型根目录 `vit.*`；多个 `global_step_*` 显式报错，避免静默选错。
- **提取点 3**: 验证日志显示 `Use ViT cruise config ... checkpoint: .../vit.safetensors`、`ArkLauncher launch done`，并已进入多轮 `global_step`；该结论只覆盖启动加载问题，不代表 70+ / 200 step 长线精度验证完成。

## [2026-06-01] lint | health check
- **扫描**: 46 个 Markdown 文件（根 `index.md`、`log.md` 与 44 个 `wiki/**/*.md` 页面）。
- **发现**: 初扫发现 25 个断链、2 个孤立页面、10 个 frontmatter 字段问题、15 个非强制反链缺口；未发现 `status: active` 超 90 天页面或明确矛盾标记。
- **自动修复**: 修正 3 处 `vit_cache_miss` 改名断链；将 `ep16_decode_deadlock_pd_count_fix` 接入根索引与 Bugfix 分类索引；补齐 10 个 frontmatter 字段问题；为 xpu_gpt graph cache bugfix 与 torch infer merge checklist 增加 1 条反链。
- **待人工处理**: 剩余 22 个断链，多数是规划占位或尚未成文页面（如 `encoder_cache_eviction_assertion`、`npu_empty_virt_addr_cache_oom`、`vllm_npu`、`mapping_table` 等）；15 个宽松关联型反链缺口未自动补齐，避免制造弱关联。

## [2026-06-01] ingest-bugfix | xpu_gpt decode graph cache 未按 max_length 隔离导致 12k->32k 图模式断言
- **来源**: 24140 xpu_gpt 图模式排查会话，失败日志 `vlm_0601_170605_24140_xg_graph_bs128.log`，验证日志 `vlm_0601_172008_24140_xg_graph_bs128.log`，历史可跑通对照 `vlm_0526_105736_2450_allcodes_ge_graph_bs128_filter_432steps.log`。
- **提取点 1**: 失败签名为 torchair cached graph 调用中的 `AssertionError: expected size 64==64, stride 65==25 ... expected size 65==25`，发生在 xpu_gpt Eagle target decode graph verification 路径。
- **提取点 2**: 诊断确认本次请求进入 `decode_graph` 前 `kv_idx` 已是 `(64,65)`，旧 `(64,25)` 更符合 stale `compile_cache` 中旧 12k 长度口径 GE/OM 图元数据复用，而不是实时输入 shape 错误。
- **提取点 3**: 修复为 xpu_gpt 后端按 `max_sequence_length` 隔离 `XPU_GRAPH_CACHE_DIR`，例如 `compile_cache/maxlen_32768`；验证跨过旧崩溃点并到达 `global_step=3`，但仍不替代 70+ step / 200 step 长线精度结论。

## [2026-05-30] ingest-bugfix | torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸
- **来源**: 24189 torch infer 排查会话及日志 (`vlm_0530_190651_24189_torchinfer_bs128.log`, `vlm_0530_193421_24189_torchinfer_bs128.log`, `vlm_0530_195746_24189_torchinfer_bs128.log`) 与健康对照 (`vlm_0529_230400_2450_bs128_torch_infer_2b5_bs128.log`)。
- **提取点 1**: 确定 step 级 `logprobs_diff` 从 `0.079/0.061` 回落到 `0.0179/0.0168` 的直接修复来自两个动作：启动时清空全局 ViT cache，以及禁止对 cached `vit_embeds` 做 sequence-dim SP 切片。
- **提取点 2**: 明确 cached `vit_embeds` 与 `pixel_values` 不是同一种张量语义。cached embedding 应完整传给模型作为 `image_embeds`，由模型内部对 hidden-dim 做 SP 切分；外层按 dim=0 切片会把图像 token 数切成约 `1/4`。
- **提取点 3**: 总结了诊断指标：BYPASS ViT mean 偏到 `-68~-70` 优先怀疑 stale cache；mean 回到 `-35~-40` 但 BYPASS shape 总量仍只有健康日志约 `1/4` 时，优先怀疑 sequence parallel 误切。

## [2026-05-29] ingest-fact | xpu_gpt + torch infer 合并验证排查清单
- **来源**: torch infer 合并验证会话及日志 (`vlm_0529_142128_torch_infer_2b5_bs2.log`, `vlm_0529_143654_torch_infer_2b5_bs2.log`, 源分支 `vlm_0528_002344.log`)。
- **提取点 1**: 明确 OOM 与 `logprobs_diff` 异常必须拆开判断；最新日志无 OOM/Traceback 时，应把显存探针作为监控项，而不是继续盲目调 batch 或 KV 参数。
- **提取点 2**: 记录 Ray `runtime_env` 相同不代表代码包相同，必须同时核对 `runtime_env.working_dir` 的 Ray package、源码行号和远端同步状态。
- **提取点 3**: 校准 torch infer 中 `XLLM_INFERENCER_KERNEL_BACKEND=xpu` 的含义：它是 xLLM torch layer registry 名，不等于 `xperf_gpt`；`Unknown kernel backend: xpu` 应优先查 `layers/xpu` 注册和同步。

## [2026-05-27] ingest-tuning | EP 零冗余权重同步与 Resharding 性能优化
- **来源**: 诊断会话上下文及 5-step 测试日志 (vlm_0527_142116_2739_verify_eptrans_bs128.log)
- **提取点 1**: 确定了 16EP NPU 多卡多模态强化学习训练大 Step 更新权重时的通信与显存瓶颈为 MoE 专家权重的全量 Broadcast。
- **提取点 2**: 明确了通过 PyTorch 原生 `dist.scatter` 进行零冗余同步（ZeroSync），以及基于共享内存（SHM）“先合后分”以动态适配 `train_ep != infer_ep` 的自适应 Resharding 优化方案。
- **提取点 3**: 提炼出开启 `MAAS_XLLM_EP_ZERO_REDUNDANT_SYNC=1` 的调优步骤，实测在 bs=128 下权重同步平均耗时缩短 1.86秒（3.36%），稳态单步耗时缩短 6.65秒（11.26%），并将数据接收缓冲区压缩至 1/16，彻底消除了权重同步瞬时显存分配引发的连环 OOM。

## [2026-05-26] ingest-precision | VLM 2.5B RL logprobs_diff 长线漂移精度对齐
- **来源**: 诊断会话上下文及调试计划日志 (logpdiff_drift_debug_plan.md)
- **提取点 1**: 确定了长线训练中训推一致性 `logprobs_diff` 漂移的主要根因在于 rollout 侧 KV Cache/Prefix Cache 的跨请求污染，它无视了环境的禁用设置而被悄悄打开并复用。
- **提取点 2**: 明确了次要主因和误差放大器为 `thinking_parser` 缺失强制前缀约束，导致模型策略滑向 no-think / 乱码前缀，超短不规则文本极易放大打分噪声。
- **提取点 3**: 提炼出消除 prefix cache 请求级覆盖并锁死 `cached_length` 恒为 0，同时在 `run.sh` 中显式硬配置 thinking_parser 强制 `<think>` prefill 的对齐修复方案。在 124 step 长跑下 diff 稳定在 0.02x 健康水平。

## [2026-05-20] ingest-bugfix | xLLM 并发压测下 EncoderResultCache 驱逐触发 AssertionError
- **来源**: Client 并发 stress 压测与服务端日志 (dp16ep16_srv_*.log)
- **提取点 1**: 确定了在高并发压测下，Eviction 机制由于访问已被 `finish_query` 清空了 `multi_modal_data` 的 `ImageEncoderQuery` 的 `mm_token_length` 与 `mm_data_hash_id` 属性，导致触发 `AssertionError` 崩溃。
- **提取点 2**: 明确了通过在创建时持久化 `token_len` 和 `hash_id` 属性，并在 `multi_modal_data` 被清空后回退使用缓存属性，从而实现安全安全的 Eviction，打通了并发稳定性。

## [2026-05-19] ingest-bugfix | VLM ViT 缓存穿透与动态 resize_ratio/use_anyres 导致的不一致
- **来源**: Agent 推断与诊断会话上下文 (vlm_0519_092103_2450_verify_cache_miss_magnitude_bs128.log)
- **提取点 1**: 确定了导致 100% Cache Miss 的根本原因并非单纯的文件读写延迟，而是 `VlmDialogProcessor` 在计算 hash 时引入了基于图像分辨率动态波动的 `resize_ratio` 和 `use_anyres` 参数，使 `CityHash64` 失效。
- **提取点 2**: 明确了通过在 Actor 端 (`template.py`) 直接实例化并调用同款 `VlmDialogProcessor` 来严格提取哈希值，从而实现数学层面上的双端 100% 缓存一致性。
- **提取点 3**: 修改了缓存下发策略，废弃了手动拼凑 `use_anyres` 后缀并模拟 Hash 的做法，彻底解决了依赖硬编码导致的高频 GLOBAL MISS，同时兼容单/多图。

## [2026-05-09] ingest-bugfix | 宿主机 OOM 与 SYS 物理内存持续增长 (glibc malloc arena 碎片化)
- **来源**: Agent 推断与诊断会话上下文 (vlm_0508_190211.log)
- **提取点 1**: 确定了导致 150GB+ 物理内存黑洞的主因是 glibc 在 33+ 个并发 Python 进程下产生的 malloc arena 碎片化。
- **提取点 2**: 明确了通过设置 `MALLOC_ARENA_MAX=2` 限制多线程内存池扩张，从根本上压制 C 运行时内存冗余的解决方案。
- **提取点 3**: 总结了清理冗余引用 (`clear_stale_entries` 与 `del mb`) 以及砍掉过度并发的后台进程 (`SERVER_PROCS=16`) 的综合排雷行动。

## [2026-05-08] ingest-bugfix | PyTorch to_empty() 精度丢失与 libc 碎片化泄漏连环坑
- **来源**: Agent 推断与诊断会话上下文 (vlm_0508_190211.log)
- **提取点 1**: 确定了 `to('cpu')` 及模型重实例化在 CPU 上引发的 2GB/step 的 `libc` Arena 碎片化硬泄露问题，这会引发 Host 端的 OOM。
- **提取点 2**: 指出了 PyTorch 的 `load_state_dict` 机制会隐式跳过 `persistent=False` 的 buffer（如 ViT 的 `inv_freq`），在使用 `to_empty` 时会导致模型特征陷入完全的未初始化内存乱码状态，引起 RL 的 `logprobs_diff` 发散。
- **提取点 3**: 提炼了零 CPU 内存开辟 (`to('meta')` + `to_empty`) 配合通过 `named_buffers()` 进行底层强行内存覆盖来达成既不泄漏也不丢精度的最终修复范式。

## [2026-05-06] ingest-bugfix | empty_virt_addr_cache 导致 NPU 物理显存泄露引发 OOM
- **来源**: Agent 推断与诊断会话上下文 (vlm_0506_201101.log, oom_leak_0429_2237.md)
- **提取点 1**: 确定了在开启 `expandable_segments` 特性后，`empty_virt_addr_cache` 仅解绑虚拟地址而不释放物理块到系统底层的特性机制。
- **提取点 2**: 指出了多进程架构下（Rollout EngineWorker 与 ActorWorker），仅使用 `empty_virt_addr_cache` 会导致跨进程显存死锁并引发 ActorWorker OOM 的根因。
- **提取点 3**: 提炼了在代码底层追加 `empty_cache()` 以强制清空进程内部 PyTorch 物理缓存池从而交还驱动层的终极修复方案。

## [2026-04-22] ingest-fact | VLM 2B5 VIT与LLM的混合部署架构约束
- **来源**: Agent 推断与诊断会话上下文 (vlm_0422_142434.log / srv_dp16ep16.sh)
- **提取点 1**: 确定了主引擎 (LLM) 和伴随引擎 (ViT Encoder) 并发架构与显存物理隔离机制。
- **提取点 2**: 明确了架构只能支持 16 DP/16 EP 并行度的 NPU 硬件资源限制。
- **提取点 3**: 解析了 Unix Domain Socket 在异构进程间通讯的重要性及 veturborpc 代理关联。

## [2026-04-16] lint | health check
- **扫描**: 32 个页面 (含根目录项与模板)
- **发现**: 26 个问题 (18 死链, 7 交叉引用不对称, 1 Frontmatter缺失)
- **自动修复**: 7 个问题 (修复全部 7 个交叉引用不对称环节，强化了精度排查指南间的网状连接)
- **待人工处理**: 18 个死链（集中于 `04_frameworks` 和 `01_operators` 目录尚未建立实际内容文件的规划分类索引节点），`README.md` 缺少元数据块。


- **来源**: `raw/docs/rl_precision_case.pdf`
- **提取点 1**: 提取并归档了全视角的精度指标含义（从 `entropy` 坍缩到 `clip_ratio` 拉扯）。
- **提取点 2**: 提炼出严格控制确定性的黄金跑图 Checklist (屏蔽 Shuffle，指定 seed_all)。
- **提取点 3**: 建立了标准化五阶段问题截断防泄漏流水线追踪体系（Scale Down -> Skip Rollout 定界）。
- **提取点 4**: 单独梳理极其隐秘危险的“训推并发 Off-Policy”根因与纠偏方法，以及专用于多流算子级探查的 `npu_sanitizer` 激活技巧。

## [2026-04-16] lint | health check
- **扫描**: 22 个文件 (21 wiki 页面 + index.md)
- **发现**: 10 个问题 (6 死链, 2 交叉引用不对称, 1 Frontmatter缺失, 1 潜在缺页)
- **自动修复**: 3 个问题 (修复 2 个交叉引用不对称，补全 1 个 Frontmatter)
- **待人工处理**: 6 个死链与 1 个缺失页面，建议日后创建归档页面或作为 Todo。

## [2026-04-16] ingest-precision | 批量排查并吸收各类强化学习模型对齐偏差与精度溢出经验 (共 9 篇)
- **来源**: `raw/docs/rl_precision_case.pdf`
- **提取点 1**: `tokenizer_config` 与 `RMSNorm_eps` 缺漏引起的 Reward 分叉。
- **提取点 2**: VeRL 分布式调度以及 `moe_shared_expert_overlap` 并发时序导致的多发性梯度爆炸与踩踏错乱。
- **提取点 3**: vLLM 驱逐分叉、NPU 硬件级异构随机数生成等底层长尾缺陷引发的推理发散。
- **提取点 4**: `dummy` 加载模式下、NZ 维度重排引起的严重 Off-Policy 行为偏移以及 Tensordict 的异步脏指针。
## [2026-04-14] ingest | CANN aclnnGather 算子零长维报错与规避
## [2026-04-12] ingest | Companion ViT 启动死锁与 EncodeRejected 双根因修复

# Log (操作日志)

所有由 AI Agent 执行的操作（Ingest / Query / Lint）均按时间倒序记录在此。
格式规范：`## [YYYY-MM-DD] <operation> | <title>`，支持 `grep "^## \[" log.md | tail -5` 快速检索。

---

## [2026-05-27] ingest-tuning | EP 零冗余权重同步与 Resharding 性能优化
- **来源**: 诊断会话上下文及 5-step 测试日志 (vlm_0527_142116_2739_verify_eptrans_bs128.log)
- **提取点 1**: 确定了 16EP NPU 多卡多模态强化学习训练大 Step 更新权重时的通信与显存瓶颈为 MoE 专家权重的全量 Broadcast。
- **提取点 2**: 明确了通过 PyTorch 原生 `dist.scatter` 进行零冗余同步（ZeroSync），以及基于共享内存（SHM）“先合后分”以动态适配 `train_ep != infer_ep` 的自适应 Resharding 优化方案。
- **提取点 3**: 提炼出开启 `MAAS_XLLM_EP_ZERO_REDUNDANT_SYNC=1` 的调优步骤，实测在 bs=128 下权重同步平均耗时缩短 1.86秒（3.36%），稳态单步耗时缩短 6.65秒（11.26%），并将数据接收缓冲区压缩至 1/16，彻底消除了权重同步瞬时显存分配引发的连环 OOM。

## [2026-05-26] ingest-precision | VLM 2.5B RL logprobs_diff 长线漂移精度对齐
- **来源**: 诊断会话上下文及调试计划日志 (logpdiff_drift_debug_plan.md)
- **提取点 1**: 确定了长线训练中训推一致性 `logprobs_diff` 漂移的主要根因在于 rollout 侧 KV Cache/Prefix Cache 的跨请求污染，它无视了环境的禁用设置而被悄悄打开并复用。
- **提取点 2**: 明确了次要主因和误差放大器为 `thinking_parser` 缺失强制前缀约束，导致模型策略滑向 no-think / 乱码前缀，超短不规则文本极易放大打分噪声。
- **提取点 3**: 提炼出消除 prefix cache 请求级覆盖并锁死 `cached_length` 恒为 0，同时在 `run.sh` 中显式硬配置 thinking_parser 强制 `<think>` prefill 的对齐修复方案。在 124 step 长跑下 diff 稳定在 0.02x 健康水平。

## [2026-05-20] ingest-bugfix | xLLM 并发压测下 EncoderResultCache 驱逐触发 AssertionError
- **来源**: Client 并发 stress 压测与服务端日志 (dp16ep16_srv_*.log)
- **提取点 1**: 确定了在高并发压测下，Eviction 机制由于访问已被 `finish_query` 清空了 `multi_modal_data` 的 `ImageEncoderQuery` 的 `mm_token_length` 与 `mm_data_hash_id` 属性，导致触发 `AssertionError` 崩溃。
- **提取点 2**: 明确了通过在创建时持久化 `token_len` 和 `hash_id` 属性，并在 `multi_modal_data` 被清空后回退使用缓存属性，从而实现安全安全的 Eviction，打通了并发稳定性。

## [2026-05-19] ingest-bugfix | VLM ViT 缓存穿透与跨边界 Hash 不一致
- **来源**: Agent 推断与诊断会话上下文 (vlm_0519_172215_2450_verify_cache_miss_magnitude_bs2.log)
- **提取点 1**: 确定了导致 ViT 缓存部分穿透 (Actor 报 `GLOBAL MISS`) 的根因是跨边界的 Hash 标尺不一致。Rollout (xLLM) 生成并存入缓存的是请求 ID 作为 CityHash64，而 Actor 在重算 Hash 时生成了不同的长整型 ID，导致去 `dev/shm` 查询失败。
- **提取点 2**: 明确了通过在 Actor 端 (`template.py`) 截获 `VlmDialogProcessor` 生成的 `image_hashes[0]`，并通过 `image_raw_hashes` 将其透传至 `_inject_vit_cache` 来彻底打通两端 Hash 锁的修正方案。
- **提取点 3**: 修改了 Rollout `xpu_inferencer.py` 获取缓存 Key 的逻辑，使其优先通过 `gen_ctx.request_ids[0]` 读取 Actor 对齐好的 `CityHash64`，实现了稳定的 `GLOBAL HIT Bypass` 效果。

## [2026-05-18] ingest-bugfix | VLM ViT 缓存穿透与 CityHash 跨端不对齐导致的 100% Cache Miss
- **来源**: Agent 推断与诊断会话上下文 (vlm_0518_223521_2450_verify_cache_miss_magnitude_bs2.log)
- **提取点 1**: 确定了导致 ViT 缓存 100% 穿透的根因是跨进程哈希算法不匹配，Rollout 使用 19位 `CityHash64` 而 Actor 默认使用了 39位 `CityHash128`。
- **提取点 2**: 指出了多进程缓存未命中不仅带来冗余计算，其重算耗时差异会引发 FSDP `AllGather` 超时死锁，且软哈希兜底会引入 `1 ULP` 精度底噪干扰 `logpdiff`。
- **提取点 3**: 总结了通过在 `template.py` 中绕过默认哈希逻辑，精确复刻 Rollout 的 `CityHash64` 公式，并将正确哈希列表沿 Batch 透传至 `actor.py` 内部展平消费，最终达成 100% 精确命中的终极修复方案。

## [2026-05-16] ingest-bugfix | VLM RL 训练中 thinking_parser 缺失导致 logpdiff 极端 Spike (0.3+)
- **来源**: Agent 推断与诊断会话上下文 (vlm_0516_113330_229_nothink_data_replay_bs2.log)
- **提取点 1**: 确定了 `logprobs_diff` 在极短回复且没有 `<think>` prefill 时会出现极端放大的根因：Rollout 和 Actor 的 Attention 在超短前向传播中存在严重结构分布发散。
- **提取点 2**: 指出了 xLLM 引擎的 `CustomModeReasoningParser` 因配置为空而被禁用的级联失效过程，并在 `run.sh` 中强行注入开启配置（`enabled=True`, `prefix='<think>'`）。
- **提取点 3**: 提炼了规避 RL Trajectory 引擎将 base64 图片替换为 `..hidden..` 机制的方法，通过开发专用脚本直接从原 `parquet` 提取数据以解决数据重放 DataLoader discard 报错问题。

## [2026-04-27] ingest-fact | Fact: VLM 2.5B RL NPU Rollout 阶段的真实耗时约束
- **来源**: 用户痛点反馈及端到端耗时日志 (`outputs/logs/vlm_bs_128_0426_203707.log`)
- **提取点 1**: 提取并归档了 `bsz=128` 并且 `generation_n=8` 时产生 1024 个请求的长序列 Rollout，其物理耗时约为 54 分钟（总体 Step1 耗时约 60 分钟），严禁将其与 ST 测试混为一谈导致错误预估。
- **提取点 2**: 明确了系统架构与调度在不发生 OOM 的健康状态下，计算密集型的等待时间是合理的。
- **提取点 3**: 确立了在端到端验证时，不能仅仅通过请求堆积速度臆测完成时间，必须以日志 `global_step=1` 或 `train/timing/data` 指标为唯一衡量标准的铁律。

## [2026-04-24] lint | health check
- **扫描**: 28 个 Wiki 页面 (包含分类目录)
- **发现**: 18 个死链/缺页，未发现孤立页面及 Frontmatter 缺失
- **自动修复**: 2 个 (补充了缺失的 `wiki/05_fact/index` 页面，剔除了内部脚本误报)
- **待人工处理**: 16 个死链。建议优先为高频引用的 `vllm_npu` 和 `overflow_handling` 创建事实与调优经验页。

## [2026-04-24] ingest-bugfix | OOM 导致 MoeDistributeDispatchV2 算子死锁与集群崩溃
- **来源**: 诊断会话上下文及报错日志 (`vlm_bs_16_0423_222341.log`, `plog.err`)
- **提取点 1**: 追踪到因 Rollout 阶段高并发导致的单卡物理显存分配失败，引发算子级死锁。通过比对 `plog` 首个 OOM (23:34) 与应用层最终退出 (23:52) 的 18 分钟时间差，印证了算子挂起后触达默认超时上限的连环崩溃因果。
- **提取点 2**: 提炼了减半 batch_size 以降低总负载，并在 PyTorch 层开启 `expandable_segments:True` 缓解碎片化的有效手段。
- **提取点 3**: 澄清了算子死锁仅为 OOM 的表象与下游影响，根本原因是共用物理 NPU 的分时复用架构。

## [2026-04-22] ingest-bugfix | VIT 接入经验总结 (Companion RPC 连接失败)
- **来源**: 诊断会话上下文及报错日志
- **提取点 1**: 追踪到因环境变量丢失导致 RPC 将 unix 错误解析为 TCP 且带有端口号 `:0` 的报错逻辑。
- **提取点 2**: 提炼了针对 `veturborpc` Proxy 的正确映射方案。
- **提取点 3**: 提炼了在 ark server 中回填继承 `MAAS_MODEL_DIR` 的解决路径。

## [2026-04-21] lint | health check
- **扫描**: 31 个页面 (包含所有分类目录与模板)
- **发现**: 14 个死链 (大抵为早期规划好的未来分类索引节点或暂未成文的主题), 3 对交叉引用不对称
- **自动修复**: 3 个问题 (补全了 `vlm_rl_actor_rope_logprobs_diff.md` 及 `vlm_2b5_rl_rollout_quantization.md` 相关的双向反演引用链接)
- **待人工处理**: 14 个待建设页面 (如 `vllm_npu.md`, `xllm_pydantic_mapping.md` 等占位符)

## [2026-04-21] ingest-fact | VLM 2.5B RL 低精度 Rollout 量化实现 (D_W8A8C8)
- **来源**: Agent 推断与底座代码解析 (`linear_dy_w8a8.py` 和 `quant_conf.py`)
- **提取点 1**: 确定了系统采用 SmoothQuant 加上 Dynamic W8A8 (D_W8A8C8) 的架构设计。
- **提取点 2**: 解析了权重 (Weight) 是离线静态且 Per-Channel 量化的。
- **提取点 3**: 解析了激活 (Activation) 是包含静态平滑因子预处理的在线动态量化，且提取极值粒度为 Per-Token。
- **提取点 4**: 补全了底层利用 `xpu_ops.smooth_quant` 现场换算与截断的核心代码实现。

## [2026-04-21] ingest-precision | VLM RL 长序列 RoPE 偏移导致 logprobs_diff 发散 (400x vs 20x)
- **来源**: Agent 推理会话及 `2b5_llm_0.1_logpdiff.md`
- **提取点 1**: 确定了 `logprobs_diff` 在长序列上误差飙升的规律，直接定位到底层物理位移的 RoPE 放缩误差。
- **提取点 2**: 揭示了 `ModelSpec` 隐式优先加载 `cruise_cli.yaml` 导致 `rope_scale=400` 残留的幽灵架构坑。
- **提取点 3**: 提炼出在不破坏 Actor (BF16) 训练回传机制的前提下，使用 Python 热补丁外科手术式改写 NPU 远程参数的设计方案。

## [2026-04-12] ingest-fact | VLM 2B5 推理服务化部署黄金配置归档
- **来源**: `srv_dp16ep16.sh` 完整环境变量 + `xllm_client_vlm.sh` 3 轮 16 并发压测通过记录
- **提取点 1**: 锁定黄金版本 Commit ID 三元组（主仓库 `7e680632` / xllm `2ed41d27` / xpu_gpt `82991346`），供后续回退对照。
- **提取点 2**: 梳理 LLM 主引擎（EP16/TP1/W8A8）和 Companion ViT 引擎（16 DP/同步/frozen）的完整并行度配置。
- **提取点 3**: 详细记录 gRPC + unix socket 通信协议的正确配置组合，标注 3 类致命错误配置及其后果。
- **提取点 4**: 归纳服务就绪判定标准和客户端验证流程。

## [2026-04-12] init | 增加 ingest-fact 工作流
- **来源**: 用户需求
- **提取点 1**: 增加了 `ingest-fact.md` 以用于归档系统架构、设计约束等纯事实类知识。
- **提取点 2**: 创建了配套的 `fact_template.md` 模板。
- **提取点 3**: 更新了 `index.md` 提供入口及事实类知识分类。

## [2026-04-12] ingest | VLM Companion Encoder gRPC 路由死锁与地址错配
- **来源**: 测试会话日志 (`srv_dp16ep16.sh` 及 `xllm_client_vlm.sh`)
- **提取点 1**: 剖析了因 NPU 下 Fork 环境限制导致 `veturborpc` 获取不到套接字并引发 20004 错误的死锁本质。
- **提取点 2**: 完整梳理了将 Companion Encoder 切换至 `grpc` 之后发生的地址漂移与路由表格式错配冲突。
- **提取点 3**: 记录了通过对齐服务端(`driver_proxy`)与客户端(`routing_table`)关于 `unix` Scheme 地址拼接格式来彻底根除 `EncodeRejected` 的架构层修复手法。

## [2026-04-11] ingest | xLLM 环境映射机制与 veturborpc 拦截
- **来源**: 本地推演 (`srv_dp16ep16.sh` 及日志)
- **提取点 1**: 追踪并记录了 `veturborpc` 报错的根本原因是单 rank 的 Companion 服务隐式继承了 Launcher 的全局 RPC 配置。
- **提取点 2**: 明确了利用 `XLLM_ENCODER_METHOD` 而非 `REMOTE_EXECUTOR_METHOD` 来绕过 `InferencerConfig` 强校验的方法。

## [2026-04-11] lint | health check
- **扫描**: 6 wiki 页面 + index.md, 共 7 项检查
- **发现**: 15 个问题 (8 死链/缺页, 2 交叉引用不对称, 5 缺 frontmatter)
- **自动修复**: 7 个 (5 index 页补 frontmatter, 1 占位死链删除, 1 交叉引用修复)
- **待人工**: 8 个缺页 (规划型链接，待日后 Ingest 填充)

## [2026-04-11] ingest | VLM RL logprobs_diff 发散根因分析
- **来源**: 本地工作汇报 (`vlm_rl_root_cause_analysis.md`)
- **提取点 1**: 定位因推理引擎底层的防错篡改，导致 RL 流程中特殊 token `-100` 全部变成 `1` 的幽灵 Bug。
- **提取点 2**: 记录下利用 `image_token_indices` 指纹定点打补丁，恢复 `Actor` 图像感知特征的手段。
- **提取点 3**: 记录了 `logprobs_diff` 从 `0.58` 断崖下降恢复到 `0.02` 以满足多模态强化学习的验收标准。

## [2026-04-11] init | 初始架构构建
- **提取点 1**: 构建项目核心目录结构 (`wiki/`, `raw/`, `templates/`)。
- **提取点 2**: 初始化 `.antigravityrules` 角色与工作流指令。
- **提取点 3**: 建立 `index.md` 全局知识图谱入口。
