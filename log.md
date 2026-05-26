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
