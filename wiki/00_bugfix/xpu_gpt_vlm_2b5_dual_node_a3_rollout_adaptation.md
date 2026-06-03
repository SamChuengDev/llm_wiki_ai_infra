---
title: "Bugfix: xpu-gpt 2B5 VLM 双节点 A3 rollout 适配"
tags: [status/validated, priority/high, category/bugfix, npu/a3, backend/xpu_gpt, rl/2b5]
last_updated: 2026-06-03
source:
  - outputs/logs/vlm_0603_115505_2614_xg_graph_perfopt_bs128.log
  - outputs/logs/vlm_0603_130806_xg_2nodes_2b5_bs128.log
  - outputs/logs/vlm_0603_153936_xg_2nodes_2b5_bsz256_bs256.log
  - outputs/run_time_envs/runtime_env_0603_153936_xg.yaml
confidence: high
status: active
open_questions:
  - "bsz=256 已验证到 early steps，仍需继续观察 10 step 目标以及 70+/200 step 长线 logprobs_diff 漂移。"
contradictions: []
---

# Bugfix Report: xpu-gpt 2B5 VLM 双节点 A3 rollout 适配

## 1. 现象描述 (Symptoms)

- **环境**: 2450 + 26100 双节点 A3，每节点 16 NPU；Ray head `10.101.24.50`，worker `10.101.26.100`；容器分别为 `mass_share` 与 `maas_rl_26_100_debug`。
- **目标配置**: xpu-gpt / W8A8 rollout，2B5 VLM RL；actor 使用 32rank，rollout 按节点形成两个 16-rank replica；入口为 `ray_xg_32rank.sh` + `run_xg_vlm_2b5.sh` + 统一 `runtime_env_xg.yaml`。
- **早期失败 1: Ray 版本不一致导致 worker 无法入集群**:
  ```text
  RuntimeError: Version mismatch: The cluster was started with:
      Ray: 2.46.0
      Python: 3.11.9
  This process on node 10.101.26.100 was started with:
      Ray: 2.47.1
      Python: 3.11.9
  ```
- **早期失败 2: rollout replica 计算被非 alive / 无设备 Ray node 污染**:
  ```text
  assert replicas_per_node != 0
  AssertionError
  ```
  当时 `ray.nodes()` 里可见一个 `('10.101.26.100', False, None)` 的残留节点。旧逻辑遍历所有 Ray nodes，未过滤非 Alive 节点，导致 rollout replica 统计可能落到 0 或被无设备节点扰动。
- **隐性问题: 双节点 rollout 的 xLLM rendezvous 地址不能继续固定 head 地址**。每个 16-rank rollout replica 在各自节点本地启动 xLLM 服务，`XLLM_INFERENCER_torch_distributed_addr` 需要按 Ray node 本机 IP 设置，否则 worker 节点 replica 会尝试用 head 地址做本地分布式 rendezvous，容易形成跨节点错配。

## 2. 根本原因分析 (Root Cause)

本次问题不是单一异常，而是双节点化后暴露的三类适配缺口：

1. **集群层版本必须一致**：Ray head/worker 的 Ray 版本不一致时，worker 会在 `ray start --address=...` 阶段被拒绝加入。这是直接由 Ray traceback 证明的环境问题。
2. **Ray node 枚举必须只使用 alive device nodes**：训练资源池和 rollout replica 枚举不能把 `Alive=False` 或无 `NPU/GPU` 资源的节点算进去。失败日志中的 `assert replicas_per_node != 0` 与 `ray.nodes()` 残留 dead node 共同支持该判断。
3. **多节点 xLLM rollout replica 需要节点本地 rendezvous 地址**：双节点下每个 rollout replica 占用本节点 16 NPU；如果沿用单节点 runtime env 的 head 地址，worker 节点 replica 的本地 distributed init 地址不再匹配。修复后日志明确打印两个 replica 分别设置为 `10.101.24.50` 和 `10.101.26.100`。

## 3. 解决方案 (Solution)

### 临时方案 (Workaround)

- 确认 2450 与 26100 容器内 Ray / Python 版本一致后再组集群。
- 启动前检查 Ray cluster 只保留两个 alive device nodes：
  ```bash
  python3 -c "import ray; ray.init(address='127.0.0.1:6379'); print([(n['NodeManagerAddress'], n['Alive'], n['Resources'].get('NPU')) for n in ray.nodes()])"
  ```
- 历史开关 `MAAS_XG_ROLLOUT_USE_NODE_IP=1` 可强制 xLLM rollout 使用本机 IP，但最终不应要求用户维护额外 runtime env 分叉。

### 最终修复 (Fix)

- `maas_finetune/trainer/rl/ray_utils.py`
  - 新增 alive device node 过滤，只保留 `Alive=True` 且存在 `NPU/GPU` 资源的节点。
  - `get_global_resource_pool()`、`get_ray_world_size()`、`get_ray_node_rank()` 均基于过滤后的节点列表，避免 dead node 污染 actor 资源池。
- `maas_finetune/trainer/rl/rollout/ray_driver.py`
  - rollout replica 枚举跳过非 alive / 无设备节点。
  - 每个节点按自己的 `node_replicas = num_gpu / (tp_size * ep_size)` 追加 replica。
  - 将旧的 `assert replicas_per_node != 0` 改为 `assert available_replicas`，报错语义对准真实问题。
- `maas_finetune/trainer/rl/rollout/xllm_backend/engine.py`
  - 新增 `_has_multiple_alive_device_nodes()`。
  - 当 Ray cluster 中 alive device nodes 大于 1，自动将 `XLLM_INFERENCER_torch_distributed_addr` / 大写同名变量改成本机 IP。
  - 保留 `MAAS_XG_ROLLOUT_USE_NODE_IP=1` 作为兼容开关。
- `runtime_env_xg.yaml`
  - 单/双节点统一使用同一 runtime env，不再维护几乎重复的 `runtime_env_xg_32rank.yaml`。
  - 补齐 `${PROJECT_ROOT}/3rdparty/seed_inference_models`、`${PROJECT_ROOT}/3rdparty`。
  - 将 CCEC compiler bin 放入 `PATH`，避免 TBE 动态编译找不到 `llvm-objdump`。
- `ray_xg_32rank.sh` 与 `run_xg_vlm_2b5.sh`
  - 32rank 脚本改为渲染 `runtime_env_xg.yaml`。
  - `BATCH_SIZE=${BATCH_SIZE:-128}`、`TEST_NAME=${TEST_NAME:-...}`，单节点默认不变，双节点可通过环境变量覆盖。

## 4. 验证结果 (Verification)

- **配置验证**:
  ```text
  [('10.101.24.50', True, 16.0), ('10.101.26.100', True, 16.0)]
  ```
- **bsz=256 runtime env 证据**:
  ```text
  PYTHONPATH includes /data01/zcs/maas_engine_vlm2b5/3rdparty/seed_inference_models
  PATH includes /usr/local/Ascend/ascend-toolkit/8.3.T102/aarch64-linux/ccec_compiler/bin
  MAAS_XG_ROLLOUT_USE_NODE_IP not found
  ```
- **自动本机 IP rendezvous 生效**:
  ```text
  set xLLM rollout rendezvous addr to Ray node ip: 10.101.24.50
  set xLLM rollout rendezvous addr to Ray node ip: 10.101.26.100
  ```
- **bsz=128 双节点 smoke**: `outputs/logs/vlm_0603_130806_xg_2nodes_2b5_bs128.log` 已验证到 10 个 global step。
- **bsz=256 验证**: `outputs/logs/vlm_0603_153936_xg_2nodes_2b5_bsz256_bs256.log` 在 2026-06-03 16:14 仍为 RUNNING，fatal 扫描未命中 Traceback / OOM / RPC / Raylet terminated；已观察到：
  ```text
  step=1 reward=0.159668 logprobs_diff=0.0387878 step_s=472.166
  step=2 reward=0.141602 logprobs_diff=0.0287212 step_s=434.071
  step=3 reward=0.157227 logprobs_diff=0.0295647 step_s=429.499
  step=4 reward=0.138672 logprobs_diff=0.0292577 step_s=417.695
  ```

这些指标只证明双节点 A3 启动、rollout、old_logprobs、actor update、权重同步链路已经跑通 early steps；不代表 70+ / 200 step 长线精度漂移已经完成验证。

## 5. 沉淀与关联

- 关联事实: [[wiki/05_fact/vlm_2b5_32rank_actor_rollout_deployment|VLM 2B5 xpu_gpt 32rank Actor 与 Rollout 部署形态]]
- 关联调优: [[wiki/03_tuning/xpu_gpt_w8a8_rl_scheduler_memory_tuning|xpu-gpt W8A8 RL scheduler batch 与显存水位调优]]
- 关联排错: [[wiki/00_bugfix/vlm_2b5_vit_checkpoint_fallback|M8 2B5 VLM Companion ViT 权重路径 fallback 缺失导致启动失败]]
- 关键经验: 双节点适配优先做代码自适应，避免复制 runtime env；但资源枚举必须显式过滤 non-alive nodes，多节点本地服务必须显式修正 rendezvous 地址。
