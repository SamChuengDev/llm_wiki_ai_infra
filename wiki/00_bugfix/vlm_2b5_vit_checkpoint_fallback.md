---
title: "Bugfix: M8 2B5 VLM Companion ViT 权重路径 fallback 缺失导致启动失败"
tags: [status/resolved, priority/high, category/model-loading, vlm, xpu-gpt]
last_updated: 2026-06-01
source: [outputs/logs/vlm_0601_161331_2450_after_merge_ge_graph_bs128.log, outputs/logs/vlm_0601_165516_2450_after_merge_bs128.log, 3rdparty/xpu_gpt/xpu_gpt/adapter/xllm_adapter/xpu_loader.py]
confidence: high
status: closed
open_questions:
  - "本条只覆盖 ViT 权重加载启动失败；不代表 70+ step / 200 step 长线精度漂移验证已经完成。"
contradictions: []
---

# Bugfix Report: M8 2B5 VLM Companion ViT 权重路径 fallback 缺失导致启动失败

## 1. 现象描述 (Symptoms)
- **环境**: M8 2B5 VLM RL, 2450, xpu_gpt backend, Companion ViT, `run_xg_vlm_2b5_before_merge.sh`, `runtime_env_xg.yaml`, `BATCH_SIZE=128`
- **失败日志**: `outputs/logs/vlm_0601_161331_2450_after_merge_ge_graph_bs128.log`
- **报错信息**:
  ```text
  2026-06-01 16:15:52.020 ... serve.py:415 [ERROR]: [Companion] ArkLauncher launch failed
  Traceback (most recent call last):
    File "/data01/zcs/maas_engine_vlm2b5/3rdparty/xllm/xllm/service/rpc/ark/serve.py", line 410, in launch_driver
      self._companion_launcher = ArkLauncher(companion_model_dir, self._companion_driver_type, yaml_file)
    File "/data01/zcs/maas_engine_vlm2b5/3rdparty/xpu_gpt/xpu_gpt/adapter/xllm_adapter/xpu_loader.py", line 139, in build_inferencer_model_config
      return self._build_vit_model_config(model_conf, sched_conf, parallel_conf, driver_conf, extra_conf)
    File "/data01/zcs/maas_engine_vlm2b5/3rdparty/xpu_gpt/xpu_gpt/adapter/xllm_adapter/xpu_loader.py", line 118, in _build_vit_model_config
      vit_model_path = self._resolve_vit_checkpoint(vit_model_dir)
    File "/data01/zcs/maas_engine_vlm2b5/3rdparty/xpu_gpt/xpu_gpt/adapter/xllm_adapter/xpu_loader.py", line 98, in _resolve_vit_checkpoint
      raise ValueError(f"ViT model not found in {checkpoint_dir}; supported files: vit.safetensors, vit.pt")
  ValueError: ViT model not found in /share_data/2b5_saved_model_for_xg/saved_model/global_step_0; supported files: vit.safetensors, vit.pt
  ```

## 2. 根本原因分析 (Root Cause)
这次启动失败的直接原因是 Companion ViT 的 checkpoint 解析逻辑只在 `saved_model/global_step_0/` 这一层查找 `vit.safetensors` / `vit.pt`，没有继续 fallback 到上层 `saved_model/` 或模型根目录。

从失败日志可见，LLM 权重 `model.safetensors` 能在 `/share_data/2b5_saved_model_for_xg/saved_model/global_step_0` 被找到，但 ViT 权重查找在同一层失败后立即抛错：

```text
checking subfolder: /share_data/2b5_saved_model_for_xg/saved_model/global_step_0 to find model file
find: model.safetensors under /share_data/2b5_saved_model_for_xg/saved_model/global_step_0
ValueError: ViT model not found in /share_data/2b5_saved_model_for_xg/saved_model/global_step_0
```

这不是 LLM backbone 权重加载问题，也不是 CANN/NPU 算子问题；失败发生在 xpu_gpt 的 Companion ViT model config 构建阶段。

## 3. 解决方案 (Solution)
### 临时方案 (Workaround)
- [x] 如果需要绕过目录解析，可显式设置 `XLLM_VIT_MODEL_PATH` 或 `VIT_MODEL_PATH` 到真实的 `vit.safetensors` / `vit.pt` 文件。

### 终极方案 (Fix)
- [x] 修改 `3rdparty/xpu_gpt/xpu_gpt/adapter/xllm_adapter/xpu_loader.py::_resolve_vit_checkpoint()`，保留显式环境变量优先级，并增加 VIT-only fallback 查找顺序：
  ```text
  saved_model/global_step_*/vit.safetensors
  saved_model/global_step_*/vit.pt
  saved_model/vit.safetensors
  saved_model/vit.pt
  <vit_model_dir>/vit.safetensors
  <vit_model_dir>/vit.pt
  ```
- [x] 对 `saved_model/` 下多个 `global_step_*` 的情况显式报错，避免静默选错 checkpoint。
- [x] 修改范围仅在 Companion ViT 模型配置路径，未改动 LLM backbone 的 `replace_maps`、`concat_maps`、`update_weights` 等已验证权重加载核心逻辑。

## 4. 验证结果 (Verification)
验证日志 `outputs/logs/vlm_0601_165516_2450_after_merge_bs128.log` 显示，同一类 xpu_gpt + Companion ViT 启动路径已越过原始失败点：

```text
2026-06-01 16:58:32.458 ... xpu_loader.py:131 [INFO]: [NPU] Use ViT cruise config: /share_data/2b5_saved_model_for_xg/cruise_cli.yaml, checkpoint: /share_data/2b5_saved_model_for_xg/saved_model/global_step_0/vit.safetensors
2026-06-01 16:58:52.191 ... serve.py:413 [INFO]: [Companion] ArkLauncher launch done
```

随后训练进入多轮 global step，说明该启动失败已解除：

```text
Training. global_step=1 ... train/reward/final_rewards.mean:0.157227, train/rollout/logprobs_diff:0.0362214
Training. global_step=10 ... train/reward/final_rewards.mean:0.200195, train/rollout/logprobs_diff:0.0237848
Training. global_step=20 ... train/reward/final_rewards.mean:0.587891, train/rollout/logprobs_diff:0.0234008
```

注意：这些日志只能证明 ViT 权重加载失败和 Companion 启动失败已被修复，并提供阶段性精度指标；不能替代 70+ step / 200 step 长线漂移验证。

## 5. 沉淀与关联
- 关联知识点: [[wiki/05_fact/vlm_2b5_vit_llm_deployment_architecture|VLM 2B5 VIT与LLM的混合部署架构约束]]
- 关联排错: [[wiki/00_bugfix/vlm_companion_vit_rpc_crash|VLM Companion ViT RPC 连接失败与环境变量断层]]
- 关联排错: [[wiki/00_bugfix/xpu_gpt_decode_graph_cache_maxlen_mismatch|xpu_gpt decode graph cache 未按 max_length 隔离导致 12k->32k 图模式断言]]
