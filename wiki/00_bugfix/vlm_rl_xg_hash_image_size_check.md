---
title: "Bugfix: VLM 2B5 xg RL hash 路径未关闭 xLLM 图片尺寸检查导致 InvalidImageURL"
tags: [status/resolved, priority/high, category/vlm, category/xllm, category/image, xpu_gpt, rl]
last_updated: 2026-06-01
source:
  - outputs/logs/vlm_0530_204441_2450_xg_bs128.log
  - outputs/logs/vlm_0601_093232_2450_xg_bs128.log
  - maas_finetune/trainer/rl/template.py
  - commit:039c20a3420c0e4f0870396b595d3bbb9a9b92cf
confidence: high
status: active
open_questions:
  - "该修复已验证越过旧 step19/20 崩溃点；70+ / 200 step 长线精度漂移仍需按长跑标准单独验证。"
contradictions: []
---

# Bugfix Report: VLM 2B5 xg RL hash 路径未关闭 xLLM 图片尺寸检查导致 InvalidImageURL

## 1. 现象描述 (Symptoms)

- **环境**: VLM 2B5 RL，xpu_gpt / xg rollout 后端，2450，`BATCH_SIZE=128`，16 NPU，入口为 `ray_xg.sh` / `run_xg_vlm_2b5.sh`。
- **失败阶段**: merge 后 xg 运行到约 `global_step=19`，在 DataProcessor 处理下一批样本时崩溃；torch inference 后端同阶段可以继续长跑。
- **第一处失败签名**:
  ```text
  File ".../maas_finetune/trainer/rl/dataloader.py", line 314, in _process_item
      item.data = await processor(item.idx, item.data)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../maas_finetune/trainer/rl/processor.py", line 241, in __call__
      raise result.underlying
  xllm.backend.exceptions.InvalidImageURL: Image dimensions exceed the maximum allowed limit.
  Current dimensions: width = 6840, height = 3549. Maximum allowed dimension: 6000 pixels.
  ```
- **旧失败日志中的阶段性指标**:
  ```text
  outputs/logs/vlm_0530_204441_2450_xg_bs128.log
  global_step=19
  train/rollout/logprobs_diff:0.024779
  train/reward/final_rewards.mean:0.490234
  ```

## 2. 根本原因分析 (Root Cause)

该异常不是 rollout 服务端主 VLM 请求路径的 Navit resize 失效，而是 Actor/DataProcessor 侧为了计算图片 hash / cache key 手动构造 `VLMDialogProcessor` 时，尺寸检查参数没有显式传入。

`RLTemplate._encode_image()` 旧逻辑只设置了环境变量：

```python
os.environ["XLLM_PROCESSOR_CHECK_IMAGE_SIZE"] = os.environ.get("XLLM_PROCESSOR_CHECK_IMAGE_SIZE", "0")
self.xllm_processor_hash = VLMDialogProcessor(
    tokenizer=self.tokenizer,
    max_seq_len=self.max_length,
    model_dir=_xllm_model_dir,
    model_config=config,
    roles=roles,
)
```

但 `VLMDialogProcessor.__init__` 的构造参数 `check_image_size` 默认仍为 `True`，不会因为上面的环境变量自动变为 `False`。因此 xg 训练在 Actor/hash 路径先调用 xLLM 图片读取与检查逻辑时，6840x3549 的原图被 6000 单边限制提前拒绝，尚未进入后续 VLM/Navit resize 处理。

证据：

- 2450 容器内直接验证：同一张 6840x3549 图片，`check_image_size=True` 会复现 `InvalidImageURL`，`check_image_size=False` 可通过读取。
- 代码验证：显式传入 `VLMDialogProcessor(..., check_image_size=False)` 后，实例内部 `_check_image_size` 为 `False`。
- 修复后 xg 在 2450 上越过旧的 step19/20 崩溃窗口，未再出现 `InvalidImageURL`。

## 3. 解决方案 (Solution)

### Fix

在 `maas_finetune/trainer/rl/template.py` 中增加通用 env bool 解析，并在 hash 路径构造 `VLMDialogProcessor` 时显式传入 `check_image_size`：

```python
def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

self.xllm_processor_hash = VLMDialogProcessor(
    tokenizer=self.tokenizer,
    max_seq_len=self.max_length,
    model_dir=_xllm_model_dir,
    model_config=config,
    roles=roles,
    check_image_size=_env_bool("XLLM_PROCESSOR_CHECK_IMAGE_SIZE", False),
)
```

默认值设为 `False`，与该路径前面设置的 `XLLM_PROCESSOR_CHECK_IMAGE_SIZE=0` 以及 xg rollout 侧允许大图进入后续 Navit/resize 的口径一致。若后续需要恢复严格检查，可显式设置 `XLLM_PROCESSOR_CHECK_IMAGE_SIZE=1`。

### 禁忌

- 不要为了规避该异常改小 `GENERATION_N`、`BATCH_SIZE` 或其它 RL 算法超参。
- 不要改 xpu_gpt LLM backbone 权重加载、W8A8 量化、`update_weights` 等已验证主路径。
- 不要把这个短程稳定性修复误写成 70+ / 200 step 精度漂移问题已经解决。

## 4. 验证结果 (Verification)

### 静态与单点验证

```text
python3 -m py_compile maas_finetune/trainer/rl/template.py
git diff --check -- maas_finetune/trainer/rl/template.py
```

本地与 2450 容器内均通过。2450 容器内 oversized image direct test 结果为：

```text
check_image_size=True  -> InvalidImageURL: width = 6840, height = 3549, max = 6000
check_image_size=False -> success, image size = (6840, 3549)
```

### 2450 xg RL 验证

验证日志：

```text
outputs/logs/vlm_0601_093232_2450_xg_bs128.log
Ray job: raysubmit_DHGGPKu2aY51assz
entrypoint: bash run_xg_vlm_2b5.sh 0601_093232 /data01/zcs/maas_engine_vlm2b5
```

关键指标：

```text
step=16 reward_mean=0.371094 logprobs_diff=0.0253211
step=17 reward_mean=0.461914 logprobs_diff=0.0246407
step=18 reward_mean=0.542969 logprobs_diff=0.0236723
step=19 reward_mean=0.496094 logprobs_diff=0.028423
step=20 reward_mean=0.605469 logprobs_diff=0.0238362
```

错误扫描截至 step20：

```text
Traceback: none
InvalidImageURL: none
RuntimeError/ValueError/AttributeError/TypeError: none
Raylet is terminated: none
```

结论：该修复已验证越过旧的 step19 后下一批数据处理崩溃点，并到达 `global_step=20`。该结论只覆盖 `InvalidImageURL 6840x3549 > 6000` 问题，不替代 xg 后端 70+ / 200 step 长线精度漂移验证。

## 5. 沉淀与关联

- 快速诊断方式：
  - 看到 `InvalidImageURL: ... Maximum allowed dimension: 6000 pixels` 时，先用 `head` 定位第一处 Traceback，确认是否发生在 `dataloader.py::_process_item` / `processor.py::__call__`。
  - 若异常发生在 Actor/DataProcessor 阶段，优先检查 `RLTemplate._encode_image()` 中手动构造的 `VLMDialogProcessor` 参数，而不是先怀疑 rollout 服务端 Navit resize。
  - 对于大图问题，分别验证 `check_image_size=True/False` 的行为，避免把 env 设置误认为已经传递到构造参数。
- 关联知识点:
  - [[wiki/00_bugfix/vit_cache_miss_cityhash|Bugfix: ViT Cache 100% Miss & Race Condition Mismatch]]
  - [[wiki/00_bugfix/vlm_cache_hash_mismatch|Bugfix: VLM Cache Miss Due to Cross-Boundary Hash Mismatch]]
  - [[wiki/00_bugfix/vlm_torch_infer_vit_cache_sp_slice_logpdiff|Bugfix: torch infer cached ViT embedding 被 SP 误切导致 logprobs_diff 爆炸]]
  - [[wiki/02_precision/vlm_2b5_logpdiff_drift|VLM 2.5B RL logprobs_diff 长线漂移精度对齐]]
