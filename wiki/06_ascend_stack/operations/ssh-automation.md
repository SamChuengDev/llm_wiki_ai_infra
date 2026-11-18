---
title: 远程机器管理与自动化
tags: [SSH, remote, automation, SFTP]
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# 远程机器管理与自动化

## 1. SSH 免密登录配置

使用 `root` 用户登录远程机器，已配置免密登录，可直接通过 SSH 访问各台机器，无需手动输入密码。

## 2. SFTP 同步策略

- **MindSpeed-LLM 仓库**：通过 SFTP 插件自动同步到远程机器，无需手动操作。
- **MindSpeed 仓库**：修改后需手动通过 `scp` 上传到目标路径：

  ```
  scp -r <本地路径> root@<远程IP>:/mnt/share/z00583051/codes/MindSpeed
  ```

## 3. 远程执行命令

通过 SSH 在远程机器上直接启动训练任务：

```bash
ssh root@<远程IP> "<训练命令>"
```

## 4. 日志同步

远程训练任务的日志文件自动保存到以下路径：

```
/mnt/share/z00583051/codes/MindSpeed-LLM/logs/
```

该路径为共享存储，可在本地直接访问。

## 5. 机器列表

| 机器 | IP 地址 |
|---|---|
| 机器 1 | `141.62.17.110` |
| 机器 2 | `141.62.17.118` |
| 机器 3 | `141.61.52.75` |