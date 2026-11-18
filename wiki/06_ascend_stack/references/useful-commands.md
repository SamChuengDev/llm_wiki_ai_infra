---
title: 常用命令速查表
tags:
  - commands
  - cheatsheet
  - reference
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# 常用命令速查表

## 1. 硬件状态检查

```bash
dmesg | grep 'ras event'
```
查看当前启动周期 RAS 事件。

```bash
journalctl -k | grep ras | tail -20
```
查看最近 RAS 事件。

```bash
watch -n 60 'dmesg | grep "ras event" | tail -5'
```
实时监控 RAS 事件。

更多硬件诊断相关内容请参考 [[troubleshooting/hccl-errors]]。

## 2. NPU 状态检查

```bash
npu-smi info
```
查看 NPU 健康状态。

```bash
npu-smi info -t device -i 0
```
查看设备详细信息。

## 3. 进程管理

```bash
ps aux | grep train
```
查找训练进程。

```bash
kill -9 <PID>
```
强制终止进程。

## 4. 日志查看

- **设备日志目录**: `/var/log/ascend_log/device_log/`
- **实时查看训练日志**: `tail -f /tmp/fp8_118_v3.log`

## 5. 文件传输

```bash
scp local_file root@remote:/path/
```
上传文件到远程机器。

```bash
scp root@remote:/path/file ./
```
从远程机器下载文件。

## 6. SSH 登录

```bash
ssh root@141.62.17.118
```
登录远程机器。