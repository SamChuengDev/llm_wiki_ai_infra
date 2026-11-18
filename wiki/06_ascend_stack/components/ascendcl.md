---
title: AscendCL 底层计算接口详解
tags:
  - AscendCL
  - ACL
  - CANN
  - NPU
created: 2026-08-05
updated: 2026-08-05
last_updated: 2026-08-05
source: ["migrated from ascend-llm-wiki"]
---

# AscendCL 底层计算接口详解

## 1. AscendCL 概述

AscendCL（Ascend Computing Language）是华为昇腾 AI 处理器在 CANN 软件栈中的底层计算接口库，提供面向 NPU 的设备管理、内存管理、算子执行等核心能力，是 CANN 中最高层级的 C 语言 API。

## 2. 核心能力

- **设备管理（Device）**：设备的初始化、复位、获取设备信息等
- **上下文管理（Context）**：Context 的创建、绑定、切换、销毁，用于管理算力资源
- **内存管理（Memory）**：NPU 显存分配/释放、内存拷贝（Host↔Device）、内存置零等
- **Stream/Event 管理**：异步任务队列管理，支持 Stream 同步、Event 创建与等待
- **算子执行（rtKernelLaunch）**：通过 `rtKernelLaunch` 等底层 API 在 NPU 上执行自定义算子

## 3. 适用场景

- 适合简单算子、需要精细控制算力资源的场景
- 对内存和执行流有特殊要求的自定义算子开发

## 4. 在 OpPlugin 中的使用

OpPlugin 中 `aclops/` 目录下的算子直接调用 AscendCL API 实现，适用于那些不需要高层抽象、追求极致控制力的场景。

## 5. 与 aclnn 的关系

[[components/aclnn]] 基于 AscendCL 封装，提供更高层的抽象，包括自动内存管理、类型推导、shape 推导等能力，使算子开发更加便捷。