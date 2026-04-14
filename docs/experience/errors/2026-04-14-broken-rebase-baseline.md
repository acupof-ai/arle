# 2026-04-14 · Broken Rebase Baseline

## Context

在 CUDA crate 抽取过程中，曾默认把 `origin/main` 当作可编译基线来判断“剩余问题是否由当前改动引入”。

用户指出：rebase 已经把一批不完整的路径迁移带进来了，`origin/main` 本身就可能处于“目录已搬、依赖和引用尚未收口”的中间态。

## Root Cause

把“当前分支能否编译”和“上游基线是否完整”混为一谈了。

具体失误是：
- 没先核对 `HEAD` / `origin/main` / 目标好基线 commit 的差异
- 没先验证上游基线本身是否可编译
- 直接把残留引用和缺失依赖默认归因到当前改动批次

## Fix

后续凡是遇到 rebase / cherry-pick / 路径迁移场景，先做三步：

1. `git rev-parse --short HEAD origin/main <known-good>` 明确比较对象
2. 先检查上游基线本身是否含有目标文件、依赖和模块声明
3. 只有在基线干净时，才把剩余编译错误归因到当前改动

## Rule

路径迁移类重构里，绝不默认 `origin/main` 是健康基线。先验证基线，再归因问题。
