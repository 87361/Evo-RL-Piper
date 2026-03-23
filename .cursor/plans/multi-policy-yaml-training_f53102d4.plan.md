---
name: multi-policy-yaml-training
overview: 在保持真实 PI 训练链路（`train_pi.py -> third_party/openpi`）的前提下，恢复并落地 `training_repo` 多策略工厂方案，支持 `openpi/act/diffusion` 的统一 YAML 入口。
todos:
  - id: define-unified-yaml-schema
    content: 定义并落地统一多策略训练 YAML schema（openpi/act/diffusion）
    status: completed
  - id: refactor-orchestrator-factory
    content: 重构 orchestrator，接入 backend/policy 工厂分发
    status: completed
  - id: add-lerobot-backend-adapter
    content: 新增 training_repo 到 lerobot 的后端适配器
    status: completed
  - id: add-unified-train-script
    content: 新增统一训练入口脚本 train_policy.py
    status: completed
  - id: tests-and-docs
    content: 补充分发与配置校验测试，并更新训练文档
    status: completed
isProject: false
---

# 统一多策略 YAML 训练入口改造计划（已恢复并实现）

## 当前状态（2026-03-17）

当前仓库已进入“双入口并存”状态：

- 真实 PI 训练入口（保持）：
  - `scripts/train_pi.py`
  - 转发到 `third_party/openpi/scripts/train.py`（JAX）或 `third_party/openpi/scripts/train_pytorch.py`（Torch）
- 多策略工厂入口（新增）：
  - `scripts/train_policy.py`
  - `src/training_repo/train/backend_factory.py`
  - `src/training_repo/backends/openpi_backend.py`
  - `src/training_repo/backends/lerobot_backend.py`
  - 支持通过 YAML 分发 `openpi/act/diffusion`

## 完成结果

- 已完成 `training_repo` 层配置驱动分发：
  - `backend=openpi|lerobot` 显式分发
  - 或按 `policy.type` 自动推断（`act/diffusion/pi0/pi05` -> `lerobot`）
- 已保留旧 `train_pi` 配置兼容：
  - 顶层 `backend=openpi_torch/openpi_jax` + `config_name` 仍可运行
- 已提供统一配置样例（用于 `scripts/train_policy.py` 工厂入口）：
  - `configs/train_policy_openpi.yaml`
  - `configs/train_policy_act.yaml`
  - `configs/train_policy_diffusion.yaml`
- 已补齐最小单测与文档更新。

## 与文档口径对齐（`docs/260317.md` 与 `docs/dataset-build-min-loop-v0.md`）

- 真实 PI 官方训练链路保持不变：
  - 最小 smoke 仍按 `docs/dataset-build-min-loop-v0.md` 运行 `scripts/train_pi.py --config configs/train_pi0_openpi.yaml ...`
- `scripts/train_policy.py` 的定位是 `training_repo` 多策略工厂入口：
  - 通过统一 YAML 在 `openpi/act/diffusion` 间切换
  - 不替代 `scripts/train_pi.py` 的官方 OpenPI 训练入口

