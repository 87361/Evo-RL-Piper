---
name: training-repo-v0-min-loop
overview: 定义与 TeleManipulation 解耦的训练仓库 v0 最小闭环：从原始数据整理到训练，再到可复制交付的推理包。仅覆盖 offline + OpenPI 后端 + APO最小思想。
todos:
  - id: v0-data-loop
    content: 实现 ingest/relabel/dataset_build 并产出最小训练数据包
    status: completed
  - id: v0-openpi-train
    content: 实现 OpenPI adapter 与训练入口，完成首次离线训练
    status: completed
  - id: v0-export-bundle
    content: 实现 inference bundle 导出与可复制交付结构
    status: in_progress
  - id: v0-delivery-smoke
    content: 完成 TeleManipulation 侧模拟接入与 smoke 验证
    status: pending
isProject: false
---

TODO:实现 inference bundle 导出与可复制交付结构
完成 TeleManipulation 侧模拟接入与 smoke 验证

# 训练仓库 v0 最小闭环（拍板版）

## A. 训练仓库 v0 的最小模块划分

- `ingest`：接收采集侧导出的原始 episode（只做离线读取，不接真机）。
- `relabel`：实现 APO 最小逻辑（3 类样本重标记 + K-step pre-intervention 回溯）。
- `dataset_build`：导出训练后端可直接消费的数据包（index/labels + split + stats + manifest，不固化 batch 配比）。
- `backend_openpi`：OpenPI 训练适配层（把内部 schema 映射到 OpenPI 输入契约）。
- `train`：训练入口与实验产物落盘（ckpt、metrics、config 快照）。
- `export`：导出推理代码包 + 配置 + 权重路径约定（给 TeleManipulation 直接拷贝）。

## B. repo 最小目录结构

- [docs/](docs/)：方案、约定、交付说明（当前结论文档落这里）。
- [configs/](configs/)：数据构建配置、训练配置、导出配置。
- [src/training_repo/ingest/](src/training_repo/ingest/)：原始数据扫描与读取。
- [src/training_repo/relabel/](src/training_repo/relabel/)：APO 最小重标记逻辑。
- [src/training_repo/dataset_build/](src/training_repo/dataset_build/)：样本索引生成、bucket/label 写出、train/val 划分、归一化统计、产物导出。
- [src/training_repo/backend_openpi/](src/training_repo/backend_openpi/)：OpenPI dataset adapter + launcher。
- [src/training_repo/train/](src/training_repo/train/)：训练 orchestrator。
- [src/training_repo/export/](src/training_repo/export/)：推理包生成器。
- [deploy/inference_bundle/](deploy/inference_bundle/)：最终给 TeleManipulation 拷贝的目录模板。
- [scripts/](scripts/)：`build_dataset.py` / `train_openpi.py` / `export_inference_bundle.py`。
- [tests/](tests/)：仅保留最小单测与端到端冒烟测试。

## C. 当前必须的最小 schema

- **Episode Manifest（episode 级）**
  - `episode_id`, `task_id`, `source_path`, `num_steps`, `success`。
- **Dataset Index / Build Manifest（构建结果级）**
  - `sample_id`, `episode_id`, `t`, `split`, `bucket`, `shard_id`。
- **Step Record（step 级，训练最小字段）**
  - `episode_id`, `t`, `obs_image_refs`, `obs_state`, `action`, `intervention_flag`, `terminal`。
- **APO Labels（step 级）**
  - `sample_type`（`correct|interaction|incorrect`），`label_source`（`expert|human_intervention|pre_intervention_relabel`）。
- **Dataset Meta（数据集级）**
  - `schema_version`, `action_dim`, `obs_spec`, `normalization_stats`, `build_config_hash`。
- 说明：LeRobot 仅参考字段命名与分层思想，不引用其运行时代码或对象。

## D. dataset build 的最小流程

- 步骤 1：`ingest` 扫描原始 episode，生成标准化中间记录（统一时间顺序）。
- 步骤 2：`relabel` 执行 APO 最小逻辑：
  - 标记三类样本：`correct / interaction / incorrect`。
  - 检测 intervention 起点，向前 K-step 回溯重标记，并写 `label_source=pre_intervention_relabel`。
- 步骤 3：生成样本索引并写出 bucket/labels；划分 train/val；不在数据构建阶段固化 batch 配比。
- 步骤 4：计算最小归一化统计（action 与 `obs_state`），写入 `normalization_stats`。
- 步骤 5：导出 OpenPI adapter 可直接读取的数据目录（含 episode manifest + build index + shard + stats）。
- 比例采样职责：由训练阶段 sampler 负责，采样比例配置放在训练配置中。

## E. OpenPI adapter 的最小职责

- 将内部 Step Record 映射为 OpenPI 训练样本字段（不改 OpenPI 训练主循环）。
- 提供最小 Sampler 接口，支持 APO 三类比例采样（优先在 sampler 层做，不侵入模型层）。
- 读取 `normalization_stats` 并在数据侧完成标准化。
- 输出统一 batch dict，保证 OpenPI 训练脚本只需要“读 batch + 训练”。
- 保留后端替换点：`TrainingBackend` 抽象，当前仅实现 `OpenPIBackend`。
- 接收训练配置中的采样比例参数（如 `correct/incorrect/interaction`），实现无需重建数据包的策略切换。

## F. inference/export 模块组织（便于复制进 TeleManipulation）

- 固定导出一个**自包含推理包目录**：[deploy/inference_bundle/](deploy/inference_bundle/)，TeleManipulation 直接整目录拷贝。
- 包内最小内容：
  - `inference_runner.py`（统一推理入口，暴露 `load_policy()` / `predict_action()`）。
  - `model_spec.yaml`（输入输出维度、预处理参数、版本）。
  - `preprocess.py` / `postprocess.py`（与训练一致的最小变换）。
  - `weights_locator.yaml`（只写 TOS 挂载路径约定，不含权重文件本体）。
  - `README_deploy.md`（TeleManipulation 接入步骤，含 3-5 条命令）。
- 交付原则：代码可复制、权重不入库、路径可配置、接口稳定（函数签名固定）。

## G. uv 环境文件与部署交付文件（最小集）

- 环境文件：
  - [pyproject.toml](pyproject.toml)（运行与训练依赖声明）。
  - [uv.lock](uv.lock)（锁版本，确保可复现）。
  - [.python-version](.python-version)（固定 Python 大版本）。
- 配置与约定：
  - [configs/train_openpi.yaml](configs/train_openpi.yaml)
  - [configs/dataset_build.yaml](configs/dataset_build.yaml)
  - [configs/export_inference.yaml](configs/export_inference.yaml)
  - [docs/weight_path_contract.md](docs/weight_path_contract.md)（TOS 挂载路径规范）。
- 交付文件（给 TeleManipulation）：
  - [deploy/inference_bundle/README_deploy.md](deploy/inference_bundle/README_deploy.md)
  - [deploy/inference_bundle/model_spec.yaml](deploy/inference_bundle/model_spec.yaml)
  - [deploy/inference_bundle/weights_locator.yaml](deploy/inference_bundle/weights_locator.yaml)
  - [deploy/inference_bundle/requirements.txt](deploy/inference_bundle/requirements.txt)（仅推理最小依赖，可由 uv 导出）。

## H. 第一阶段明确不做什么

- 不做 online RL、不接 RLinf，不做 actor-learner infra。
- 不做多后端训练抽象的完整实现（只落 OpenPI 一个实现）。
- 不做重型 canonical episode framework（仅保留最小 manifest + step record）。
- 不做 TeleManipulation 主仓结构改造，只交付可复制推理包。
- 不做权重仓库存储与同步服务（只定义共享挂载路径契约）。
- 不做复杂实验平台（调度、可视化平台、权限系统）。

## I. 推荐开发顺序（1/2/3/4）

1. **数据链路先通**：完成 `ingest + relabel + dataset_build`，产出可复现数据包（含 split/index/labels/stats，不含比例采样固化）。
2. **训练先跑通**：实现 `backend_openpi + train`，在一个小数据集上完成首个可收敛训练。
3. **导出与推理对齐**：实现 `export`，产出 `inference_bundle` 并本地跑通离线推理 smoke test。
4. **交付验证**：按 `README_deploy.md` 在“模拟 TeleManipulation 目录”中拷贝运行，验证权重路径约定与接口稳定。

