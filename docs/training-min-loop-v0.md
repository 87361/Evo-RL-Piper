---
name: training-repo-v0-min-loop
overview: 定义与 TeleManipulation 解耦的训练仓库 v0 最小闭环：从原始数据整理到训练，再到可复制交付的推理包。当前 PI 训练入口统一为 `scripts/train_pi.py` + `third_party/openpi`。
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
- `third_party/openpi`：真实 PI 模型训练后端（官方训练入口）。
- `train`：训练入口与实验产物落盘（ckpt、metrics、config 快照）。
- `export`：导出推理代码包 + 配置 + 权重路径约定（给 TeleManipulation 直接拷贝）。

## B. repo 最小目录结构

- [docs/](docs/)：方案、约定、交付说明（当前结论文档落这里）。
- [configs/](configs/)：数据构建配置、训练配置、导出配置。
- [src/training_repo/ingest/](src/training_repo/ingest/)：原始数据扫描与读取。
- [src/training_repo/relabel/](src/training_repo/relabel/)：APO 最小重标记逻辑。
- [src/training_repo/dataset_build/](src/training_repo/dataset_build/)：样本索引生成、bucket/label 写出、train/val 划分、归一化统计、产物导出。
- [third_party/openpi/](third_party/openpi/)：OpenPI 官方训练仓（子模块）。
- [src/training_repo/train/](src/training_repo/train/)：训练 orchestrator。
- [src/training_repo/export/](src/training_repo/export/)：推理包生成器。
- [deploy/inference_bundle/](deploy/inference_bundle/)：最终给 TeleManipulation 拷贝的目录模板。
- [scripts/](scripts/)：`build_dataset.py` / `train_pi.py` / `export_inference_bundle.py`。
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
  - [configs/train_pi0_openpi.yaml](configs/train_pi0_openpi.yaml)
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
2. **训练先跑通**：通过 `scripts/train_pi.py` 调用 `third_party/openpi`，在小数据集完成首个可收敛训练。
3. **导出与推理对齐**：实现 `export`，产出 `inference_bundle` 并本地跑通离线推理 smoke test。
4. **交付验证**：按 `README_deploy.md` 在“模拟 TeleManipulation 目录”中拷贝运行，验证权重路径约定与接口稳定。

## J. 当前进展同步（260313）

- **dataset build 最小闭环**已通过：
  - 相对路径解析
  - episode 级 split
  - 全局唯一 `sample_id = <episode_id>:<t>`
  - OpenPI 最小可读契约字段（`sample_id/obs_state/action/sample_type/label_source`）
- **OpenPI 训练入口已统一**：
  - 已移除 `training_repo` 线性 OpenPI 后端
  - 使用 `scripts/train_pi.py` 转发至 `third_party/openpi/scripts/train.py|train_pytorch.py`
  - 训练产物以 `third_party/openpi/checkpoints/...` 为准
- **采样职责边界保持不变**：
  - APO 比例采样仅在训练侧 sampler 生效
  - 不回写到 dataset build
- **本轮未扩展范围**：
  - 未扩展 resume、多卡、RLinf、真机、完整 DataLoader 分层

复验命令（repo-relative）：

- `PYTHONPATH=src python -m pytest -q --confcutdir=tests/training_repo tests/training_repo/test_dataset_build_min_loop.py`
- `python scripts/train_pi.py --config configs/train_pi0_openpi.yaml --exp-name=smoke --overwrite`

## K. 260316 数据标注与任务拆分补充

### 1) 标注入口（Headless Web GUI）

- 脚本：`scripts/review_episode_tasks_gui.py`
- 最小目标：
  - 批量审阅 episode 的多相机视频
  - 多类别标注（不再限制 `A/B/uncertain`）
  - 立即写入 CSV，避免断连丢标注
- 当前能力补充（260316）：
  - 支持新增/删除类别
  - 支持删除当前 episode 标注记录（从 CSV 删除该行）
  - 支持视频倍速 `1.0x / 1.5x / 2.0x`（默认 `2.0x`，快捷键 `X` 可在 `1.0x/2.0x` 间切换）
- 运行示例：

```bash
PYTHONPATH=src python scripts/review_episode_tasks_gui.py \
  --video-root "<dataset>/videos" \
  --label-csv "<dataset>/task_labels.csv" \
  --host 127.0.0.1 \
  --port 18080
```

### 2) 标注结果回写并构建 A/B 数据集

- 脚本：`scripts/apply_labels_and_build.py`
- 作用：
  - 从 CSV 读取 `episode_id -> label`
  - 批量回写 raw episode 的 `task_id`
  - 分别构建 `A` 与 `B` 两个独立数据集目录
- 运行示例：

```bash
PYTHONPATH=src python scripts/apply_labels_and_build.py \
  --raw-data-root "<raw_episode_json_root>" \
  --label-csv "<dataset>/task_labels.csv" \
  --rewritten-raw-root "<rewritten_root>" \
  --output-root "<ab_dataset_root>" \
  --task-a-name shirt_open_middle \
  --task-b-name shirt_flatten \
  --drop-non-ab
```

### 3) 训练侧语义说明

- 当前仓库已移除 `OpenPIBackend` 线性后端。
- 因此 A/B 最稳妥方式是分目录分别训练，而非单目录混合后仅在 JSON 打标签。
- 结论：标注 GUI 已升级为多类别，但当前构建脚本 `scripts/apply_labels_and_build.py` 仍是 A/B 定向构建逻辑。

## L. 260316 训练落地 SOP 与实现更新

### 1) SOP 关键约束

- 训练样本必须三相机齐全（`head/left_wrist/right_wrist`）。
- 仅使用 CSV 标注 `A/B`；`uncertain` 全部丢弃。
- 已确认 11 条头相机-only 样本（`episode_000153`~`episode_000163`）直接不纳入训练集。

### 2) A 批闭环（实测）

- 数据准备：
  - 对齐三相机后导出 raw json
  - 执行 `apply_labels_and_build.py --drop-non-ab`
  - 得到 A/B：A=76 episodes，B=75 episodes，uncertain=2 dropped
- OpenPI smoke（A）：
  - 配置：`configs/train_pi0_openpi.yaml`
  - 指标：以 `third_party/openpi` 训练日志为准
- ACT smoke（A）：
  - 先转 LeRobot v3：`scripts/convert_training_repo_to_lerobot.py`
  - 训练 loss：`55.847 -> 21.768 -> 14.993 -> 10.462 -> 8.636 -> 7.297`

### 3) 本轮代码改动

- 新增：`scripts/convert_training_repo_to_lerobot.py`
- 新增：`configs/train_pi0_openpi.yaml`
- 修改：`src/lerobot/datasets/utils.py`
  - `load_nested_dataset` 在 cast 前先按 schema 重排列顺序，提高数据加载鲁棒性。

## M. 远端数据集增量同步 SOP（TOS）

### 1) 远端目录确认

```bash
tosutil ls "tos://drobotics-ailab/users/lingyue.yang/dataset/WBCD/" | rg '/$'
```

- 建议按“批次子目录”同步（例如 `alicia_dual_piper_0316_batch1/`），避免从最外层 `WBCD/` 直接拷贝导致本地目录多嵌套一层。

### 2) 本地落盘根目录

```bash
ls -la "/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD"
```

### 3) 增量同步命令（推荐）

```bash
tosutil cp -r -j=50 -u \
  "tos://drobotics-ailab/users/lingyue.yang/dataset/WBCD/alicia_dual_piper_0316_batch1/" \
  "/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD/"
```

- 参数：`-r` 递归，`-j=50` 并发，`-u` 增量更新（仅新增/变化对象）。

### 4) 同步后核对

```bash
tosutil ls "tos://drobotics-ailab/users/lingyue.yang/dataset/WBCD/alicia_dual_piper_0316_batch1/"
```

- 关注：
  - `Failed count is: 0`
  - 重跑增量时出现 `Skip count`（表示一致性已达成）
