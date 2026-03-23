# Training Repo V0 Dataset Build 最小闭环契约

## 范围边界

- 仅 offline：只处理离线 episode json，不接 online 流程。
- 仅最小 schema：不新增平台级字段。
- APO 仅保留：
  - 三类样本重标记：`correct | interaction | incorrect`
  - `K-step pre-intervention relabel`
- dataset build 阶段不固化 batch 比例；比例采样由训练侧 sampler 承担。
- 不引入重型抽象层。

## 路径与切分规则

- 配置中的 repo 内路径使用相对路径。
- 相对路径以 **dataset build 配置文件所在目录** 为锚点解析。
- split 默认按 episode 级切分（`split_mode: episode`）：
  - 同一 `episode_id` 的所有 step 必须在同一 split。
- `sample_id` 在整个 build 结果中必须全局唯一：
  - 规则固定为 `sample_id = "{episode_id}:{t}"`。

## OpenPI 可读数据包最小契约

数据包根目录（示例）：`data/processed/openpi_v0`

- `manifests/build_manifest.jsonl` 每行必须包含：
  - `sample_id`, `episode_id`, `t`, `split`, `bucket`, `shard_id`
- `steps/shard-*.jsonl` 每行必须包含：
  - `sample_id`, `episode_id`, `t`, `obs_image_refs`, `obs_state`, `action`, `intervention_flag`, `terminal`
- `labels/sample_labels.jsonl` 每行必须包含：
  - `sample_id`, `sample_type`, `label_source`
- `meta/normalization_stats.json` 必须包含：
  - `obs_state.mean`, `obs_state.std`, `action.mean`, `action.std`
- `meta/dataset_meta.json` 最小字段：
  - `schema_version`, `action_dim`, `obs_spec`, `normalization_stats`, `build_config_hash`

取值约束：

- `split` 仅允许 `train | val`
- `bucket` / `sample_type` 仅允许 `correct | interaction | incorrect`
- `label_source` 仅允许 `expert | human_intervention | pre_intervention_relabel`
- 对任一 `sample_id`：`build_manifest.bucket == sample_labels.sample_type`

## 最小运行步骤

1. 构建数据包：
   - `python scripts/build_dataset.py --config configs/dataset_build.yaml`
2. 运行最小 E2E 测试：
   - `pytest -q tests/training_repo/test_dataset_build_min_loop.py`
3. 运行 PI 训练入口 smoke：
   - `python scripts/train_pi.py --config configs/train_pi0_openpi.yaml --exp-name=dataset_smoke --overwrite`

## WBCD pi05 LoRA 训练

说明：

- 本节仅记录 `norm stats` 与训练配置/启动方式。
- 当前推荐输入是 **LeRobot v2.1** 数据集（保留视频），不走 v3 转换。

0. 先按 CSV 切分 v2.1 的 A/B（保留三路视频）：
   - `PYTHONPATH=src python scripts/split_lerobot_v21_by_labels.py --src-root <v2.1源数据根目录> --label-csv <task_labels.csv> --output-root <tmp/pipeline_ab/lerobot_v21_split> --task-a-name shirt_open_middle --task-b-name shirt_flatten --drop-non-ab --require-all-videos --overwrite`
   - 目标：`<output-root>/A` 和 `<output-root>/B` 都是 LeRobot `v2.1`，且 `video_path` 非空。

1. 计算归一化统计（必做）：
   - `CUDA_VISIBLE_DEVICES=2,3 uv run scripts/compute_norm_stats.py --config-name pi05_aloha_wbcd_lora`
2. 训练配置关键点（`pi05_aloha_wbcd_lora`，与旧名称等价，字段为准）：
   - `model`: `Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")`
   - `data.repo_id`: `pipeline_ab/A`
   - `data.adapt_to_pi`: `false`
   - 本地数据目录映射到：`HF_LEROBOT_HOME/pipeline_ab/A`（建议链接到 `tmp/pipeline_ab/lerobot_v21_split/A`）
   - `data.repack_transforms`: 图像键需与数据集字段一致（WBCD 原始键为 `head_cam/left_wrist_cam/right_wrist_cam`）
   - `base_config.prompt_from_task`: `true`
   - `weight_loader`: `gs://openpi-assets/checkpoints/pi05_base/params`
   - `freeze_filter`: 使用同构 LoRA `Pi0Config(...).get_freeze_filter()`
   - `num_train_steps=20000`, `batch_size=16`, `fsdp_devices=2`, `ema_decay=None`
3. 启动训练（JAX，FSDP 双卡）：
   - `CUDA_VISIBLE_DEVICES=5,7 uv run scripts/train.py --config-name pi05_aloha_wbcd_lora`

## Smoke 通过标准

- 构建产物完整：
  - `steps/shard-*.jsonl`
  - `manifests/build_manifest.jsonl`
  - `labels/sample_labels.jsonl`
  - `meta/normalization_stats.json`
  - `meta/dataset_meta.json`
- E2E 测试通过：
  - 覆盖 ingest -> relabel -> split/index -> stats 一致性
- 训练入口 smoke 可启动：
  - 由 `scripts/train_pi.py` 转发到 `third_party/openpi` 官方训练脚本
- pi05 LoRA 训练可进入循环：
  - 日志出现 `world_size=2`、`step=...`、`loss=...`
