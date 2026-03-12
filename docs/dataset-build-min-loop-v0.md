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
3. 运行 OpenPI 可读 smoke：
   - `python scripts/smoke_dataset_openpi.py --dataset-root data/processed/openpi_v0`

## Smoke 通过标准

- 构建产物完整：
  - `steps/shard-*.jsonl`
  - `manifests/build_manifest.jsonl`
  - `labels/sample_labels.jsonl`
  - `meta/normalization_stats.json`
  - `meta/dataset_meta.json`
- E2E 测试通过：
  - 覆盖 ingest -> relabel -> split/index -> stats -> adapter read
- smoke 脚本通过：
  - 契约字段完整
  - `sample_id` 跨文件一致
  - `OpenPIDatasetAdapter.load_split("train")` 成功返回
