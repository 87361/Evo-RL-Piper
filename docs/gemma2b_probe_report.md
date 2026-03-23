# Gemma 2B Probing Report

## 1. Setup

- Config: `pi05_aloha`
- Checkpoint: `/data/vepfs/users/intern/lingyue.yang/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch`
- Dataset (from config): `pipeline_ab/A`
- Output root: `artifacts/gemma2b_probe/`
- Runtime: `CUDA_VISIBLE_DEVICES=0,1`, `--device cuda:0`
- Probe scale: `max-episodes=1` (quick validation pass)

## 2. Run Commands

### A) VLM Prompting

```bash
CUDA_VISIBLE_DEVICES=0,1 "/data/vepfs/users/intern/lingyue.yang/openpi/.venv/bin/python" \
  scripts/probe_gemma2b_vlm_prompting.py \
  --config-name pi05_aloha \
  --checkpoint-dir "/data/vepfs/users/intern/lingyue.yang/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch" \
  --repo-id "pipeline_ab/A" \
  --output-dir "artifacts/gemma2b_probe/A_vlm" \
  --max-episodes 1 \
  --device cuda:0
```

### B) Attention Visualization

```bash
CUDA_VISIBLE_DEVICES=0,1 "/data/vepfs/users/intern/lingyue.yang/openpi/.venv/bin/python" \
  scripts/probe_gemma2b_attention.py \
  --config-name pi05_aloha \
  --checkpoint-dir "/data/vepfs/users/intern/lingyue.yang/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch" \
  --repo-id "pipeline_ab/A" \
  --output-dir "artifacts/gemma2b_probe/B_attention" \
  --max-episodes 1 \
  --device cuda:0
```

### C) Hidden-State Phase Clustering (pre/post contact)

```bash
CUDA_VISIBLE_DEVICES=0,1 "/data/vepfs/users/intern/lingyue.yang/openpi/.venv/bin/python" \
  scripts/probe_gemma2b_hidden_phase.py \
  --config-name pi05_aloha \
  --checkpoint-dir "/data/vepfs/users/intern/lingyue.yang/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch" \
  --repo-id "pipeline_ab/A" \
  --output-dir "artifacts/gemma2b_probe/C_phase" \
  --max-episodes 1 \
  --frames-per-phase 2 \
  --device cuda:0
```

## 3. Outputs

- A: `artifacts/gemma2b_probe/A_vlm/`
  - `vlm_prompt_results.jsonl`
  - `vlm_prompt_results.csv`
  - `summary.json`
- B: `artifacts/gemma2b_probe/B_attention/`
  - `attention_per_sample.jsonl`
  - `attention_metrics.json`
  - `heatmaps/*.png`
- C: `artifacts/gemma2b_probe/C_phase/`
  - `phase_features.npz`
  - `phase_cluster_metrics.json`
  - `phase_sample_index.json`
  - `figures/tsne_phase_scatter.png`
  - `figures/umap_phase_scatter.png` (if `umap-learn` installed, this run is `null`)

## 4. Results (Current Run)

### A) Semantic + Coordinate Probe

- Semantic hit rate: `0.0`
- Coordinate parse rate: `0.0`
- Coordinate in-bounds rate: `0.0`
- Notes on failure patterns:
  - `answer_1` 输出为乱码（`囷`），未命中“褶皱/中间”等关键词。
  - `answer_2` 也为乱码，无法解析 `x=..., y=...` 或任意有效坐标。

### B) Attention Probe

- Mean entropy (lower is more concentrated): `0.8441`
- Mean peak weight: `0.0811`
- Mean top10% mass: `0.5024`
- Heatmap qualitative notes:
  - 注意力分布偏分散（高 entropy），没有非常尖锐的局部聚焦。
  - top10% 质量约 50%，说明存在一定热点，但尚不足以证明稳定“中间拨开路径”聚焦。

### C) Phase Clustering Probe

- Silhouette (joint/text/visual): `-0.1574 / -0.1534 / -0.1455`
- Davies-Bouldin (joint/text/visual): `1.8661 / 1.8669 / 1.8355`
- Shuffled-label baseline (joint): `silhouette=-0.1574, DB=1.8661`
- Conclusion:
  - pre/post 在当前采样规模下不可自然分离（silhouette 为负）。
  - 与 shuffled-label 指标几乎一致，未观察到显著 phase structure 信号。

## 5. Final Verdict

- Backbone prior is likely:
  - `[x] weak`
  - `[ ] moderate`
  - `[ ] strong`
- Reason:
  - A 探针语义与坐标解析均失败（全部 0）。
  - B 探针注意力较分散，未形成明确任务关键区域聚焦。
  - C 探针聚类指标接近随机标签基线，阶段结构不可分。
  - 本结论基于 `max-episodes=1` 的快速测试；建议扩展到 `20+` episodes 做最终判定。
