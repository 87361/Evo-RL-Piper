Exploring ~/kuavo_data_challenge with read-only tools: locating the repo, mapping its structure, and searching for LeRobot patches and depth-related code.
Here is a consolidated report for /data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge (your workspace resolves ~/kuavo_data_challenge here). All findings are read-only from that tree.
1. Overall directory structure (2–3 levels)
kuavo_data_challenge/
├── README.md, README_ZH.md
├── setup.py
├── look_safetensor.py
├── lerobot_patches/
│   └── custom_patches.py
├── configs/
│   ├── policy/
│   │   ├── act_config.yaml
│   │   └── diffusion_config.yaml
│   ├── data/
│   │   └── KuavoRosbag2Lerobot.yaml
│   ├── accelerate/
│   │   └── accelerate_config.yaml
│   └── deploy/
│       ├── kuavo_env.yaml
│       └── others_env.yaml
├── kuavo_data/
│   ├── CvtRosbag2Lerobot.py
│   ├── common/
│   │   ├── kuavo_dataset.py
│   │   ├── config_dataset.py
│   │   ├── ros_handler.py, utils.py, logger.py, key_listener.py
│   │   └── __init__.py
│   └── task1_trans/lerobot/meta/   # sample converted dataset metadata
│       ├── info.json
│       └── stats.json
├── kuavo_train/
│   ├── train_policy.py
│   ├── train_policy_with_accelerate.py
│   ├── README.md, README_ZH.md
│   ├── utils/ (utils.py, transforms.py, augmenter.py)
│   └── wrapper/
│       ├── dataset/
│       │   └── LeRobotDatasetWrapper.py
│       └── policy/
│           ├── act/ (ACTConfigWrapper, ACTModelWrapper, ACTPolicyWrapper)
│           └── diffusion/ (Diffusion* wrappers, DiT models, …)
├── kuavo_deploy/
│   ├── config.py
│   ├── eval_others.py
│   ├── kuavo_service/ (server.py, test_on_bag_dataset.py, …)
│   ├── kuavo_env/ (KuavoBaseRosEnv.py, …)
│   ├── src/eval/ (sim_auto_test.py, real_single_test.py)
│   └── utils/ (obs_buffer.py, …)
├── outputs/                        # training logs / checkpoints (if present)
└── third_party/
  └── lerobot/                      # vendored LeRobot (src/, tests/, …)
2. lerobot_patches / custom_patches
Only one module: lerobot_patches/custom_patches.py.
Import rule: Several entry points import it first so monkey-patches apply before other LeRobot imports:
kuavo_train/train_policy.py, train_policy_with_accelerate.py
kuavo_data/CvtRosbag2Lerobot.py
kuavo_deploy/eval_others.py, kuavo_service/server.py, src/eval/*.py
What it patches (at runtime):
Target	Change
lerobot.configs.types	Replaces FeatureType with an extended enum (STATE, VISUAL, ENV, ACTION, REWARD, RGB, DEPTH) and replaces PolicyFeature; assigns back via sys.modules.
lerobot.datasets.compute_stats	Replaces compute_episode_stats with a version that handles uint16 depth arrays, custom image sampling, and different post-processing for depth vs RGB video.
lerobot.datasets.utils	Replaces dataset_to_policy_features so keys with "depth"/"DEPTH" and image-like dtypes map to FeatureType.DEPTH, others to VISUAL, etc.
Commented-out (not active): a full replacement of create_stats_buffers in lerobot.policies.normalize that would treat DEPTH like visual modalities for (c,1,1) stat shapes—that block is entirely commented.
Full file is 253 lines; the active logic is quoted below in section 4–6.
3. Depth data processing (pipelines)
Rosbag → numpy (uint16): kuavo_data/common/kuavo_dataset.py — process_depth_image decodes PNG from compressedDepth, optional uint16, resize with INTER_NEAREST.
Topics: depth_h / depth_l / depth_r map to /cam_h/depth/..., /cam_l/..., /cam_r/... compressed depth (KuavoRosbagReader).
LeRobot export (CvtRosbag2Lerobot.py): For cameras whose name contains "depth", each frame:
Clips raw depth to kuavo.DEPTH_RANGE (from config, default mm).
Per-frame min/max normalize to [0,1], scales to 255, casts to uint8, then repeat(3) on the last axis so the stored tensor is pseudo-RGB (3×H×W).
So the on-disk LeRobot feature for depth is still declared like RGB/video (dtype mode is "video" or "image" from create_empty_dataset), not uint16, in the current conversion script.
Dataset wrapper: CustomLeRobotDataset pairs meta.camera_keys with depth keys (keys containing "depth"), applies the same crop as RGB and nearest resize for depth.
Training augmenter: AugmentationProcessorStep in train_policy_with_accelerate.py sets cam_keys to exclude "depth" so color jitter etc. do not run on depth.
Deploy: kuavo_deploy/utils/obs_buffer.py decodes depth, then depth_preprocess clamps by depth_range, per-frame min–max normalize to [0,1] (float tensor), consistent with conversion intent. Similar logic appears in test_on_bag_dataset.py / KuavoBaseRosEnv.py for keys containing depth.
4. How LeRobot format is extended for depth
No fork of the parquet schema: depth streams are stored as video (or image) features with shape [3, H, W] after stacking normalized depth as three identical channels.
Semantic extension: Patched dataset_to_policy_features marks observation keys whose names include depth as FeatureType.DEPTH so policies and normalization can treat them separately from VISUAL RGB streams.
Optional uint16 path: Patched compute_episode_stats can compute stats when features[key]["dtype"] == "uint16" and episode data is a Bx1xHxW array—this supports an alternate dataset definition that is not what CvtRosbag2Lerobot.py currently writes.
5. Feature types and enumerations
Upstream (vendored) LeRobot — third_party/lerobot/src/lerobot/configs/types.py:
FeatureType: STATE, VISUAL, ENV, ACTION, REWARD, LANGUAGE
PipelineFeatureType, NormalizationMode, PolicyFeature
After custom_patches import, FeatureType gains RGB and DEPTH (string-valued enum, JSON-serializable).
Config-driven modality list for cameras is built in config_dataset.py (default_camera_names): with use_depth: true and which_arm: both, cameras include head_cam_h, depth_h, wrist_cam_l, depth_l, wrist_cam_r, depth_r.
6. Normalization buffers / stats and uint16
Policy configs (act_config.yaml, diffusion_config.yaml) set:
RGB → MEAN_STD (overridden/merged with defaults; RGB keys matter only if features use FeatureType.RGB; exported RGB keys are mostly classified as VISUAL in dataset_to_policy_features, so VISUAL default MEAN_STD still applies to RGB streams).
DEPTH → MIN_MAX (maps [min,max] → [-1,1] in NormalizerProcessorStep._apply_transform).
Dataset stats: For the sample task1_trans dataset, info.json lists observation.depth_* as dtype: "video", shape [3,480,640]. stats.json then has per-channel min/mean/max in [0, ~0.996]—i.e. stats computed after the same /255 image path as RGB video, not raw uint16 mm values.
Patch stats behavior for uint16: custom_sample_depth downsamples with auto_downsample_height_width; get_feature_stats uses axis (0,2,3) like images; unlike images, stats are not divided by 255—only squeeze batch dim. So if you stored true uint16 depth in episodes, aggregated min/max/mean/std would reflect raw depth scale (subject to sampling).
Commented create_stats_buffers: Would have used (c,1,1) shapes for visual/depth for MEAN_STD / MIN_MAX buffers; it is not installed, so behavior is whatever LeRobot’s current processor uses with stats dicts from the dataset (tensor conversion in _NormalizationMixin).
7. Training / inference code paths for depth
ACT
ACTConfigWrapper: depth_features = inputs with FeatureType.DEPTH.
ACTPolicyWrapper / ACTModelWrapper: build observation.depth as a list of tensors (one per depth camera), mean(dim=-3) so 3-channel → 1-channel before a ResNet with conv1 adapted to 1 channel (initialized from RGB weights via channel mean).
Cross-modal fusion between RGB tokens and depth tokens, then projected for the transformer.
Diffusion
DiffusionPolicyWrapper / DiffusionModelWrapper: OBS_DEPTH, optional separate DiffusionDepthEncoder, attention / multimodal fusion; same idea—depth as 1×H×W after channel averaging.
Deploy / eval
obs_buffer.py depth pipeline; KuavoBaseRosEnv.py and tests branch on 'depth' in keys; policy entry points import lerobot_patches.custom_patches so FeatureType.DEPTH and dataset utility behavior match training.
8. Key file contents (LeRobot patches & depth)
The patch module is self-contained. Core active sections:
custom_patches.py
Lines 7-153
Depth framing in conversion (note uint8 + 3 channels, not uint16 in parquet):
CvtRosbag2Lerobot.py
Lines 438-451
Config defining depth usage and range:
KuavoRosbag2Lerobot.yaml
Lines 20-28
ACT policy depth + normalization (feature types in YAML):
act_config.yaml
Lines 109-156
9. Takeaways on uint16 depth statistics
Shipped conversion path: Depth is converted to 3×uint8-like video before writing; compute_episode_stats follows the normal image/video branch (sample paths, ÷255), so stats.json for observation.depth_* matches normalized pseudo-RGB in [0,1], as in your sample stats.json.
Patch uint16 branch: Intended for datasets that actually store dtype: "uint16" episode arrays; your CvtRosbag2Lerobot.py does not set that dtype on depth features today, so that branch is infrastructure for an alternate format, not the default competition pipeline.