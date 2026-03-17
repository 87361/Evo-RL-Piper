我会按你指定的三个文件做只读梳理，先快速读取原文，再补充检索 train_policy.py 里与 policy type 分发和 OmegaConf/Hydra/argparse 相关调用，最后按 A-E 五部分给你结构化总结。
下面是对你指定仓库的只读分析结果（未做任何修改）。
关键证据片段
train_policy.py
Lines 209-237
@hydra.main(config_path="../configs/policy/", config_name="diffusion_config", version_base=None)def main(cfg: DictConfig):    ...    policy_cfg = build_policy_config(cfg, input_features, output_features)    ...    policy = build_policy(cfg.policy_name, policy_cfg)
train_policy.py
Lines 106-111
def build_policy(name, policy_cfg):    policy = {        "diffusion": CustomDiffusionPolicyWrapper,        "act": CustomACTPolicyWrapper,    }[name](policy_cfg)
custom_patches.py
Lines 117-149
def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:    ...    if ft["dtype"] in ["image", "video", "uint16"]:        if "depth" in key or "DEPTH" in key:            type = FeatureType.DEPTH        else:            type = FeatureType.VISUAL    elif key == "observation.environment_state":        type = FeatureType.ENV
A. YAML 配置结构（顶层字段与 policy 相关字段）
文件：/data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge/configs/policy/act_config.yaml
顶层字段
hydra（run/sweep 输出目录策略）
task、method、timestamp
repoid、root
training（训练超参与数据增强）
policy_name（act 或 diffusion）
policy（策略配置主体，含 _target_）
training 关键子字段
输出与复现：output_directory、seed
训练控制：max_epoch、max_training_step、accumulation_steps
资源与数据：device、batch_size、num_workers、drop_last
调度：scheduler_name、scheduler_warmup_steps
恢复：resume、resume_timestamp
增强：RGB_Augmenter（enable/max_num_transforms/random_order/tfs）
policy 关键子字段（ACT 配置）
实例化入口：_target_: ...CustomACTConfigWrapper
时序：n_obs_steps、chunk_size、n_action_steps
归一化：normalization_mapping（示例里设置了 RGB: MEAN_STD、DEPTH: MIN_MAX）
精度：use_amp
网络超参：vision_backbone、dim_model、n_heads 等
VAE：use_vae、latent_dim、kl_weight
优化器：optimizer_lr、optimizer_weight_decay、optimizer_lr_backbone
自定义扩展：custom（如 use_depth、depth_backbone）
B. 训练入口如何加载 YAML（Hydra/argparse/OmegaConf）
文件：/data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge/kuavo_train/train_policy.py
Hydra：是主要入口
@hydra.main(config_path="../configs/policy/", config_name="diffusion_config", ...)
运行时把配置注入为 cfg: DictConfig。
argparse：没有使用
此训练入口没有 argparse 参数解析流程。
OmegaConf：有使用
在 build_policy_config 里把 DictConfig 转普通 dict：
OmegaConf.to_container(..., resolve=True)
并将字典项转成 PolicyFeature 对象（_normalize_feature_dict）。
Hydra instantiate：有使用
instantiate(cfg.policy, input_features=..., output_features=..., device=...)
即通过 YAML 中 policy._target_ 动态构造策略配置对象（不是直接构造 policy 模型本体）。
C. 策略工厂如何按 policy type 实例化（ACT / diffusion）
核心在 build_policy(name, policy_cfg)：
name == "diffusion" → CustomDiffusionPolicyWrapper(policy_cfg)
name == "act" → CustomACTPolicyWrapper(policy_cfg)
用字典索引直接分发，未知类型会直接抛 KeyError（无兜底分支）。
完整路径是两段式：
先实例化 config 对象
由 instantiate(cfg.policy, ...) + YAML _target_ 完成（例如 ACT 走 CustomACTConfigWrapper）。
再按 policy_name 选择 wrapper 类
build_policy(cfg.policy_name, policy_cfg) 完成模型包装器实例化。
补充（实现差异）：
CustomACTPolicyWrapper：forward 计算 L1 +（可选）KL；支持把深度特征聚合为 observation.depth。
CustomDiffusionPolicyWrapper：forward 调 self.diffusion.compute_loss(batch)；在训练/推理里对 RGB/Depth 做裁剪与 resize 后再堆叠。
D. 自定义 feature type + 统计/归一化补丁（DEPTH/ENV/VISUAL）
文件：/data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge/lerobot_patches/custom_patches.py
FeatureType 扩展与猴子补丁
自定义 FeatureType 包含 VISUAL/ENV/DEPTH/RGB/...
通过 importlib + sys.modules 覆盖 lerobot.configs.types 中的 FeatureType 和 PolicyFeature。
dataset_to_policy_features 改写要点
dtype in [image, video, uint16] 先归到视觉分支：
key 含 depth/DEPTH ⇒ FeatureType.DEPTH
否则 ⇒ FeatureType.VISUAL
key == observation.environment_state ⇒ FeatureType.ENV
一般 observation* ⇒ FeatureType.STATE，action ⇒ ACTION
结果：把 Kuavo 的 RGB/Depth/Env 显式映射到可被 policy config 识别的 feature type。
compute_episode_stats 改写要点
RGB/视频：采样后按 (0,2,3) 统计并除以 255.0
Depth(uint16)：单独采样函数 custom_sample_depth，同样按 (0,2,3) 统计，但不做 /255
ENV/STATE/ACTION 等数值特征：沿 batch 维统计
归一化策略实际落点
在 YAML 里通过 policy.normalization_mapping 指定（示例里有 DEPTH: MIN_MAX）。
在 CustomACTConfigWrapper / CustomDiffusionConfigWrapper 中会把默认映射与用户映射 merge。
custom_patches.py 里曾尝试补丁 create_stats_buffers，但该段目前是注释状态，不生效。
E. 对你们仓库可直接复用的 5 条设计建议
保留“两段式工厂”：config instantiate 与 policy wrapper instantiate 解耦，便于同一 policy_name 切不同 config wrapper。
复用 feature 映射层：单独维护 dataset_to_policy_features 适配器，把数据 schema（尤其 depth/env）差异隔离到一个入口。
显式 normalization contract：像这里一样在 YAML 定义 normalization_mapping，并在 config __post_init__ 做 default+override 合并，减少隐式行为。
把增强插到 preprocessor pipeline：insert_before_normalizer(...) 这种位置控制很实用，能确保增强与归一化顺序稳定。
统一恢复训练状态打包：除模型权重外同时保存 optimizer/lr_scheduler/scaler/RNG/epoch/step，可做到更接近“无缝续训”。