我会先做一轮并行检索：按 gemma / vlm / inference / prompt / image 关键词定位脚本、配置和调用入口，然后读最相关文件，最后按可复用优先级给你清单。
下面是按“可直接复用价值”排序的快速探索结果（medium thoroughness）：
P0：Gemma 2B / PaliGemma 推理主链路（最优先）
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/pi05/modeling_pi05.py
这里是 PI05Policy 的核心推理实现：select_action -> predict_action_chunk -> model.sample_actions，并且显式构建 PaliGemma + Gemma expert。如果你要做 Gemma 2B VLM 推理改造，这个文件是第一入口。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/pi05/processor_pi05.py
已有完整 prompt 预处理：把 task + 离散化 state 组装成 "Task: ... State: ...; Action: "，再走 paligemma tokenizer。非常适合复用做“文本条件动作推理”前处理。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/pi05/configuration_pi05.py
配置层已支持 paligemma_variant="gemma_2b"、action_expert_variant、推理步数/图像分辨率等。可直接作为 Gemma2B 推理实验的参数总入口。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/pi0/modeling_pi0.py
与 pi05 同族的 Gemma+PaliGemma 推理实现（含 predict_action_chunk/sample_actions），适合对比或迁移改动。若你要做 pi0/pi05 双分支兼容，这里复用价值高。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/pi0/processor_pi0.py
pi0 的 prompt/tokenizer 流程（包含 newline 兼容 PaliGemma）在这里。对“纯 task prompt 驱动推理”很实用，改动成本低。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/factory.py
统一策略装配入口，pi0/pi05/pi0_fast 都从这里注册和实例化。需要新增/替换 Gemma 系策略时，这里是接线位。
P1：OpenPI 原生模型封装与服务入口（高优先）
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi/src/openpi/models_pytorch/gemma_pytorch.py
OpenPI 的 PaliGemmaWithExpertModel PyTorch 封装原型，定义了 VLM+expert 融合 forward。可作为你在 lerobot 侧实现的“上游对照版本”。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi/src/openpi/models_pytorch/pi0_pytorch.py
OpenPI 的动作采样推理主逻辑（sample_actions/denoise_step）。如果你要对齐 OpenPI 推理行为，这是最关键参考实现。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi/src/openpi/models/tokenizer.py
明确了 PaliGemma prompt 模板与 token 化约定（含 Task/State/Action）。复用它可避免 tokenizer 细节偏差导致性能漂移。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi/src/openpi/policies/policy.py
Policy.infer() 封装了输入 transform + sample_actions + 输出 transform，是 OpenPI 推理调用入口。适合快速接入离线推理/服务化推理。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi/src/openpi/policies/policy_config.py
create_trained_policy() 负责 checkpoint、norm stats、prompt 注入和 transform 组装。做“从 checkpoint 到可推理 policy”的一站式复用非常方便。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi/scripts/serve_policy.py
WebSocket 推理服务入口，可直接启动已训练策略并支持 default_prompt。做在线推理 demo/联调时复用价值很高。
P2：调用入口与训练配置桥接（中优先）
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/scripts/lerobot_eval.py
仓库内最标准的推理调用链之一：make_policy -> preprocessor -> policy.select_action。适合快速验证 Gemma 系策略的端到端推理。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/utils/control_utils.py
实机/在线控制里的单步推理封装 predict_action() 在这里。要接机器人实时推理，这个函数可直接复用。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/configs/train_pi05_openpi.yaml
已有 OpenPI-JAX 的 pi05 配置模板（含 config_name 与 extra args）。你可以直接复制为 Gemma2B 实验配置脚手架。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/configs/train_pi0_openpi.yaml
OpenPI-Torch 的 pi0 启动模板，适合作为本地快速实验 baseline。可用于对照 pi05 或迁移到自定义 config。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/training_repo/backends/openpi_backend.py
训练编排层到 third_party/openpi/scripts/train*.py 的桥接入口。若要统一训练/推理 workflow，这里是关键胶水层。
关于你特别关注的“图像+文本问答 / prompt 推理流程”
已有 prompt 推理流程（Gemma 主链）：有。pi0/pi05/pi0_fast 都是“图像 + task 文本（+状态离散 token）-> 动作生成”流程，核心在 processor_* + modeling_*。
已有显式图文问答（VQA/chat-template）流程：有，但主要在非 Gemma 路线：
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/wall_x/utils.py（含“Predict ... in language”，等价 VQA 子任务分支）
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/wall_x/modeling_wall_x.py（把图像与构造文本一起送入模型）
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/groot/processor_groot.py（apply_chat_template 的图文消息格式）

我会先并行检索与你提到的四类关键词（attention/hidden states/t-SNE聚类/成功失败标注）以及数据集读取相关实现，然后快速打开最相关文件做中等深度梳理，最后给你“可复用能力 + 缺口”的清单。
已按 medium 深度快速扫过，结论如下。
1) attention map / hidden states / t-SNE-聚类 / 成功失败标注
关键文件	可复用能力	缺口
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/scripts/analyze_a_dataset_distribution.py	对 A 数据集做特征导出（a_dataset_features.npz）、PCA+KMeans、silhouette 扫描、分布图输出。	只有 PCA+KMeans；没有 t-SNE/UMAP；没有 attention/hidden-state 导出。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/wall_x/modeling_wall_x.py	模型 forward 支持 output_attentions、output_hidden_states，并在输出结构里返回。	没有“落盘导出/可视化”脚本，只有模型级能力。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/policies/xvla/modeling_florence2.py	编码器/解码器支持返回 hidden_states 与 attentions。	同样缺少统一抽取管线（batch dump、按 episode 对齐、可视化）。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py	OpenPI 的 Gemma 支持 output_hidden_states/output_attentions，返回 BaseModelOutputWithPast。	是底层模型能力，不是任务脚本；无现成导出器。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py	视觉编码器可返回 hidden_states/attentions。	无现成 attention map 渲染或对齐到图像像素的工具链。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/utils/recording_annotations.py	定义并校验 episode 成功/失败标签（success/failure）。	只做标签语义与校验，不做分析统计。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/utils/control_utils.py + /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/scripts/lerobot_record.py	录制时支持热键标注 success/failure，并写入 episode 元数据 episode_success。	与 A/B 任务标签链路是两套体系，未统一。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/scripts/review_episode_tasks_gui.py + /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/scripts/split_lerobot_v21_by_labels.py + /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/scripts/apply_labels_and_build.py	完整的 episode 级 A/B 标注、审阅、拆分、重排索引、构建流程。	这是“任务类别标注”，不是成功/失败；也不包含 attention/hidden-state 分析。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/training_repo/relabel/apo.py	step 级 relabel（correct/interaction/incorrect + pre-intervention 回溯）。	属于样本类型重标记，不是 episode success/failure。
补充：仓库里未检索到 t-SNE 实现（tsne/t-SNE/TSNE 全局无命中）。
2) A 数据集与 episode 末帧 / head camera 读取方式
关键文件	可复用能力	缺口
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/configs/train_policy_act.yaml 与 /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/configs/train_policy_diffusion.yaml	明确使用 dataset.repo_id: pipeline_ab/A（A 子集训练入口）。	配置层仅声明数据源，不处理相机/帧级读取细节。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi/src/openpi/training/config.py	pi05_aloha_wbcd_lora 中把 observation.images.head_cam/left_wrist_cam/right_wrist_cam 映射到 cam_high/cam_left_wrist/cam_right_wrist。	映射存在，但没有单独“只读 head camera 末帧”的工具函数。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi/src/openpi/training/data_loader.py	通过 repo_id 创建 LeRobotDataset，可作为 A 数据集读取主入口。	读取是通用全量流，不是针对末帧分析。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/datasets/lerobot_dataset.py	通过 episode 元数据索引视频文件（get_video_file_path），支持按 episode/camera 定位视频。	没有显式 get_last_frame() API。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/scripts/value_infer_viz.py	可选 video_key/video_keys，按 episode timestamp 解码视频帧，支持多相机叠加导出。	偏可视化导出，不是轻量“抽末帧到数组/表”的专用脚本。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/scripts/review_episode_tasks_gui.py	从 videos/**/episode_*.mp4 自动发现相机视频；按 episode 读取多相机和对应 parquet 关节数据。	GUI 读的是完整视频流，不直接做末帧抽取。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/scripts/split_lerobot_v21_by_labels.py	从 meta/info.json 提取 video 特征键，支持 --require-all-videos 过滤掉缺相机 episode（如 head-only 情况可被筛除）。	主要是拆分/重排，不做帧级内容解析。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/src/lerobot/scripts/lerobot_dataset_viz.py + /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/tests/datasets/test_datasets.py	提供“episode 内最后帧”相关语义与索引方法（dataset_to_index 近尾帧访问）。	文档已提醒“最后帧不一定是任务最终状态”；缺少“终态帧判定”统一逻辑。
