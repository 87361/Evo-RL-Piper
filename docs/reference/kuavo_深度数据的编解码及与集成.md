1. 深度图如何编码 / 存储
两层：ROS 里一种，写入 LeRobot 又是另一种。
A. Rosbag / compressedDepth（解码前）
话题如 /cam_l/depth/image_rect_raw/compressedDepth。载荷里内嵌 PNG：在 msg.data 里查找 PNG 魔数，再 cv2.imdecode(..., IMREAD_UNCHANGED)，得到 np.uint16 单通道图（注释里会警告若不是 16 位）。这是典型的「压缩深度 = PNG 包一层」的用法，等价于按 PNG 存的 16 位深度，不是自定义二进制格式。
B. 写入 LeRobot v2 数据集（当前 CvtRosbag2Lerobot.py 实现）
深度没有以 uint16 列存进 parquet。做法是：先按 DEPTH_RANGE clip，再对每一帧做 min–max 归一化到 
[
0
,
1
]
[0,1]，乘 255 变成 uint8，再在最后一维 repeat(3) 伪造成 3 通道，与 RGB 一样走 image/video 的 feature 定义（dtype 为 mode，即 "video" 或 "image"）。
因此：管线里「训练用的 LeRobot 数据」主要是 3×uint8 的伪 RGB，不是 16bit PNG 文件落盘；16 位只出现在从 rosbag 解码后的内存数组阶段。
2. 解码深度图：位置与关键函数
角色	路径	关键逻辑
Rosbag → uint16 数组	kuavo_data/common/kuavo_dataset.py	静态方法 RosbagMsgProcessor.process_depth_image：找 PNG 头 → np.frombuffer → cv2.imdecode(..., cv2.IMREAD_UNCHANGED) → cv2.resize(..., INTER_NEAREST)
kuavo_dataset.py
Lines 133-158
    def process_depth_image(msg):        if not (hasattr(msg, 'format') and hasattr(msg, 'data')):            print(f"Skipping invalid message")        # print(f"message format: {msg.format}")        png_magic = bytes([137, 80, 71, 78, 71, 13, 10, 26, 10])        idx = msg.data.find(png_magic)        if idx == -1:            print("PNG header not found, unable to decode.")            return None        png_data = msg.data[idx:]        np_arr = np.frombuffer(png_data, np.uint8)        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)        ...        depth_image = cv2.resize(image, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_NEAREST)        return {"data": depth_image, "timestamp": msg.header.stamp.to_sec()}
（注意：上面魔数字节列表在源码里是 80, 78, 71 对应 ASCII P,N,G，属于标准 PNG signature。）
写入 LeRobot 时的「再编码」在 CvtRosbag2Lerobot.py 的帧循环里（clip + 每帧 min-max + uint8 + 三通道 repeat），不是 imencode，而是纯 numpy/torch 张量写入数据集 API。
3. 如何集成进 LeRobot 数据格式
特征定义（create_empty_dataset）：深度相机名里含 'depth' 时，增加键 observation.{cam}（与 RGB 的 observation.images.{cam} 区分），shape 仍为 (3, H, W)，dtype 与 RGB 相同为 mode（video/image）。
CvtRosbag2Lerobot.py
Lines 220-240
    for cam in cameras:        if 'depth' in cam:            features[f"observation.{cam}"] = {                "dtype": mode,                 "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),  # Attention: for datasets.features "image" and "video", it must be c,h,w style!                 ...            }        else:            features[f"observation.images.{cam}"] = {                "dtype": mode,                "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),                ...            }
写帧（populate_dataset）：深度分支写入上述伪 RGB uint8 张量。
CvtRosbag2Lerobot.py
Lines 438-451
            for idx, (camera, img_array) in enumerate(imgs_per_cam.items()):                if "depth" in camera:                    min_depth, max_depth = kuavo.DEPTH_RANGE[0], kuavo.DEPTH_RANGE[1]                    depth_uint16 = np.clip(img_array[i], min_depth, max_depth)                    max_depth = depth_uint16.max()                    min_depth = depth_uint16.min()                    depth_normalized = (depth_uint16 - min_depth) / (max_depth - min_depth + 1e-9)                    depth_normalized = (depth_normalized * 255).astype(np.uint8)                    frame[f"observation.{camera}"] = depth_normalized[..., np.newaxis].repeat(3, axis=-1)                else:                    frame[f"observation.images.{camera}"] = img_array[i]
运行时补丁 lerobot_patches/custom_patches.py（须在 import lerobot 相关逻辑之前执行）：
扩展 FeatureType.DEPTH，并替换 lerobot.configs.types 里的枚举。
重写 dataset_to_policy_features：若 feature 的 dtype 为 image/video/uint16 且 key 中含 "depth"，则标为 FeatureType.DEPTH。
重写 compute_episode_stats：对 dtype == "uint16" 的列走 custom_sample_depth（为「若你以后把深度做成 parquet 里的 uint16 数组」预留；与当前转换脚本的三通道 video 路径可并存）。
训练入口：例如 kuavo_train/train_policy_with_accelerate.py 第一行 import lerobot_patches.custom_patches，保证上述 monkey-patch 生效。
4. 配置与 transforms
策略配置（示例 configs/policy/act_config.yaml）：normalization_mapping 里 DEPTH: MIN_MAX、RGB: MEAN_STD；policy.custom 里 use_depth: true、depth_backbone 等。
数据集几何对齐：kuavo_train/wrapper/dataset/LeRobotDatasetWrapper.py 的 __getitem__：若有 depth key，则按 meta.camera_keys 与 depth key zip 对齐，对 RGB 做 image_transforms 后，把 crop_position / resize_shape 同步到深度，深度 resize 使用 InterpolationMode.NEAREST。
LeRobotDatasetWrapper.py
Lines 72-93
        if self.image_transforms is not None:            image_keys = self.meta.camera_keys            depth_keys = [key for key in item.keys() if "depth" in key and "is_pad" not in key]            ...                for rgb_cam, depth_cam in zip(image_keys, depth_keys):                    item[rgb_cam], crop_position, resize_shape = self.image_transforms(item[rgb_cam])                    ...                    if resize_shape is not None:                        item[depth_cam] = torchvision.transforms.functional.resize(item[depth_cam], resize_shape,torchvision.transforms.InterpolationMode.NEAREST)
ACT：kuavo_train/wrapper/policy/act/ACTPolicyWrapper.py 把各 depth feature mean(dim=-3) 压成 1 通道，供 ACTModelWrapper 里改过的 Conv2d(in_channels=1, ...) ResNet depth backbone 使用。
Diffusion：DiffusionPolicyWrapper.py / DiffusionModelWrapper.py 里对 observation.depth 与 depth ResNet / cross-attn 的集成（逻辑类似，堆叠多相机深度等）。
方案一句话总结
环节	做法
Ros 侧	compressedDepth 里找 PNG → OpenCV 解码 → uint16 深度图
LeRobot 落盘（本仓库现状）	每帧 min-max → uint8 → 3 通道重复，键名 observation.*depth*，走标准 video/image feature
补丁	custom_patches 注入 FeatureType.DEPTH、统计与 dataset_to_policy_features；可选 uint16 统计分支供「原生 uint16 列」数据集
训练	先 import lerobot_patches.custom_patches；CustomLeRobotDataset 对 depth 做与 RGB 一致的 crop + 最近邻 resize；策略里 use_depth + normalization_mapping.DEPTH
若你希望 LeRobot 里保留原始毫米级 uint16，需要改 CvtRosbag2Lerobot.py 的 feature dtype/shape 与写帧逻辑，并确保与 compute_episode_stats 的 uint16 分支及下游 policy 输入 dtype 一致；当前官方转换脚本走的是 伪 RGB uint8 路径以降低与 LeRobot 视频管线的摩擦。