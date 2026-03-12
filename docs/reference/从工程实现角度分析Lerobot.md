## 从工程实现角度分析 Lerobot

### 1. robot、camera、teleop、dataset writer、policy runner 核心类/模块

**Robot:**
- 抽象基类：[Robot](src/lerobot/robots/robot.py#L30-L211) - 统一机器人抽象
- 双臂示例：[BiSOFollower](src/lerobot/robots/bi_so_follower/bi_so_follower.py#L30-L159)、[BiOpenArmFollower](src/lerobot/robots/bi_openarm_follower/bi_openarm_follower.py#L30-L179)

**Camera:**
- 抽象基类：[Camera](src/lerobot/cameras/camera.py#L26-L184) - 统一摄像头抽象
- 实现：[opencv](src/lerobot/cameras/opencv/camera_opencv.py)、[realsense](src/lerobot/cameras/realsense/camera_realsense.py)、[zmq](src/lerobot/cameras/zmq/camera_zmq.py)

**Teleoperator:**
- 抽象基类：[Teleoperator](src/lerobot/teleoperators/teleoperator.py#L29-L208) - 统一遥操作抽象
- 实现：[bi_so_leader](src/lerobot/teleoperators/bi_so_leader/bi_so_leader.py)、[bi_openarm_leader](src/lerobot/teleoperators/bi_openarm_leader/bi_openarm_leader.py)

**Dataset Writer:**
- 核心类：[LeRobotDataset](src/lerobot/datasets/lerobot_dataset.py#L566-L1720) - 数据集创建和写入
- 工厂方法：[LeRobotDataset.create()](src/lerobot/datasets/lerobot_dataset.py#L1642-L1660)

**Policy Runner:**
- 工厂函数：[make_policy()](src/lerobot/policies/factory.py#L405-L529) - 策略实例化
- 处理流水线：[DataProcessorPipeline](src/lerobot/processor/pipeline.py#L254-L1431)

### 2. 接入自定义机器人的最小接口

**必须实现的 Robot 接口（[Robot](src/lerobot/robots/robot.py#L30-L211)）：**

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `observation_features` | - | `dict[str, type]` | 观测特征描述（proprioception + exteroception） |
| `action_features` | - | `dict[str, type]` | 动作特征描述（关节位置/速度等） |
| `connect()` | `calibrate: bool = True` | `None` | 建立通信连接 |
| `get_observation()` | - | `RobotObservation` (dict) | 读取当前状态（关节位置、图像等） |
| `send_action()` | `RobotAction` (dict) | `RobotAction` (dict) | 发送控制指令并返回实际执行结果 |
| `disconnect()` | - | `None` | 断开连接并清理资源 |

**特征定义格式：**
```python
# 双臂机器人示例（见 BiSOFollower）
observation_features = {
    "left_joint_positions": np.ndarray,  # 形状: (n_joints,)
    "right_joint_positions": np.ndarray,
    "observation.image": (np.ndarray, (H, W, C)),  # 图像: (type, shape)
}

action_features = {
    "left_joint_positions": np.ndarray,
    "right_joint_positions": np.ndarray,
}
```

### 3. 数据采集到训练的数据流

**数据流路径（[record.py](examples/lekiwi/record.py#L38-L140)）：**
```
Robot.get_observation() -> Processor Pipeline -> dataset.add_frame() -> dataset.save_episode() -> Parquet + Video files
```

**Schema 关键落点：**

1. **特征定义入口** ([hw_to_dataset_features](src/lerobot/datasets/utils.py) 转换)：
   - 将硬件 `observation_features`/`action_features` 转换为 dataset schema
   
2. **数据集创建** ([LeRobotDataset.create()](src/lerobot/datasets/lerobot_dataset.py#L1642-L1660))：
   - `features` 参数定义完整 schema（dtype + shape）
   - 生成 `info.json`（元数据）和特征配置

3. **帧缓冲** ([add_frame()](src/lerobot/datasets/lerobot_dataset.py#L1171-L1224))：
   - 按照特征结构收集单帧数据
   - 图像先写入临时目录，最终编码为视频

4. **episode 保存** ([save_episode()](src/lerobot/datasets/lerobot_dataset.py#L1225-L1584))：
   - Parquet 格式存储非图像数据（action, state, episode_index 等）
   - FFmpeg 编码图像为 MP4 视频（可配置 vcodec）

5. **Processor Pipeline 变换特征** ([transform_features()](src/lerobot/processor/pipeline.py#L1317-L1337))：
   - 每个 ProcessorStep 静态声明特征变换
   - 支持重命名、归一化、device 迁移等操作

### 4. 值得继承的抽象 vs 强依赖生态的部分

**强烈建议继承的抽象：**

1. **统一硬件接口** - Robot/Camera/Teleoperator ABC
   - 标准化的连接/断开/观测/动作接口
   - 支持 context manager (`__enter__`/`__exit__`)

2. **处理流水线系统** - [DataProcessorPipeline](src/lerobot/processor/pipeline.py#L254-L1431)
   - 串行数据变换管道，支持状态保存和加载
   - 与 HuggingFace Hub 集成，可版本化 pipeline 配置

3. **特征变换机制** - [ProcessorStep](src/lerobot/processor/pipeline.py#L143-L216)
   - `transform_features()` 静态声明特征变化
   - 支持归一化、重命名、device 转移等标准变换

4. **数据集抽象** - LeRobotDataset 的 Parquet + Video 混合存储
   - 结构化数据用 Parquet（列式存储，压缩好）
   - 图像用视频编码（节省空间，支持流式读取）

**强依赖生态、不建议照搬的部分：**

1. **HuggingFace Hub 深度绑定**
   - 数据集和模型都依赖 HF Hub 的 `push_to_hub()`/`pull_from_repo()`
   - 除非你的 infra 也用 HF，否则需要重写存储层

2. **视频编码栈**
   - 默认使用 `libsvtav1` 编码器，依赖 FFmpeg
   - 布料操作可能需要高帧率视频，编码开销大

3. **特定电机/摄像头驱动**
   - Dynamixel、Feetech、Realsense 等特定硬件驱动
   - 双臂机器人架构虽可借鉴，但底层电机控制需重写

4. **训练脚本** ([train_policy.py](examples/training/train_policy.py))
   - 强依赖其 processor pipeline 和 policy 工厂
   - 训练逻辑可参考，但需适配你的任务需求

### 5. 模块分析表格

| 模块名 | 关键文件 | 职责 | 可借鉴点 | 风险点 |
|--------|----------|------|----------|--------|
| **Robot** | [robot.py](src/lerobot/robots/robot.py#L30-L211) | 统一机器人硬件接口 | - 标准化连接/观测/动作接口<br>- 支持 context manager<br>- 校准机制 | - 特征定义格式固定<br>- 假设位置控制模式（布料可能需力控） |
| **Camera** | [camera.py](src/lerobot/cameras/camera.py#L26-L184) | 统一摄像头接口 | - 同步/异步读取接口<br>- 支持多摄像头<br>- find_cameras() 自动发现 | - 视频编码与存储深度耦合<br>- 默认 MP4 编码，布料操作可能需高帧率 |
| **Teleoperator** | [teleoperator.py](src/lerobot/teleoperators/teleoperator.py#L29-L208) | 统一遥操作接口 | - 标准化 action/feedback 接口<br>- 双臂 leader 示例（bi_so_leader） | - 依赖其特定硬件（SO-ARM100）<br>- 布料操作可能需力反馈设备 |
| **Dataset Writer** | [lerobot_dataset.py](src/lerobot/datasets/lerobot_dataset.py#L566-L1720) | 数据集创建和存储 | - Parquet + Video 混合存储<br>- 静态特征 schema<br>- 支持流式写入 | - 强依赖 HuggingFace Hub<br>- 视频编码栈复杂<br>- 布料接触状态需自定义 schema |
| **Processor Pipeline** | [pipeline.py](src/lerobot/processor/pipeline.py#L254-L1431) | 数据处理流水线 | - 串行变换管道<br>- 支持状态保存/加载<br>- 与 Hub 集成 | - 特征变换逻辑复杂<br>- 需理解 EnvTransition 结构<br>- 布料操作可能需 custom steps |
| **Policy Factory** | [factory.py](src/lerobot/policies/factory.py#L405-L529) | 策略实例化 | - 统一策略接口<br>- pre/post processor 分离<br>- 支持多种 policy（ACT、Diffusion 等） | - 策略架构固定<br>- 强依赖 processor pipeline<br>- 布料操作可能需 custom policy |

---

对于布料操作，建议：
1. 继承 Robot/Teleoperator 接口但重写底层驱动
2. 复用 Processor Pipeline 架构但添加布料相关的特征变换
3. 考虑添加接触状态、张力等布料特定特征到 observation_features


## Lerobot 数据组织方式深度分析

### 1. 原始样本最小字段

从 [create_episode_buffer()](src/lerobot/datasets/lerobot_dataset.py#L1141-L1150) 和 [add_frame()](src/lerobot/datasets/lerobot_dataset.py#L1171-L1224) 可以看到，**每帧必须包含的字段**：

| 字段名 | 类型 | 说明 | 来源 |
|--------|------|------|------|
| `episode_index` | `int` | 所属 episode 的全局索引 | 自动生成 |
| `frame_index` | `int` | episode 内的帧序号 | 自动生成 |
| `timestamp` | `float` | 相对时间戳（秒），默认 `frame_index / fps` | 自动生成或手动提供 |
| `task` | `str` | 任务描述/语言指令 | 每帧必须提供（episode 级也存） |

**加上 features 定义的字段**（来自 `info.json`）：
- `observation.*` - 传感器数据（proprioception + exteroception）
- `action.*` - 执行的动作
- 可选的 `reward`, `done`, `info` 等（RL 场景）

### 2. Episode / Frame / Video / Metadata 组织方式

**目录结构**（从 [__init__ 文档](src/lerobot/datasets/lerobot_dataset.py#L566-L670)）：

```
dataset_root/
├── meta/                          # 元数据层
│   ├── info.json                  # Schema 定义、fps、shapes、tasks 数量等
│   ├── stats.json                 # 每个特征的统计量（mean/std/quantile）
│   ├── tasks.parquet              # task_index -> task_description 映射
│   └── episodes/                  # episode 级元数据
│       ├── chunk-000/file-000.parquet
│       │   # 每行一个 episode，包含：
│       │   - episode_index
│       │   - length (帧数)
│       │   - data/chunk_index
│       │   - data/file_index
│       │   - videos/obs.image1/chunk_index
│       │   - videos/obs.image1/file_index
│       │   - ...
│       └── ...
│
├── data/                          # 帧级数据（结构化）
│   ├── chunk-000/file-000.parquet  # 多个 episode 的帧级数据
│   │   # 每行一帧，包含所有 features 定义的列
│   │   # 图像/视频列存的是文件路径（字符串）
│   │   # 其他列是实际数值
│   │   # 每个文件大小由 data_files_size_in_mb 控制
│   └── ...
│
└── videos/                        # 视频数据（可选）
    ├── observation.images.camera1/
    │   ├── chunk-000/file-000.mp4  # 一个或多个 episode 的视频
    │   │   # 每个 video 文件包含连续帧
    │   │   # 通过 frame_index + video 内部时间戳对齐
    │   └── ...
    └── ...
```

**关键组织规则**：

1. **Chunking 机制**：多个 episode 合并到同一个文件，提高 I/O 效率
   - 每个 chunk 限制：`chunks_size`（最大 episode 数）、`data_files_size_in_mb`、`video_files_size_in_mb`
   - Episode 元数据中记录 `chunk_index` 和 `file_index` 用于定位

2. **帧级数据**（Parquet）：
   - 所有数值型数据（关节位置、状态、动作）直接存储
   - 图像/视频列存储**文件路径**（`"videos/camera1/chunk-000/file-000.mp4"`）
   - 视频列在 Parquet 中是 `None`，通过 episodes 元数据定位实际视频文件

3. **视频级数据**（MP4）：
   - 每个摄像头一个独立目录
   - 视频内部时间戳与 Parquet 的 `timestamp` 对齐
   - 支持**流式编码**（[streaming_encoding](src/lerobot/datasets/lerobot_dataset.py#L646-L647)），先写 MP4 不等 episode 结束

### 3. 时间同步依赖的字段

从 [delta_timestamps](src/lerobot/datasets/lerobot_dataset.py#L615) 和 [_query_videos()](src/lerobot/datasets/lerobot_dataset.py#L1048-L1066) 可以看到：

| 字段 | 作用 | 同步机制 |
|------|------|----------|
| `timestamp` | 每帧的绝对时间戳 | Parquet 帧按时间戳排序，视频通过时间戳随机访问 |
| `fps` | 数据集帧率 | 计算默认时间戳：`frame_index / fps` |
| `tolerance_s` | 时间容差（默认 `1e-4`） | 丢弃与目标时间差超过容差的帧 |
| `delta_timestamps` | 查询历史帧的时间偏移 | 例如 `{"observation.images": [-0.1, -0.2, -0.5]}` 查询 100ms/200ms/500ms 前的帧 |

**同步流程**（[__getitem__](src/lerobot/datasets/lerobot_dataset.py#L1082-L1119)）：
1. 根据 `episode_index` + `frame_index` 找到当前帧在 Parquet 中的位置
2. 如果提供 `delta_timestamps`，计算目标时间戳 `timestamp + delta`
3. 在同一 episode 的 Parquet 中查找时间戳最接近 `±tolerance_s` 的帧
4. 对于视频帧，通过时间戳在 MP4 中随机访问

### 4. 扩展 Intervention、Phase、Success/Failure、Quality Score 的字段设计

**推荐字段表**（基于 Lerobot 的扩展规范）：

| 字段名 | 位置 | 类型 | 说明 | 参考 Lerobot 实现 |
|--------|------|------|------|------------------|
| `language_instruction` | 帧级 | `str` | 当前帧的自然语言指令 | 类似 `task`，但可以是帧级变化 |
| `intervention_type` | 帧级 | `str | int` | 干预类型（teleop/ai/human/safety） | 新增，枚举值定义在 `info.json` |
| `intervention_timestamp` | 帧级 | `float` | 干预发生的精确时间戳 | 与 `timestamp` 对齐 |
| `phase` | 帧级 | `str | int` | 操作阶段（reach/grasp/manipulate/retract） | 新增，枚举值 |
| `success` | Episode 级 | `bool` | 任务是否成功 | 存储在 `meta/episodes/` parquet |
| `failure_reason` | Episode 级 | `str | None` | 失败原因（如果失败） | 存储在 `meta/episodes/` parquet |
| `quality_score` | Episode 级 | `float` | 任务质量评分（0-1） | 存储在 `meta/episodes/` parquet |
| `task_index` | 帧级 | `int` | 关联到 `meta/tasks.parquet` 的索引 | Lerobot 已支持 |

**扩展位置的策略**：

1. **帧级字段**（高频变化）：
   - 直接加入 `features` 定义，存储在 `data/` Parquet
   - 例如：`intervention_type`（每帧可能变化）

2. **Episode 级字段**（整个 episode 不变）：
   - 存储**在 `meta/episodes/` parquet**，不重复存储每帧
   - 例如：`success`, `failure_reason`, `quality_score`
   - 访问时通过 `episode_index` 查找

3. **语言指令**：
   - Lerobot 已经有 `task` 字段和 `meta/tasks.parquet`
   - 可以扩展为**帧级语言**（指令可能在 episode 中变化）
   - 例如：`language_instruction` 字段 + `meta/prompts.parquet`（如果需要复用）

### 5. 最值得保留的 5 条设计原则（不照搬格式，只借鉴思想）

**设计原则表**：

| 原则 | 说明 | 在你的 infra 中的体现 |
|------|------|----------------------|
| **1. Schema-First** | 所有数据结构预先在元数据中定义（`info.json`），特征描述包含 dtype + shape + names | 定义一个全局 schema 文件，描述所有观测、动作、干预、标记的字段结构 |
| **2. 时间轴统一** | 所有模态（关节、图像、视频）统一对齐到同一时间轴，通过 `timestamp` + `fps` + `tolerance` 同步 | 每帧必须有时间戳，所有传感器数据按时间戳对齐，支持 delta_timestamps 查询历史 |
| **3. 分层存储** | 元数据、帧级数据、视频数据三层分离，根据访问频率选择存储格式 | Episode 级标记（success/quality）存元数据，帧级干预存 Parquet，视频存 MP4/HDF5 |
| **4. Chunking + 索引** | 多个 episode 合并到 chunk 文件，通过元数据索引快速定位 | 避免一个 episode 一个文件，按大小/数量 chunk，用索引文件映射 episode → chunk |
| **5. 向后兼容版本化** | `codebase_version` + 数据集版本，支持迁移脚本（如 v2.1 → v3.0） | 设计版本字段，支持 schema 演进，提供转换工具 |

---

## 推荐字段表（布料操作场景）

```json
{
  "info": {
    "fps": 30,
    "total_episodes": 1000,
    "total_frames": 300000,
    "features": {
      // === 观测 ===
      "observation/left_arm/joint_positions": {"dtype": "float32", "shape": [7]},
      "observation/left_arm/joint_velocities": {"dtype": "float32", "shape": [7]},
      "observation/right_arm/joint_positions": {"dtype": "float32", "shape": [7]},
      "observation/right_arm/joint_velocities": {"dtype": "float32", "shape": [7]},
      "observation/gripper/left_state": {"dtype": "float32", "shape": [1]},
      "observation/gripper/right_state": {"dtype": "float32", "shape": [1]},
      
      // === 多相机 ===
      "observation/images/overhead": {"dtype": "video", "shape": [3, 480, 640]},
      "observation/images/wrist_left": {"dtype": "video", "shape": [3, 480, 640]},
      "observation.images/wrist_right": {"dtype": "video", "shape": [3, 480, 640]},
      "observation/images/depth": {"dtype": "video", "shape": [1, 480, 640]},
      
      // === 动作 ===
      "action/left_arm/joint_positions": {"dtype": "float32", "shape": [7]},
      "action/right_arm/joint_positions": {"dtype": "float32", "shape": [7]},
      "action/gripper/left_cmd": {"dtype": "float32", "shape": [1]},
      "action/gripper/right_cmd": {"dtype": "float32", "shape": [1]},
      
      // === 干预标记（新增）===
      "intervention/type": {"dtype": "string", "shape": []},  // "teleop", "ai", "human_corrective", "safety_stop"
      "intervention/timestamp": {"dtype": "float32", "shape": []},  // 干预发生的精确时间
      "intervention/reason": {"dtype": "string", "shape": []},  // "collision", "drift", "manual_override"
      
      // === 阶段标记（新增）===
      "phase": {"dtype": "string", "shape": []},  // "reach", "grasp", "manipulate", "retract"
      
      // === 语言指令 ===
      "language_instruction": {"dtype": "string", "shape": []},  // 当前帧指令（可变）
      "task_index": {"dtype": "int64", "shape": []}  // 关联到 tasks.parquet
    },
    "tasks": ["fold the cloth", "smooth the cloth", "place the cloth"]
  },
  "meta/episodes": {
    "episode_index": "int64",
    "length": "int64",
    "success": "bool",           // 新增
    "failure_reason": "string",  // 新增，可为 null
    "quality_score": "float32",  // 新增
    "task_index": "int64",
    "data/chunk_index": "int64",
    "data/file_index": "int64",
    "videos/observation.images.overhead/chunk_index": "int64",
    "videos/observation.images.overhead/file_index": "int64",
    // ... 其他 cameras
  }
}
```

---

## 设计原则表（从 Lerobot 提炼）

| # | 原则 | Lerobot 体现 | 你的 infra 实现 |
|---|------|--------------|----------------|
| **1** | **Schema-First** | `info.json` 预先定义所有特征的 dtype/shape/names | 创建全局 `schema.yaml`，描述所有字段及其语义 |
| **2** | **统一时间轴** | 所有模态通过 `timestamp` 对齐，`tolerance_s` 容差 | 每帧必须有 `timestamp`，所有传感器按时间戳对齐，支持 delta queries |
| **3** | **分层存储** | meta（元数据）→ data（帧级）→ videos（视频） | Episode 级标记存元数据表，帧级干预存时序数据库，视频存对象存储 |
| **4** | **Chunking 策略** | 按 chunks_size + file_size_in_mb 分片 | 多个 episode 合并到分片文件，避免文件碎片化 |
| **5** | **可扩展版本化** | `codebase_version` + 迁移脚本 | 设计 `schema_version` 字段，提供向后兼容的迁移工具 |

---

- 深入研究 [info.json 的完整 schema](src/lerobot/datasets/utils.py)（查看 `load_info` 函数）
- 看 [episode 元数据 parquet 的结构](src/lerobot/datasets/lerobot_dataset.py#L328-L406)（`_save_episode_metadata`）
- 如果要完全脱离 HF，参考 [OnlineBuffer](src/lerobot/datasets/online_buffer.py#L53-L296) 使用 numpy.memmap 的方案