# 机器人型号绑定点分析

## 绑定点清单

### 1. 机器人本体参数定义

**关节命名**
- 位置：[kuavo_dataset.py](kuavo_data/common/kuavo_dataset.py#L14-L33)
```python
DEFAULT_ARM_JOINT_NAMES = [
    "zarm_l1_link", "zarm_l2_link", ..., "zarm_r7_link",  # 14个手臂关节
]
DEFAULT_DEXHAND_JOINT_NAMES = [
    "left_qiangnao_1", ..., "right_qiangnao_6",  # 12个灵巧手关节
]
DEFAULT_LEJUCLAW_JOINT_NAMES = ["left_claw", "right_claw"]  # 2个夹爪关节
```

**动作维度计算**
- 位置：[KuavoBaseRosEnv.__init__](kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py#L60-L130)
- 根据 `which_arm` + `eef_type` 动态计算 action_space 和 observation_space
- 双臂夹爪：14(arm) + 2(gripper) = 16 维
- 单臂：7(arm) + 1(gripper) = 8 维

**关节限制**
- 位置：[kuavo_real_env.yaml](configs/deploy/kuavo_real_env.yaml#L49-L55) 和 [kuavo_sim_env.yaml](configs/deploy/kuavo_sim_env.yaml#L51-L57)
```yaml
arm_min: [-180,-180,...]  # 角度限制
arm_max: [180,180,...]
eef_min: [0]  # 末端执行器限制
eef_max: [1]
```

### 2. 传感器命名与映射

**相机命名**
- 位置：[kuavo_dataset.py](kuavo_data/common/kuavo_dataset.py#L100) 和配置文件
- 默认相机列表：`['head_cam_h', 'depth_h', 'wrist_cam_l', 'depth_l', 'wrist_cam_r', 'depth_r']`
- 根据 `which_arm` 和 `use_depth` 动态选择（见 [config_dataset.py](kuavo_data/common/config_dataset.py#L57-L65)）

**ROS Topic 映射**
- 位置：[kuavo_dataset.py](kuavo_data/common/kuavo_dataset.py#L161-L224)
- 图像：`CompressedImage` → np.array (H,W,3)
- 关节状态：提取 slice `[12:26]` 对应手臂关节（见 [L277-L284](kuavo_data/common/kuavo_dataset.py#L277-L284)）
- 夹爪/灵巧手：独立处理函数

### 3. 末端执行器类型定义

**位置**：配置文件 + dataclass 属性
- [kuavo_real_env.yaml](configs/deploy/kuavo_real_env.yaml#L18)：`eef_type: qiangnao/leju_claw/rq2f85`
- [config_kuavo_env.py](configs/deploy/config_kuavo_env.py#L25-L35)：动态属性 `use_leju_claw`, `use_qiangnao`
- [kuavo_dataset.py](kuavo_data/common/config_dataset.py#L28-L34)：处理逻辑相同

### 4. 切片配置（机器人特有的映射逻辑）

**位置**：[config_dataset.py](kuavo_data/common/config_dataset.py#L66-L105)
```python
@property
def slice_robot(self):  # 从全量关节中提取手臂部分
    # left: [(12, 19), (19, 19)]  # 左臂[12:19]，右臂不用
    # right: [(12, 12), (19, 26)] # 左臂不用，右臂[19:26]
    # both: [(12, 19), (19, 26)]  # 双臂都用

@property
def claw_slice(self):  # 从末端执行器状态中提取
    # right: [[0, 0], [1, 2]]  # 取右手[1:2]
```

**为什么强绑定**：索引 `[12:19]` 和 `[19:26]` 硬编码了 Kuavo 关节顺序（12-26 是手臂关节）。

---

## 迁移到另一套机械臂的最小修改文件

### 必改文件（硬绑定点）

1. **[kuavo_dataset.py](kuavo_data/common/kuavo_dataset.py)**
   - 修改 `DEFAULT_ARM_JOINT_NAMES` 改为新机械臂关节名
   - 修改 `DEFAULT_DEXHAND_JOINT_NAMES` / `DEFAULT_LEJUCLAW_JOINT_NAMES` 改为新末端执行器
   - 修改 `process_sensors_data_raw_extract_arm` 中的 `[12:26]` 改为新机械臂关节索引
   - 修改 `process_claw_state/cmd`、`process_qiangnao_state/cmd` 等适配新末端执行器消息

2. **[config_dataset.py](kuavo_data/common/config_dataset.py)**
   - 修改 `slice_robot` 的索引范围
   - 修改 `dex_slice` / `claw_slice` 的自由度映射

3. **[kuavo_real_env.yaml](configs/deploy/kuavo_real_env.yaml)** / **[kuavo_sim_env.yaml](configs/deploy/kuavo_sim_env.yaml)**
   - 修改 `arm_min/arm_max` 为新机械臂关节限制
   - 修改 `eef_type` 支持新末端执行器类型
   - 修改 `input_images` 相机名称列表
   - 修改 `image_size` 匹配新相机分辨率

4. **[KuavoBaseRosEnv.py](kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py)**
   - 修改 `init_kuavo_sdk` 和 `initial_topics` 适配新 SDK/ROS 接口
   - 可能需要修改关节控制发布逻辑（目前支持 leju_claw/qiangnao/rq2f85）

5. **[script.py](kuavo_deploy/examples/scripts/script.py)**
   - 修改 `_pub_leju_claw` / `_pub_qiangnao` / `_pub_rq2f85` 适配新末端执行器命令格式

---

## Deploy Entry 与 Runtime Config 组织

### Deploy 入口结构

**主入口脚本**：[eval_kuavo.sh](kuavo_deploy/eval_kuavo.sh) → [script.py](kuavo_deploy/examples/scripts/script.py)

**命令调用模式**：
```bash
python script.py --task <go|run|go_run|here_run|back_to_zero> --config <config_path>
```

**配置分层**（三层架构）：
```
YAML 配置文件
    ├─ kuavo_real_env.yaml (环境参数：关节限制、相机配置、eef_type)
    │   └─ load_kuavo_env_config() → Config_Kuavo_Env (dataclass)
    │
    └─ config_inference (推理参数：模型路径、epoch、task、method)
        └─ load_inference_config() → Config_Inference (dataclass)
```

**组织方式特点**：
1. **Hydra 隐式管理**：虽然配置文件用 Hydra 格式（`hydra.run.dir`），但实际加载用 Python dataclass，避免 Hydra 的"隐式魔法"
2. **环境与推理配置分离**：`config_kuavo_env` 关注机器人硬件，`config_inference` 关注模型运行
3. **数据验证**：dataclass 加载时自动验证 `eef_type` 和 `which_arm` 的合法性

### Runtime 配置传递链

```
script.py (CLI args)
    ↓ 解析 argparse
    ↓ 加载 config_path (YAML)
    ↓ load_kuavo_env_config()
    ↓ KuavoBaseRosEnv.__init__(config_path)
    ↓ self.real/only_arm/eef_type/which_arm/...
    ↓ 动态构建 action_space / observation_space
    ↓ initial_topics() 订阅 ROS topics
```

---

## 可迁移配置组织建议

### 1. 硬绑定索引 → 命名映射

**当前问题**：`slice_robot` 用 `[12:19]` 硬编码索引，迁移必须改代码

**建议改进**：
```python
# 在 YAML 中定义命名到索引的映射
joint_mapping:
  arm:
    left: [l_shoulder_pan, l_shoulder_lift, ..., l_wrist_roll]
    right: [r_shoulder_pan, ..., r_wrist_roll]
  hand:
    qiangnao: [q1, q2, q3, q4, q5, q6]
    claw: [gripper]

# 自动计算 slice
def get_slice(joint_names, joint_mapping):
    return [joint_mapping.index(name) for name in joint_names]
```

**优点**：新机器人只需改 YAML，不改 Python 代码

---

### 2. dataclass + 分层验证（值得借鉴）

**当前实践**：[config_kuavo_env.py](configs/deploy/config_kuavo_env.py#L1-L50)
```python
@dataclass
class Config_Kuavo_Env:
    real: bool
    eef_type: str  # 限制为 'qiangnao'/'leju_claw'/'rq2f85'
    
    @property
    def use_leju_claw(self) -> bool:
        return self.eef_type == 'leju_claw'
```

**优点**：
- 类型安全（IDE 自动补全）
- 自动文档（docstring）
- 数据验证（加载时检查 `eef_type` 合法性）
- 动态属性（`use_leju_claw` 避免到处 `if eef_type == 'leju_claw'`）

**建议**：所有配置都用 dataclass，避免 `dict['key']` 的运行时拼写错误

---

### 3. 配置文件按"硬件抽象层"组织

**当前结构**：
```
configs/
├── data/KuavoRosbag2Lerobot.yaml  # 数据转换配置
└── deploy/
    ├── config_kuavo_env.py       # 环境配置 dataclass（硬件）
    ├── config_inference.py       # 推理配置 dataclass（模型）
    ├── kuavo_real_env.yaml       # 真机环境参数
    └── kuavo_sim_env.yaml        # 仿真环境参数
```

**优点**：
- 环境配置（硬件）与推理配置（模型）分离
- 真机/仿真配置共享 dataclass 逻辑
- YAML 仅存储参数，dataclass 提供验证和属性

**建议**：
```
configs/
├── robot/                          # 机器人硬件配置
│   ├── joints.yaml                # 关节命名/限制/映射
│   ├── cameras.yaml               # 相机命名/分辨率
│   └── end_effectors.yaml         # 末端执行器类型
├── env/                           # 环境配置（组合 robot 配置）
│   ├── real_env.yaml              # 真机：选择 robot + 传感器配置
│   └── sim_env.yaml               # 仿真：同上
└── inference/                      # 推理配置
    └── policy.yaml                # 模型路径/epoch/task
```

---

### 4. Gym Action Space 动态计算（值得借鉴）

**当前实践**：[KuavoBaseRosEnv.__init__](kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py#L60-L130)
```python
if self.which_arm == 'both':
    action_low = np.concatenate((self.arm_min[:7], self.eef_min, ...))
    action_high = np.concatenate((self.arm_max[:7], self.eef_max, ...))
```

**优点**：
- 配置驱动：改 YAML 即可改 action space
- 支持单臂/双臂/末端执行器组合

**建议**：显式暴露配置校验日志
```python
log.info(f"Action space: {action_space.shape}")
log.info(f"  - Arm joints: {self.arm_joint_dim}")
log.info(f"  - EEF DOF: {self.eef_dof}")
```

---

### 5. ROS Topic 解耦（硬绑定点）

**当前问题**：ROS topic 名称散布在代码中（如 `/kuavo/pause_state`, `/cam_l/color/image_raw/compressed`）

**建议**：
```python
# robot_config.yaml
ros_topics:
  pause: "/kuavo/pause_state"
  stop: "/kuavo/stop_state"
  cameras:
    head_cam_h: "/zedm/zed_node/left/image_rect_color/compressed"
    wrist_cam_l: "/cam_l/color/image_raw/compressed"
```

```python
# KuavoBaseRosEnv 初始化时读取
self.pause_topic = config.ros_topics.pause
self.pause_sub = rospy.Subscriber(self.pause_topic, Bool, ...)
```

---

## 总结

| 绑定点类型 | 位置 | 迁移成本 |
|-----------|------|---------|
| 关节命名 | [kuavo_dataset.py](kuavo_data/common/kuavo_dataset.py#L14-L33) | 低（改列表） |
| 关节索引切片 | [config_dataset.py](kuavo_data/common/config_dataset.py#L66-L105) + [kuavo_dataset.py](kuavo_data/common/kuavo_dataset.py#L277-L284) | 中（硬编码索引） |
| 末端执行器类型 | 配置 + dataclass | 低（改 YAML） |
| ROS Topic | 散布代码 | 高（多处改） |
| SDK 接口 | [KuavoBaseRosEnv.py](kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py) + [script.py](kuavo_deploy/examples/scripts/script.py) | 高（重构） |

**最小迁移文件清单**（按优先级）：
1. [kuavo_dataset.py](kuavo_data/common/kuavo_dataset.py) - 关节命名 + 切片逻辑
2. [config_dataset.py](kuavo_data/common/config_dataset.py) - slice 属性
3. [kuavo_real_env.yaml](configs/deploy/kuavo_real_env.yaml) - 关节限制 + 相机配置
4. [KuavoBaseRosEnv.py](kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py) - SDK 初始化 + topic 订阅
5. [script.py](kuavo_deploy/examples/scripts/script.py) - 末端执行器命令发布

---

不要学习这个仓库的索引切片写法，要学习它的配置分层和 dataclass 校验；不要继承它的 ROS topic 散布方式，要反过来把 topic 全部收口进 robot config

想深入了解数据流水线的具体实现，看 [ROS 数据处理流水线](6-ros-data-processing-pipeline)。想了解环境封装设计，看 [Gym 环境封装器设计](15-gym-environment-wrapper-design)。