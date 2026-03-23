# 左臂仿真部署（PI05 + TeleManipulation Dummy）

本流程用于在 headless 服务器上验证：
- 三路图像来自数据集（`head_cam/left_wrist_cam/right_wrist_cam`）
- `state` 来自仿真左臂 `joint_states`
- 仅控制左臂 `joint_cmd`

## 1. 前置路径

- Evo 仓库：`/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper`
- 仿真仓库：`/data/vepfs/users/intern/lingyue.yang/TeleManipulation`
- checkpoint：`third_party/openpi/checkpoints/pi05_aloha_wbcd_lora/evorl_pi05_lora_A5_prompt_260318`

## 2. 启动 TeleManipulation 单臂 dummy

建议用 `tmux` 启动长期进程，避免 SSH 断开中断。

```bash
cd /data/vepfs/users/intern/lingyue.yang/TeleManipulation
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source ros2_ws/install/setup.bash
tmux new -s left_dummy -d \
  "ros2 run arm_interface arm_interface_node --ros-args -p arm_type:=dummy -p single_arm:=true -p state_frequency:=60.0"
```

验证 topic：

```bash
ros2 topic list | rg "/left_arm/joint_states|/left_arm/joint_cmd"
```

## 3. 启动策略桥接节点（Evo）

```bash
cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper
source /opt/ros/humble/setup.bash
tmux new -s left_policy -d \
  "python scripts/run_left_arm_sim_policy.py \
  --checkpoint-dir third_party/openpi/checkpoints/pi05_aloha_wbcd_lora/evorl_pi05_lora_A5_prompt_260318 \
  --config-name pi05_aloha_wbcd_lora \
  --repo-id pipeline_ab/A \
  --episode-index 0 \
  --prompt 'pick and place' \
  --device cuda \
  --loop-hz 10"
```

该脚本会：
- 订阅 `/left_arm/joint_states`
- 发布 `/left_arm/joint_cmd`
- 发布可视化辅助 topic：
  - `/sim_policy/current_state`
  - `/sim_policy/target_action`
  - `/sim_policy/state_action_error`

## 4. Headless 可视化（端口转发）

### 4.1 启动 Web GUI（服务器端）

```bash
cd /data/vepfs/users/intern/lingyue.yang/TeleManipulation
source /opt/ros/humble/setup.bash
source .venv/bin/activate
source ros2_ws/install/setup.bash
tmux new -s arm_tuner -d \
  "ros2 run drdc_gui arm_tuner --ros-args -p arm_side:=left -p port:=5003"
```

### 4.2 本地机器做 SSH 端口转发

```bash
ssh -L 5003:127.0.0.1:5003 <user>@<server_ip>
```

浏览器访问：`http://127.0.0.1:5003`

## 5. 轨迹验收 checklist

- 左臂状态持续更新：GUI 中 Hz > 0，关节值变化正常
- 仅左臂被控制：`/right_arm/joint_cmd` 无发布（单臂模式）
- 动作闭环无异常：桥接脚本日志持续输出 `step ok`
- 误差收敛趋势可观测：
  - `ros2 topic echo /sim_policy/state_action_error`
  - 或在 GUI + 日志联合观察

## 6. 常见问题

- `No left-arm state received yet`
  - `arm_interface_node` 未启动或 topic 名不一致
- `Checkpoint not found`
  - 检查 `--checkpoint-dir` 是否相对 Evo 根目录
- `Unexpected action shape`
  - 配置与 checkpoint 不匹配，确认 `--config-name pi05_aloha_wbcd_lora`
- 无法打开网页
  - 检查 `arm_tuner` 是否在服务器监听 `0.0.0.0:5003`
  - 检查 SSH 端口转发命令和防火墙策略
