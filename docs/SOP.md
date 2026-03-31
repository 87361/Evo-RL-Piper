# SOP: 从数据后处理、数据集分割、启动不同策略的训练到开环测试、真机部署打包

## 数据集处理图形化页面命令（电脑端）

PYTHONPATH=src python scripts/gui/episode_review/main.py --host 127.0.0.1 --port 18080

需要本地输入ssh隧道转发命令： ssh -N -L 18080:127.0.0.1:18080 -p 1022 -i ~/.ssh/id_ed25519_pi05 lingyue.yang@115.190.168.234

## 数据集处理图形化页面命令（手机端）

### 首次启动

使用 tmux 后台启动，防止终端断开导致服务中断：

```bash
tmux new-session -d -s gui_phone \
  "cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/scripts/gui_phone \
   && source .venv/bin/activate \
   && WEB_PORT=3389 python server.py 2>&1 | tee server.log"
```

然后手机或其他设备打开网址：`115.190.168.234:49200`，输入微信群聊内写的密码。

### 重启服务

当代码更新或服务异常时，按以下步骤重启：

```bash
# 1. 查找并杀掉旧进程
ps aux | grep '[s]erver\.py' | grep lingyue          # 确认 PID
kill <PID>                                            # 替换为实际 PID

# 如果有旧的 gui_phone tmux 会话，先关掉
tmux kill-session -t gui_phone 2>/dev/null

# 2. 启动新会话
tmux new-session -d -s gui_phone \
  "cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/scripts/gui_phone \
   && source .venv/bin/activate \
   && WEB_PORT=3389 python server.py 2>&1 | tee server.log"

# 3. 验证服务已启动
sleep 3 && tmux capture-pane -t gui_phone -p | tail -5
# 应看到 "Uvicorn running on http://0.0.0.0:3389" 字样
```

> **Tips**:
> - 查看实时日志：`tmux attach -t gui_phone`（按 `Ctrl-B D` 脱离）
> - 日志文件同时写入 `scripts/gui_phone/server.log`

## 数据集管理与拉取（TOS）

从 TOS 拉取最新采集的数据集（以 `alicia_dual_piper_0330_batch1` 为例），请使用 `tosutil`，以增量下载模式拉取。
**注意目录的嵌套结构**：GUI 要求数据集存放在类似于 `.../WBCD/WBCD/<数据集目录>/<内部同名目录>/` 的双层嵌套结构下（即 `alicia_dual_piper_0330_batch1/alicia_dual_piper_0330_batch1/videos/`），因此拉取时的目标路径直接指定为第一层即可：

```bash
# 鉴于数据集较大，建议使用 tmux 后台防止断开
tmux new-session -s download_wbcd -d 'for i in {1..4}; do tosutil cp -r tos://drobotics-ailab/users/lingyue.yang/dataset/WBCD/alicia_dual_piper_0330_batch$i/ /data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD/alicia_dual_piper_0330_batch$i/ -u; done'

# 附加: 如果想查看下载进度，可以执行 `tmux attach -t download_wbcd`
```

## 启动 LeRobot 系列策略 (如 ACT, Diffusion, VLA 等)

> ⚠️ **命名约定**：v3.0 转换后的数据集 **必须** 使用 `<原名>_lerobot` 后缀命名，**严禁覆盖原始 v2.1 数据集**。
> 原始 v2.1 数据仍被 OpenPI (pi 系列) 管线引用，如果被原地替换会导致正在运行的进程崩溃。
>
> | 目录 | 格式 | 用途 |
> |---|---|---|
> | `alicia_dual_piper_0330_merged` | v2.1 (原始) | OpenPI / pi 系列训练、HF cache |
> | `alicia_dual_piper_0330_merged_lerobot` | v3.0 (转换后) | LeRobot ACT / Diffusion / VLA 训练 |

### Step 1: 转换 v2.1 数据集到 v3.0 格式

转换脚本会 **原地** 操作：将旧数据移到 `<repo-id>_old`，新 v3.0 数据输出到 `<repo-id>`。
**因此我们必须先复制一份再转换**，保护原始数据不受影响。

```bash
# ── 以 alicia_dual_piper_0330_merged 为例 ──
DATASET_DIR=/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD
ORIG_NAME=alicia_dual_piper_0330_merged
LEROBOT_NAME=${ORIG_NAME}_lerobot          # 转换后目标名

# 1. 复制一份作为转换工作区（保护原始数据）
cp -r "${DATASET_DIR}/${ORIG_NAME}" "${DATASET_DIR}/${LEROBOT_NAME}"

# 2. 执行 v2.1 → v3.0 转换（在副本上操作）
cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper
PYTHONPATH=src python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id "${LEROBOT_NAME}" \
    --root "${DATASET_DIR}" \
    --push-to-hub false

# 转换完成后的目录结构：
#   ${DATASET_DIR}/${ORIG_NAME}          ← 原始 v2.1，未被改动 ✅
#   ${DATASET_DIR}/${LEROBOT_NAME}       ← 新的 v3.0 数据 ✅
#   ${DATASET_DIR}/${LEROBOT_NAME}_old   ← 转换脚本自动备份的副本（可安全删除）

# 3. (可选) 清理转换产生的 _old 备份副本
rm -rf "${DATASET_DIR}/${LEROBOT_NAME}_old"
```

### Step 2: 规范底层特征命名（`agent_pos` → `observation.state`）

LeRobot 框架只识别 `observation.state` 键名，而我们的数据使用 `agent_pos`。
需一次性清洗 v3.0 数据集中 parquet/json 的列名。

```bash
cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper

# 直接运行以下 Python 脚本（修改 dataset_root 为你的 v3.0 数据集路径）
PYTHONPATH=src python -c "
import pandas as pd, json
from pathlib import Path

dataset_root = Path('${DATASET_DIR}/${LEROBOT_NAME}')   # ← 指向 v3.0 副本

# 1. 修改 info.json
info_path = dataset_root / 'meta' / 'info.json'
with open(info_path) as f:
    info = json.load(f)
if 'agent_pos' in info['features']:
    info['features']['observation.state'] = info['features'].pop('agent_pos')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    print('info.json modified.')

# 2. 修改 stats.json
stats_path = dataset_root / 'meta' / 'stats.json'
if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)
    if 'agent_pos' in stats:
        stats['observation.state'] = stats.pop('agent_pos')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print('stats.json modified.')

# 3. 修改所有 parquet 列名
for pq_file in dataset_root.rglob('*.parquet'):
    df = pd.read_parquet(pq_file)
    renames = {c: c.replace('agent_pos', 'observation.state') for c in df.columns if 'agent_pos' in c}
    if renames:
        df.rename(columns=renames, inplace=True)
        df.to_parquet(pq_file)
        print(f'{pq_file.name} modified: {renames}')

print('=== Key rename complete! ===')
"
```

*(详细原理参考 `docs/260328.md`)*

### Step 3: 一键启动对应策略

> **`--dataset.root` 必须指向数据集的完整绝对路径**（即含 `meta/info.json` 的那一层），而不是它的父目录。

#### A: 训练传统免依赖模型 (ACT / Diffusion)

```bash
source .venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=alicia_dual_piper_0330_merged_lerobot \
  --dataset.root=/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD/alicia_dual_piper_0330_merged_lerobot \
  --dataset.revision=v3.0 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=100000 \
  --batch_size=8 \
  --num_workers=2
```

#### B: 训练大型依赖模型 (SmolVLA / XVLA)

```bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

# --extra 后面写对应的策略名字如 smolvla 或 xvla
uv run --extra smolvla python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=alicia_dual_piper_0330_merged_lerobot \
  --dataset.root=/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD/alicia_dual_piper_0330_merged_lerobot \
  --dataset.revision=v3.0 \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=100000 \
  --batch_size=2 \
  --num_workers=2
```

### 附录: 恢复被意外覆盖的原始 v2.1 数据集

如果转换脚本不小心覆盖了原始数据集（原始被移到 `_old`），可以这样恢复：

```bash
DATASET_DIR=/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD
ORIG_NAME=alicia_dual_piper_0330_merged

# 1. 将已转换的 v3.0 数据重命名为 _lerobot
mv "${DATASET_DIR}/${ORIG_NAME}" "${DATASET_DIR}/${ORIG_NAME}_lerobot"

# 2. 将 _old 备份恢复为原名
mv "${DATASET_DIR}/${ORIG_NAME}_old" "${DATASET_DIR}/${ORIG_NAME}"

# 验证
cat "${DATASET_DIR}/${ORIG_NAME}/meta/info.json" | python3 -c "import sys,json; print('Restored:', json.load(sys.stdin)['codebase_version'])"
# 应输出: Restored: v2.1
```

## 启动 pi 系列策略
需要准备 lerobotv2.1 格式的数据，然后计算 norm_stats。

### Step 1: 计算 norm_stats

```bash
cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi
source ../../../openpi/.venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 uv run scripts/compute_norm_stats.py --config-name pi05_aloha_wbcd_lora \
  > "/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/tmp/compute_norm_stats_A.log" 2>&1
```

### Step 2: 后处理 norm_stats（clamp 近常量维度）

> 对于单臂任务或部分关节不运动的数据集，q99-q01 会极窄（接近 0），导致归一化值爆炸。
> 需要用 postprocess_norm_stats.py 将窄 range 维度 clamp 到最小值。

```bash
cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper

# 先 dry-run 预览
python scripts/postprocess_norm_stats.py \
    --input  third_party/openpi/assets/pi05_aloha_wbcd_lora/pipeline_ab/A/norm_stats.json \
    --output third_party/openpi/assets/pi05_aloha_wbcd_lora/pipeline_ab/A/norm_stats.json \
    --min-range 0.1 \
    --dry-run

# 确认无误后正式执行（去掉 --dry-run）
python scripts/postprocess_norm_stats.py \
    --input  third_party/openpi/assets/pi05_aloha_wbcd_lora/pipeline_ab/A/norm_stats.json \
    --output third_party/openpi/assets/pi05_aloha_wbcd_lora/pipeline_ab/A/norm_stats.json \
    --min-range 0.1
```

### Step 3: 启动训练

```bash
cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi
source ../../../openpi/.venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export XLA_FLAGS='--xla_gpu_enable_command_buffer='

CUDA_VISIBLE_DEVICES=0,1,4,5 uv run --active --no-sync scripts/train.py pi05_aloha_wbcd_lora \
  --project-name=EvoRL-Piper \
  --exp-name=<实验名称> \
  --wandb-enabled \
  > /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/tmp/train_pi05_A.log 2>&1
```

**参数说明：**
- `batch_size=64` (config 里定义), `fsdp_devices=4` → 4 卡 FSDP
- 如需改用 2 卡：`fsdp_devices=2`，`CUDA_VISIBLE_DEVICES` 设 2 张卡

## 在火山云引擎H20队列启动训练
入口命令：
cd /drobotics-ailab/lingyue.yang/Evo-RL-Piper/third_party/openpi
source .venv/bin/activate

python scripts/train.py pi05_aloha_wbcd_4cam_lora --project-name=EvoRL-Piper --exp-name=evorl_pi05_lora_alicia_dual_piper_0330_merged_260331_h20 --batch-size=64 --fsdp-devices=4 --num-train-steps=20000 --save-interval=2000 --wandb-enabled

环境变量：
HF_HOME = /drobotics-ailab/lingyue.yang/cache/huggingface
HUGGINGFACE_HUB_CACHE = /drobotics-ailab/lingyue.yang/cache/huggingface/hub
TRANSFORMERS_CACHE = /drobotics-ailab/lingyue.yang/cache/huggingface/transformers
TORCH_HOME = /drobotics-ailab/lingyue.yang/cache/torch
UV_CACHE_DIR = /drobotics-ailab/lingyue.yang/cache/uv
HF_ENDPOINT = https://hf-mirror.com

挂载的共享文件系统：vePFS 
容器内访问路径 /drobotics-ailab/lingyue.yang



## 服务器端开环测试

在服务器端，可以直接加载训练好的模型并在原始数据集上跑大批量测试（前向推理计算），以统计诸如 MSE, MAE 和 Cosine Similarity 的精准表现，无需启动 ZMQ 服务端或连接真实机器。

### 运行Openpi开环测试脚本

建议使用 `tmux` 会话运行以防止网络断开导致测试中断。该脚本将会加载模型并用验证集进行测试。
/data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/outputs/train/2026-03-31/05-52-44_act/checkpoints/060000
```bash
# 开启 tmux (可选但推荐)
tmux new-session -s open_loop_eval

# 进入环境
cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper/third_party/openpi
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
# 指定单卡或多卡GPU测试（通常测试只需1张）
export CUDA_VISIBLE_DEVICES=0

# 回到项目根目录执行脚本
cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper
python scripts/open_loop_eval.py \
  --checkpoint-dir third_party/openpi/checkpoints/<你的检查点config_name>/<你的检查点名称> \
  --step <检查点步数,例如:19999> \
  --config-name <你的检查点config_name> \
  --repo-id <你注册的训练集名称> \
  --episodes 0,1,2 \
  --prompt 'shirt open middle and catch' \
  --device cuda \
  --output-dir tmp/open_loop_eval 2>&1 | tee tmp/open_loop_eval.log
```

**关键参数说明：**
- `--checkpoint-dir`: 对应的训练输出目录（该目录下包含诸如 `1000/`, `2000/` 等步数的文件夹）。
- `--step`: 测试的具体步数。
- `--config-name`: 选择的训练配置（需与训练时保持对应，用于解析观察空间与动作空间）。
- `--repo-id`: 验证用的 LeRobot 数据集名称或绝对路径。
- `--episodes`: 指定测试的片段序号（逗号分隔，如 `0,1,2`）。如果需要全测，可以使用 `--episodes all`。
- `--prompt`: 当前任务对应的自然语言 prompt。
- `--output-dir`: 测试产生的动作误差对比曲线以及 JSON 格式数值统计将被输出到该目录下。由于其会自动依 `step` 分目录存储，建议直接配置一个公共临时目录。

执行完毕后，控制台会输出总体与各个维度的 MSE、MAE 和 Cosine Similary，并在指定的 `output-dir` 下面输出每个维度的残差对比折线图。

### 运行 LeRobot 模型的开环测试 (ACT/Diffusion/VLA)

与 OpenPI 的开环测试相似，我们也支持对基于 LeRobot 训练得到的模型执行一键开环测试并生成曲线。这会自动验证策略在前向推理和动作 Chunking 控制上的拟合程度。

```bash
# 开启 tmux (可选但推荐)
tmux new-session -s open_loop_eval_lerobot

# 进入主环境
cd /data/vepfs/users/intern/lingyue.yang/Evo-RL-Piper
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

# 执行 LeRobot 开环测试脚本
python scripts/lerobot_open_loop_eval.py \
  --checkpoint-dir outputs/train/2026-03-31/05-52-44_act/checkpoints/060000 \
  --repo-id alicia_dual_piper_0330_merged_lerobot \
  --dataset-root /data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD \
  --episodes 0,1,2 \
  --device cuda \
  --output-dir tmp/open_loop_eval_lerobot 2>&1 | tee tmp/open_loop_eval_lerobot.log
```

**关键参数说明：**
- `--checkpoint-dir`: 你的 LeRobot checkpoint 目录（需要指向到具体 step 层级例如 `checkpoints/060000`）。
- `--repo-id`: 你在 V3.0 转换后的数据集文件夹名称。
- `--dataset-root`: 存放该数据集的上一级父目录绝对路径。
- `--episodes`: 测试片段序号，格式同上。
- `--left-arm-dims`: 如果想监测左机械臂的指标（默认 `7:14`），可以通过传递它覆写左臂维度位置。
- `--action-chunk-index`: (实验性) 指预测出的动作序列块中你想提取与当前 step 进行对照的第几个前向步，默认是 `0` (即提取出网络给出的当前时刻控制指令进行误差评估)。

执行后，结果图片将被存放到 `tmp/open_loop_eval_lerobot/step_xxxx` 文件夹中，并在终端打印详细数值。

## 真机部署客户端与客户端开环测试

为了在本地/部署机器上开启推理服务或测试，需将服务器上的文件同步至本地（除数据集和 ROS 工作空间代码外）。

### 1. 从服务器打包必需文件

需要同步的核心内容包括：
1. **归一化统计文件 (norm_stats.json)**：
   通常位于 `third_party/openpi/assets/<config_name>/<PIPER_REPO_ID>/norm_stats.json`。该文件通过 `TrainConfig.assets_dirs` 加载，不可或缺。
2. **OpenPI 核心代码与自定义改变**：
   至少包含 `src/openpi/`（含 `config.py` 和定制的 `piper_policy.py`及 `transforms` 等）、`scripts/`、`inference/`（推理服务端如 ZMQ 代码）、以及用于锁定依赖的 `pyproject.toml` / `uv.lock`。 
3. **模型检查点 (Checkpoint)**（权重目录）。

**快捷打包命令示例：**
（在 `Evo-RL-Piper/third_party` 目录下执行）
```bash
# 1. 提取所有 OpenPI 代码及环境配置文件（排除冗余项）
tar -czvf openpi_deploy_package.tar.gz \
    --exclude="openpi/.git" \
    --exclude="openpi/.venv" \
    --exclude="openpi/wandb" \
    --exclude="openpi/checkpoints" \
    openpi

# 2. 提取权重
tar -czvf openpi_checkpoint.tar.gz openpi/checkpoints/<config_name>/<ckpt_name>
```

### 2. 部署主机运行流程 (ZMQ 架构)

> *数据流： openloop_eval.py → ZmqClient → [ZMQ tcp] → openpi_realtime_inference_service.py → OpenPI Policy*

**步骤一：还原运行环境并启动 OpenPI Server**
在部署环境解压上述打包的代码和权重。
```bash
cd openpi
# 在本地按 uv.lock 重建环境即可，无需拖拽服务器端庞大的 .venv
uv sync
source .venv/bin/activate

# 启动推理服务器
bash scripts/run_server.sh \
    --config-name pi05_piper_grasp \
    --checkpoint-dir /path/to/your/downloaded/checkpoint
```

**步骤二：启动本地端调用客户端脚本（新终端）**
假设本地已有对应的测试框架（如 `ros2_ws/src/inference_runner/`）。
```bash
cd ros2_ws
python src/inference_runner/scripts/openloop_eval.py \
    --policy-type pi05 \
    --dataset-path /path/to/your/dataset \
    --episode-id 0 \
    --server-port 5650 \
    --prompt "your task prompt"
```