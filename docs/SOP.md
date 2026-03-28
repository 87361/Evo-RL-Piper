# SOP: 从数据后处理、数据集分割、启动不同策略的训练到开环测试、真机部署打包

## 数据集处理图形化页面命令

PYTHONPATH=src python scripts/gui/episode_review/main.py --host 127.0.0.1 --port 18080

## 数据集管理与拉取（TOS）

从 TOS 拉取最新采集的数据集（以 `alicia_dual_piper_0316_batch1` 为例），请使用 `tosutil`，以增量下载模式拉取。
**注意目录的嵌套结构**：GUI 要求数据集存放在类似于 `.../WBCD/WBCD/<数据集目录>/<内部同名目录>/` 的双层嵌套结构下（即 `alicia_dual_piper_0316_batch1/alicia_dual_piper_0316_batch1/videos/`），因此拉取时的目标路径直接指定为第一层即可：

```bash
# 鉴于数据集较大，建议使用 tmux 后台防止断开
tmux new-session -s download_wbcd -d 'tosutil cp -r tos://drobotics-ailab/users/lingyue.yang/dataset/WBCD/alicia_dual_piper_0316_batch1/ /data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD/alicia_dual_piper_0316_batch1/ -u'

# 附加: 如果想查看下载进度，可以执行 `tmux attach -t download_wbcd`
```

## 启动 LeRobot 系列策略 (如 ACT, Diffusion, VLA 等)

### Step 1: 转换 v2.1 数据集到 v3.0 格式

通过 LeRobot 自带转换脚本进行原地转换：

```bash
PYTHONPATH=src python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id <数据集的子目录名,例如A> \
    --root <存放数据集A的父目录的绝对路径> \
    --push-to-hub false
```
*(注：转换过程会自动将旧数据移动到 `<repo-id>_old` 目录中，并将新的 v3.0 格式生成在原名的 `<repo-id>` 目录里。)*

### Step 2: 规范底层特征命名

由于 LeRobot 默认提取且只处理名为 `observation.state` 的本体位姿键名，而我们的历史数据中采用的是 `agent_pos`，这里强烈建议跑一个脚本将底层 `parquet` 数据中的列名彻底一次性冲刷提纯。详细参考代码见 👉 `docs/260328.md`。

```bash
# 修改 `docs/260328.md` 提供的脚本中的 path 路径，然后一键执行：
python /tmp/convert_parquet_keys.py
```

### Step 3: 一键启动对应策略

保证数据格式已经无缝贴合 LeRobot 的原生 Pipeline 设计，接下去即可使用干净整洁的方式训练对应模型，且零跑不报错。我们提供两种方式：依赖系统与依赖 `uv`。

#### A: 训练传统免依赖模型 (ACT / Diffusion)

如果模型本身不需要比如大型 `transformers` 依赖，直接在 `evo-rl` 现有环境运行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=<任意取名> \
  --dataset.root=/绝对路径/你的v3.0格式数据集文件夹/ \
  --dataset.revision=v3.0 \
  --policy.type=act \
  --policy.device=cuda \
  --steps=100000 \
  --batch_size=8 \
  --num_workers=2
```

#### B: 训练大型依赖模型 (SmolVLA / XVLA)

对于依赖 `transformers` 等庞大依赖组的 VLA 模型，建议使用轻量的 `uv` 无痕执行。`uv` 将帮你在后台10秒内建立纯净沙盒而绝不弄乱或拖慢现存的 `conda` 环境！

```bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

# 注意 --extra 后面写对应的策略名字如 smolvla 或 xvla
uv run --extra smolvla python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=<任意取名> \
  --dataset.root=/绝对路径/你的v3.0格式数据集文件夹/ \
  --dataset.revision=v3.0 \
  --policy.type=smolvla \
  --policy.device=cuda \
  --steps=100000 \
  --batch_size=2 \
  --num_workers=2
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

## 开环测试 & 真机部署文件准备

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
