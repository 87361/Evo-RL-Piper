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
