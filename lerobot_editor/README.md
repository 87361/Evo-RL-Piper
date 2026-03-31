# LeRobot 数据集帧级别裁剪工具 使用说明

## 一、工具简介

`lerobot_editor.py` 是一个独立的 LeRobot v2.1 数据集帧级别裁剪工具，不依赖 lerobot 库，支持对机器人采集的数据集进行精细化编辑。

### 支持的数据类型

- **Parquet 数据**：关节角度、动作、末端位姿等逐帧数值数据
- **MP4 视频**（`dtype: video`）：各摄像头的 RGB 视频
- **逐帧图片**（`dtype: image`）：深度图等以 PNG 存储的逐帧数据

### 支持的操作

| 模式 | 功能 |
|------|------|
| `inspect` | 查看数据集概览或某个 episode 的详细信息 |
| `trim` | 裁剪指定 episode 的帧（删除开头/尾部/中间的帧） |
| `remove` | 删除整个 episode |
| `batch` | 批量裁剪多个 episode |
| `verify` | 验证数据集完整性（检查 parquet、视频、图片是否一致） |

---

## 二、环境准备

### 2.1 激活 conda 环境

```bash
conda activate lerobot-editor
```

### 2.2 安装 Python 依赖

```bash
pip install pandas pyarrow numpy
```

### 2.3 安装 ffmpeg（视频裁剪必需）

```bash
conda install -c conda-forge ffmpeg -y
```

验证安装：

```bash
ffmpeg -version
```

---

## 三、使用方法

### 总体流程

1. 用编辑器打开 `lerobot_editor.py`
2. 修改顶部**【用户配置区】**的参数
3. 运行脚本：`python lerobot_editor.py`

所有参数都在代码顶部集中配置，不需要命令行参数。

---

## 四、配置参数详解

### 4.1 数据集路径

```python
DATASET_DIR = "/path/to/your_dataset"
```

填写你要操作的数据集根目录的绝对路径。该目录下应包含 `data/`、`videos/`、`meta/` 等子目录。

### 4.2 操作模式

```python
MODE = "trim"
```

修改引号内的值，可选：`inspect`、`trim`、`remove`、`batch`、`verify`。

### 4.3 输出方式

```python
COPY_TO_NEW_DIR = True
```

| 值 | 行为 | 适用场景 |
|----|------|---------|
| `True` | 先把数据集完整拷贝到新目录，再在新目录上修改。原数据不受影响。 | 推荐，安全 |
| `False` | 直接在原目录上修改（通过临时文件中转，不会误删，但修改不可逆）。 | 磁盘空间紧张或数据量大时使用 |

当 `COPY_TO_NEW_DIR = True` 时，可以自定义输出路径：

```python
OUTPUT_DIR = ""              # 留空则自动生成：原目录名 + OUTPUT_SUFFIX
OUTPUT_SUFFIX = "_trimmed"   # 默认后缀
```

例如原目录为 `/data/my_dataset`，输出目录自动为 `/data/my_dataset_trimmed`。

---

## 五、各模式操作示例

### 5.1 inspect — 查看数据集信息

**查看整个数据集概览：**

```python
MODE = "inspect"
INSPECT_EPISODE = None       # 设为 None 查看概览
```

输出示例：

```
数据集: /data/my_dataset
  格式版本: v2.1
  机器人类型: piper_dual
  FPS: 10
  总 episodes: 59
  总帧数: 8234
  视频特征 (MP4): ['observation.images.head_cam', ...]
  图片特征 (PNG): ['observation.depths.head_cam', ...]

  Episode 帧数列表:
    Episode   0:   200 帧
    Episode   1:   208 帧
    ...
```

**查看某个 episode 的详细信息：**

```python
MODE = "inspect"
INSPECT_EPISODE = 5          # 查看 episode 5
```

输出示例：

```
Episode 5:
  元数据帧数: 134
  Parquet 实际行数: 134
  视频 (dtype=video):
    observation.images.head_cam: ✓
    observation.images.left_wrist_cam: ✓
  图片 (dtype=image):
    observation.depths.head_cam: ✓ 134 帧
```

### 5.2 trim — 裁剪帧

```python
MODE = "trim"
TRIM_EPISODE = 5             # 要裁剪的 episode 编号
TRIM_FRAMES = "0-60"         # 要删除的帧范围
COPY_TO_NEW_DIR = True       # 推荐设为 True
```

**帧范围写法：**

| 写法 | 含义 | 示例 |
|------|------|------|
| `"0-50"` | 删除开头 50 帧（第 0~49 帧） | 去掉机器人启动前的空闲帧 |
| `"200-end"` | 删除第 200 帧到末尾 | 去掉任务完成后的多余帧 |
| `"100-200"` | 删除中间第 100~199 帧 | 去掉中间异常的数据段 |

**裁剪过程中脚本会自动完成：**

1. 裁剪 Parquet 数据并重建 `frame_index` 和 `timestamp`
2. 用 ffmpeg 裁剪对应时间段的视频
3. 拷贝保留的帧图片并重新编号
4. 更新 `episodes.jsonl` 和 `info.json` 中的帧数
5. 重新计算所有 episode 的统计信息（`episodes_stats.jsonl`）

### 5.3 remove — 删除整个 episode

```python
MODE = "remove"
REMOVE_EPISODE = 3           # 要删除的 episode 编号
COPY_TO_NEW_DIR = True
```

会删除该 episode 的 parquet 文件、所有视频文件、所有图片目录，并更新元数据。

### 5.4 batch — 批量裁剪

```python
MODE = "batch"
BATCH_TASKS = [
    {"episode": 0, "frames": "0-30"},       # episode 0 删除开头 30 帧
    {"episode": 5, "frames": "0-60"},       # episode 5 删除开头 60 帧
    {"episode": 12, "frames": "180-end"},   # episode 12 删除尾部
    {"episode": 20, "frames": "50-80"},     # episode 20 删除中间段
]
COPY_TO_NEW_DIR = True
```

脚本会按顺序依次处理每个任务。如果某个 episode 不存在或帧范围无效，会跳过并继续处理下一个。

### 5.5 verify — 验证数据集完整性

```python
MODE = "verify"
```

检查项目包括：

- 每个 episode 的 Parquet 文件是否存在，行数是否与元数据一致
- 每个摄像头的视频文件是否存在
- 每个深度图目录的帧数是否与元数据一致
- `info.json` 中的 `total_frames` 是否正确

输出示例：

```
正在验证: /data/my_dataset_trimmed
  共 59 个 episode

  Episode   0: ✓ 正常 (170 帧)
  Episode   1: ✓ 正常 (208 帧)
  Episode   5: ✓ 正常 (74 帧)
  ...

✔ 验证通过，无错误
```

---

## 六、ffmpeg 编码参数

如果需要调整视频裁剪的编码质量或速度，修改以下参数：

```python
FFMPEG_VCODEC = "libx264"    # 视频编码器
FFMPEG_PRESET = "fast"       # 编码速度预设
FFMPEG_CRF = "18"            # 视频质量
```

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `FFMPEG_VCODEC` | `libx264`、`libx265`、`libsvtav1` | 编码器，`libx264` 兼容性最好 |
| `FFMPEG_PRESET` | `ultrafast`、`fast`、`medium`、`slow` | 越慢质量越好，但编码时间越长 |
| `FFMPEG_CRF` | `0`~`51` | 0=无损，18=高质量，23=默认，28=低质量 |

---

## 七、典型工作流

### 场景：清洗一批采集数据

```
第一步：inspect 查看数据集，了解每个 episode 的帧数
        ↓
第二步：用可视化工具（如 lerobot-data-studio）逐个回放 episode
        标记哪些 episode 需要裁剪开头/尾部，哪些需要整个删除
        ↓
第三步：配置 batch 模式的 BATCH_TASKS，填入所有裁剪任务
        设置 COPY_TO_NEW_DIR = True
        ↓
第四步：运行 python lerobot_editor.py，等待批量处理完成
        ↓
第五步：verify 验证裁剪后的数据集完整性
        ↓
第六步：用裁剪后的数据集进行模型训练
```

---

## 八、注意事项

1. **首次裁剪建议用 `COPY_TO_NEW_DIR = True`**，确认结果正确后再考虑原地操作。

2. **裁剪是针对单个 episode 的**，不同 episode 之间互不影响，各 episode 裁剪后长度可以不一致。

3. **帧索引从 0 开始**，`"0-50"` 表示删除第 0、1、2...49 帧，共 50 帧。

4. **没有深度图的数据集也能用**，脚本会自动跳过不存在的图片目录，不会报错。

5. **批量裁剪时注意顺序**，如果对同一个 episode 裁剪两次，第二次的帧范围应基于第一次裁剪后的帧数。

6. **磁盘空间**：`COPY_TO_NEW_DIR = True` 会拷贝整个数据集，确保磁盘有足够空间（至少与原数据集等大）。

---

## 九、常见问题

### Q: 运行报错 `FileNotFoundError: ffmpeg`

A: ffmpeg 没有安装。运行 `conda install -c conda-forge ffmpeg -y` 安装。

### Q: 视频裁剪显示 `✗ 失败`

A: 可能原因：ffmpeg 未安装、视频文件损坏、磁盘空间不足。运行 `ffmpeg -version` 确认安装，检查磁盘空间 `df -h`。

### Q: 图片裁剪显示 `0 帧`

A: 该数据集没有对应的图片数据（`images/` 目录不存在或为空），这是正常的，不影响其他数据的裁剪。

### Q: 裁剪后能否撤销？

A: 如果用了 `COPY_TO_NEW_DIR = True`，原数据完好，删掉输出目录即可"撤销"。如果用了 `False`，修改不可逆，需要从备份恢复。

### Q: 能否同时裁剪开头和尾部？

A: 不能一次性完成，但可以用 batch 模式对同一个 episode 做两次裁剪。先裁剪开头，再裁剪尾部（注意第二次的帧范围要基于第一次裁剪后的帧数）。

### Q: 支持 v3.0 格式吗？

A: 不支持。v3.0 格式将多个 episode 合并到一个文件中，裁剪操作更复杂。建议在 v2.1 格式上裁剪完成后，再用 LeRobot 官方工具转换为 v3.0。