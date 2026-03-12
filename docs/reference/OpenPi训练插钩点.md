## 一、核心组件位置

**训练入口文件：**
- JAX: [scripts/train.py](scripts/train.py#L194) (`main()` 函数)
- PyTorch: [scripts/train_pytorch.py](scripts/train_pytorch.py#L625) (`main()` 函数)

**数据集类：**
- LeRobot Dataset: [src/openpi/training/data_loader.py#L130](src/openpi/training/data_loader.py#L130) (`create_torch_dataset` → `lerobot_dataset.LeRobotDataset`)
- RLDS Dataset: [src/openpi/training/droid_rlds_dataset.py#L36](src/openpi/training/droid_rlds_dataset.py#L36) (`DroidRldsDataset`)
- Wrapper: [src/openpi/training/data_loader.py#L53](src/openpi/training/data_loader.py#L53) (`TransformedDataset`)

**Sampler：**
- 当前使用 PyTorch 标准 `torch.utils.data.Sampler`
- DistributedSampler: [src/openpi/training/data_loader.py#L289](src/openpi/training/data_loader.py#L289)

**Collator：**
- `_collate_fn`: [src/openpi/training/data_loader.py#L471](src/openpi/training/data_loader.py#L471)

**Trainer：**
- JAX: [scripts/train.py#L137](scripts/train.py#L137) (`train_step`)
- PyTorch: [scripts/train_pytorch.py#L309](scripts/train_pytorch.py#L309) (`train_loop`)

---

## 二、数据从磁盘到 Batch 的完整路径

```
磁盘
  ↓
[LeRobotDataset] lerobot 库加载原始数据
  ↓
[TransformedDataset] 应用 repack_transforms (字段映射)
  ↓
[TransformedDataset] 应用 data_transforms (机器人特定变换)
  ↓
[TransformedDataset] 应用归一化 (norm_stats)
  ↓
[TransformedDataset] 应用 model_transforms (tokenization 等)
  ↓
[TorchDataLoader.__iter__] 通过 sampler 采样索引
  ↓
[TorchDataLoader] 通过 Dataset.__getitem__ 获取样本
  ↓
[_collate_fn] 将样本堆叠成 batch
  ↓
[batch] 传入 train_step
```

关键文件：
- Dataset 创建: [src/openpi/training/data_loader.py#L130-153](src/openpi/training/data_loader.py#L130-153)
- Transform 应用: [src/openpi/training/data_loader.py#L172-201](src/openpi/training/data_loader.py#L172-201)
- DataLoader: [src/openpi/training/data_loader.py#L381-452](src/openpi/training/data_loader.py#L381-452)

---

## 三、三池采样的最自然插入点

**最佳插入层：Sampler 层**，具体在 [src/openpi/training/data_loader.py#L288-307](src/openpi/training/data_loader.py#L288-307)

**原因：**

1. **职责分离**：APO 的三池采样是**数据选择逻辑**，不属于数据变换或模型计算，应该和现有的 `DistributedSampler` 在同一抽象层级

2. **最小影响**：Sampler 是数据访问的入口层，修改这里不会影响：
   - 底层的 LeRobotDataset（数据格式）
   - Transform pipeline（数据增强、归一化）
   - 模型代码（前向传播）

3. **架构一致性**：PyTorch 生态中，定制采样器的标准做法就是继承 `torch.utils.data.Sampler`

**实现方式：**
```python
class BalancedAPOSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, correct_ratio=0.5, ...):
        # 初始化三个索引列表
        self.correct_indices = [...]
        self.interaction_indices = [...]
        self.incorrect_indices = [...]
        
    def __iter__(self):
        # 按比例采样并打乱
        yield from self._sample_batch()
```

然后在 [data_loader.py#L288](src/openpi/training/data_loader.py#L288) 替换 `DistributedSampler`：

```python
sampler = BalancedAPOSampler(dataset, local_batch_size, ...)
```

**为什么不选择其他层：**

| 层级 | 缺点 |
|------|------|
| Dataset.__getitem__ | 需要在数据加载时动态判断样本类型，破坏了 Dataset 的随机访问语义 |
| TransformedDataset | Transform 应该是无状态的，不应包含采样逻辑 |
| Collator | 此时索引已确定，无法重采样 |
| Train Loop | 太晚了，batch 已经构造好 |

---

## 四、最小修改清单

假设你的布料数据已经转换为 LeRobot 格式，以下是必须修改的文件：

**必须修改（3个文件）：**

1. **[src/openpi/training/config.py](src/openpi/training/config.py)**
   - 新增 `ClothDataConfig` 类（参考 `LeRobotAlohaDataConfig`#L229）
   - 在 `_CONFIGS` 列表添加新的训练配置

2. **[src/openpi/training/data_loader.py](src/openpi/training/data_loader.py)**
   - 新增 `create_cloth_dataset` 函数（类似 `create_torch_dataset`#L130）
   - 新增 `BalancedAPOSampler` 类（放在文件顶部）

3. **数据转换脚本**（新建）
   - 参考 [examples/libero/convert_libero_data_to_lerobot.py](examples/libero/convert_libero_data_to_lerobot.py)
   - 将你的布料数据转换为 LeRobot 格式

**可选修改（1个文件）：**

4. **[src/openpi/transforms.py](src/openpi/transforms.py)**
   - 如果你的数据需要特殊的归一化或增强（如布料的拉伸变换），添加新的 Transform

**完全不需要改：**
- 模型代码
- 训练循环
- Collator（如果 batch 字段结构一致）

---

## 五、模型绑定 vs 外部 Adapter

**与模型强绑定的部分（不应该提取为 adapter）：**

1. **Tokenization** [src/openpi/transforms.py#L248-268](src/openpi/transforms.py#L248-268)
   - `TokenizePrompt`, `TokenizeFASTInputs` 直接依赖模型架构
   - 这些是模型输入的契约，必须严格匹配

2. **Action Pad/Delta 变换** [src/openpi/transforms.py#L328-337](src/openpi/transforms.py#L328-337)
   - `PadStatesAndActions` 依赖 `model.action_dim`
   - `DeltaActions/AbsoluteActions` 影响模型学习的表示空间

3. **Model Transform Group** [src/openpi/training/config.py#L107-163](src/openpi/training/config.py#L107-163)
   - `ModelTransformFactory` 中的变换序列是模型特定的

**适合作为外部 Adapter 的部分：**

1. **Repack Transforms** [src/openpi/training/config.py#L241-253](src/openpi/training/config.py#L241-253)
   ```python
   repack_transforms = _transforms.Group(
       inputs=[_transforms.RepackTransform({
           "my_cloth_image": "observation.image",
           "cloth_state": "observation.state",
           ...
       })]
   )
   ```
   纯粹的字段映射，完全可插拔

2. **Data Transforms** [src/openpi/training/config.py#L259-262](src/openpi/training/config.py#L259-262)
   ```python
   data_transforms = _transforms.Group(
       inputs=[ClothInputs(...)],  # 自定义
       outputs=[ClothOutputs(...)]
   )
   ```
   布料特定的状态/动作变换

3. **Sampler**
   - 完全独立，通过 `sampler` 参数传入

4. **归一化统计量** [src/openpi/transforms.py#L115-145](src/openpi/transforms.py#L115-145)
   - 通过 `norm_stats` 配置，可以是预计算的

**分离策略总结：**

```
模型边界
  ↓
【Model Transforms】 ← 强绑定
【Normalization】  ← 中立（stats 可以是外部的）
  ↓
【Data Transforms】 ← 适配器层（布料特定）
【Repack Transforms】 ← 适配器层（字段映射）
【Sampler】 ← 适配器层（采样策略）
  ↓
磁盘数据
```

---

## 数据流路径 + 可插桩点 + 最小修改清单

```
[布料数据磁盘]
  ↓
【可插桩点1】数据转换脚本（新建）
  convert_cloth_to_lerobot.py → LeRobot 格式
  ↓
[LeRobotDataset] (lerobot 库)
  ↓
【可插桩点2】repack_transforms (config.py)
  字段映射：cloth_* → observation.*
  ↓
【可插桩点3】data_transforms (config.py)
  布料状态变换
  ↓
[归一化] (transforms.py)
  使用预计算的 norm_stats
  ↓
【可插桩点4】model_transforms (config.py)
  Tokenization（不修改）
  ↓
【可插桩点5】BalancedAPOSampler (data_loader.py 新增)
  correct/interaction/incorrect 三池采样
  ↓
[TorchDataLoader]
  ↓
[collate_fn]
  ↓
[train_step]
```

**最小修改清单：**

| 文件 | 修改类型 | 代码量估计 |
|------|----------|-----------|
| `src/openpi/training/config.py` | 新增类 `ClothDataConfig` + 注册配置 | ~50 行 |
| `src/openpi/training/data_loader.py` | 新增 `BalancedAPOSampler` 类 | ~100 行 |
| `examples/convert_cloth_to_lerobot.py` | 新建 | ~200 行 |
| `src/openpi/transforms.py` | 可选：新增布料 Transform | 0-50 行 |

**总计：** 核心修改主要集中在配置和数据加载层。

---

## 下一步建议

1. **查看数据转换示例**：[数据转换框架](10-data-transforms-framework) 了解 transform 机制
2. **深入归一化细节**：[归一化与预处理](11-normalization-and-preprocessing) 理解 norm_stats 的作用
3. **学习配置系统**：[训练配置系统](19-training-configuration-system) 掌握如何定义新配置

**关键洞察：** 这个仓库的设计非常模块化。APO 的三池采样和你的布料数据都可以通过配置和 Adapter 层注入，核心训练逻辑几乎不需要改动。你需要在"适配器层"（Data Transforms + Sampler）做文章。