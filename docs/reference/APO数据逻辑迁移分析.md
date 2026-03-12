## APO 数据逻辑 Engineering Pattern 抽取

### 核心问题答案

**1. Correct / Interaction / Incorrect 三类样本构建**

| 样本类型 | 数据源 | 关键标记 | 索引列表 | 代码位置 |
|---------|-------|---------|----------|----------|
| **Correct** | `data_root_dir/*.h5` | `is_expert: True` | `correct_data_idx_list` | [balance_apo_dataset.py#L124-L144](dataset/balance_apo_dataset.py#L124-L144) |
| **Interaction** | `interaction_root_dir/*.h5` | `is_human == 1` | `interaction_data_idx_list` | [balance_apo_dataset.py#L151-L171](dataset/balance_apo_dataset.py#L151-L171) |
| **Incorrect** | **从 correct 动态迁移** | `is_human == 0` | `incorrect_data_idx_list` | [balance_apo_dataset.py#L156-L163](dataset/balance_apo_dataset.py#L156-L163) |

**关键函数**：
- `BalanceAPODataset.__init__()`：数据加载、分类、K-step 回溯
- `BalanceAPODataset.get_balance_list()`：返回三类索引
- `BalancedInteractionDistributedSampler.__iter__()`：按比例组合 batch

---

**2. K-step Pre-intervention Relabel 精确触发条件**

```python
# dataset/balance_apo_dataset.py#L156-L163
is_first_human = True
for timestep in range(timestep_len):
    is_human = interaction_data[f'timestep_{timestep}']['is_human'][()]
    
    # 触发条件：检测到第一个 human intervention
    if is_first_human and is_human == 1:
        is_first_human = False
        # 回溯标记前 K 步为 failure
        for k in range(1, previous_K + 1):
            data_corresponding[-k]['is_human'] = 0
            data_idx = self.correct_data_idx_list.pop()
            self.incorrect_data_idx_list.append(data_idx)
    
    # 重置标志：遇到 success 标记（is_human == 2）时
    if is_human == 2:
        is_first_human = True
```

**精确触发逻辑**：
1. 遍历 interaction 数据时维护 `is_first_human` 标志
2. **第一次遇到 `is_human == 1` 时触发**
3. 回溯 `data_corresponding` 最后 K 个样本（此时已添加到列表）
4. 修改 `is_human = 0`，并将索引从 `correct_data_idx_list` 迁移到 `incorrect_data_idx_list`
5. **重置**：遇到 `is_human == 2`（success）时重置 `is_first_human = True`，支持同一 episode 多个 intervention 段

---

**3. Sampler 保证 Batch 内比例固定的机制**

```python
# dataset/sampler.py#L145-L210 (关键循环)
while len(use_interaction_indices_list) > 0:
    # 1. 以 interaction 为锚点，取固定数量
    batch_interaction = use_interaction_indices_list[:self.interaction_select_num]
    use_interaction_indices_list = use_interaction_indices_list[actual_interaction_num:]
    
    # 2. 从 correct 列表循环采样
    while collected_correct < needed_correct:
        if not temp_correct_indices_list:
            temp_correct_indices_list = use_correct_indices_list.copy()
        take_now = min(needed_correct - collected_correct, len(temp_correct_indices_list))
        batch_correct.extend(temp_correct_indices_list[:take_now])
    
    # 3. 从 incorrect 列表循环采样
    # (同 correct 逻辑)
    
    # 4. 组合 batch
    batch = batch_correct + batch_incorrect + batch_interaction
```

**保证比例的三层机制**：
1. **Interaction 锚定**：以最少的数据类别为驱动，每步固定取 `interaction_select_num`
2. **循环补充**：correct/incorrect 用临时列表 `temp_*_indices_list` 循环采样
3. **截断控制**：每个 sub-step 的采样量不超过 `needed_correct`

---

**4. 扩展 Task Phase / Retry Count / Quality Score 的最佳位置**

**推荐方案**：在 `data_corresponding` 字典中扩展字段

```python
# dataset/balance_apo_dataset.py#L140-L170 (改造建议)
data_corresponding.append({
    # === 原有字段（可直接复用）===
    "action": action,
    "timestep": timestep,
    "language": task_language_prompt,
    "task_name": interaction_data_name,
    "demo_id": demo,
    "is_expert": False,
    "is_human": is_human,
    
    # === 新增字段 ===
    "task_phase": extract_task_phase(timestep, episode_metadata),  # 从 episode 元数据提取
    "retry_count": current_retry_count,                             # 当前重试次数
    "quality_score": compute_quality_score(action, ground_truth),    # 动作质量评分
    "pre_intervention_window": pre_intervention_mask,               # 布尔掩码
    "gripper_state": extract_gripper_state(observation),             # gripper 状态
    "ee_pose": extract_end_effector_pose(observation),              # 末端位姿
    "joint_state": extract_joint_state(observation),                 # 关节状态
})
```

**扩展点选择理由**：
- ✅ `data_corresponding` 是**核心数据结构**，所有采样和加载都基于它
- ✅ 在 `__init__` 中填充，与原有逻辑耦合度低
- ✅ 后续 `__getitem__` 可以直接访问这些字段
- ✅ 如果需要按 phase/quality 采样，只需在 `sampler` 中添加新的索引列表

---

## 三栏分类：可参考复用 / 需要改造 / 不建议照搬

| 类别 | 组件 | 可复用逻辑 | 需改造逻辑 | 不建议照搬逻辑 |
|------|------|-----------|-----------|---------------|
| **数据分类** | 三类索引列表架构 | ✅ 独立维护 `correct_idx_list`、`interaction_idx_list`、`incorrect_idx_list` | - | - |
| | K-step 回溯机制 | ✅ `is_first_human` 状态机设计<br>✅ 回溯修改 `is_human` 标记 | ⚠️ 布料操作可能需要更复杂的失败检测（如布料撕裂、轨迹漂移） | ⚠️ `is_human == 2` 重置逻辑可能不适合双臂协作场景 |
| **采样器** | 比例固定机制 | ✅ 以最少类别为锚点（interaction）<br>✅ 临时列表循环采样<br>✅ 分布式切分逻辑 | ⚠️ 布料操作可能需要 4+ 类别（左右臂分治） | - |
| | Batch 组合逻辑 | ✅ `batch = correct + incorrect + interaction` | - | - |
| **数据结构** | `data_corresponding` 字典 | ✅ timestep 级粒度设计<br>✅ 包含 action、observation、language | - | ❌ 简单的 `is_human ∈ {0,1,2}` 标记过于粗糙 |
| **预处理** | 动作归一化 | ✅ quantile-based 归一化（q01/q99） | ⚠️ 布料操作可能需要双臂联合归一化 | - |
| | 图像增强 | ✅ RandomResizedCrop + ColorJitter | ⚠️ 布料操作可能需要几何变换（透视、旋转） | - |
| **Collator** | Padding 逻辑 | ✅ `pad_sequence` + `IGNORE_INDEX` | - | - |
| | Batch 组装 | ✅ pixel_values、input_ids、labels 组装 | - | ❌ `mismatch_label = labels[indices]` 是 VLA 特有的对比损失，不适合通用场景 |
| **Tokenization** | Prompt 构建 | ❌ 完全绑定 OpenVLA 的对话格式 | - | ❌ `prompt_builder.add_turn("human"/"gpt")` 是多模态对话特有 |
| | Action tokenization | ❌ 完全绑定离散动作空间 | - | ❌ 布料操作可能是连续动作（force control） |
| **损失函数** | `is_correct` 标记 | ⚠️ 可复用二分类标记 | ⚠️ 需扩展为 multi-label（phase、quality） | - |
| | `wrong_input_ids` / `wrong_labels` | ❌ 完全绑定 APO 的对比损失 | - | ❌ 你的布料操作可能不需要对比学习 |

---

## 迁移建议（双臂布料操作场景）

### ✅ 参考复用

1. **三类索引架构**：`correct_idx_list`、`interaction_idx_list`、`incorrect_idx_list` 完全通用
2. **K-step 回溯算法**：只需替换失败检测逻辑（从 `is_human == 1` 改为你的布料失败指标）
3. **采样器核心逻辑**：`BalancedInteractionDistributedSampler.__iter__` 可直接复制，调整类别数量即可
4. **量化归一化**：q01/q99 归一化适合连续动作空间

### ⚠️ 需要改造

1. **数据结构扩展**：
   ```python
   data_corresponding.append({
       # 原有字段
       "action": action,
       "timestep": timestep,
       ...
       
       # 双臂特有
       "arm_type": "left" / "right",           # 臂标识
       "cloth_state": "unfolded" / "folded",    # 布料状态
       "grasp_quality": score,                  # 抓取质量
       
       # 元数据
       "task_phase": "approach" / "grasp" / "manipulate" / "release",
       "retry_count": current_retry,
       "quality_score": trajectory_smoothness,
   })
   ```

2. **采样器扩展**：添加 `left_arm_idx_list`、`right_arm_idx_list` 实现双臂平衡采样

3. **Collator 适配**：移除 `mismatch_label`，添加 `phase_labels`、`quality_labels`

### ❌ 不建议照搬

1. **VLA Prompt 格式**：你的场景可能是简单的 "Action: [7d vector]"，不需要对话结构
2. **对比损失**：APO 的 `wrong_input_ids` 对比学习不适合布料操作，用简单的 BCE/MSE 即可
3. **离散动作 tokenization**：布料操作通常是连续力控，直接回归到实数空间
4. **单臂假设**：原代码假设单机器人，双臂需重构状态表示（`gripper_state` 改为 `left_gripper` + `right_gripper`）

---

## 推荐迁移步骤

1. **复制核心骨架**：`balance_apo_dataset.py` 的三类索引逻辑 + `sampler.py` 的比例采样
2. **改造数据加载**：替换 HDF5 结构为你的布料日志格式（可能是 JSON/ROS bag）
3. **扩展标记逻辑**：用布料失败指标（如撕裂检测、轨迹偏差）替换 `is_human == 1`
4. **调整数据结构**：在 `data_corresponding` 中添加双臂/布料特有字段
5. **简化 collator**：移除 VLA 特有的 `mismatch_label`，保留 padding 逻辑
6. **替换损失函数**：用简单的 `MSELoss` 或 `SmoothL1Loss` 替代 APO 对比损失

[平衡数据集采样策略](11-balanced-dataset-sampling-strategy)
[数据整理与批处理](19-data-collation-and-batching)
[APO 损失函数与计算](14-apo-loss-function-and-computation)