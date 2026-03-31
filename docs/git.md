# 仓库 Git 工作流指南 (Git Workflow Guide)

本项目采用基于 Fork 和 Pull Request 的工作流。由于本项目配置了上游 (`upstream`: `MINT-SJTU/Evo-RL.git`) 和个人/团队 Fork 仓库 (`origin`: `87361/Evo-RL-Piper.git`)，请遵循以下标准的协作开发规范。

## 1. 远程仓库状态
确保你已经配置了 `origin` 和 `upstream` 两个远程仓库。
可以通过以下命令检查当前的远程仓库配置：
```bash
git remote -v
```

## 2. 日常开发工作流

### 步骤 1: 同步主分支
在开始任何新功能的开发之前，请务必保证本地 `main` 分支是最新的，以防后续产生复杂的代码冲突。
```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

### 步骤 2: 创建特性分支
请尽量**避免**直接在 `main` 分支上开发。为每一个新功能或 Bug 修复创建一个独立命名的分支。
```bash
git checkout -b feature/your-feature-name
# 或者针对修复制定规范命名
git checkout -b fix/your-bugfix-name
```

### 步骤 3: 提交更改 (Commit)
按照 **Conventional Commits (约定式提交)** 规范编写 Commit Message。这有助于保持清晰且结构化的历史记录。

```bash
git add <需要提交的文件>
git commit -m "type(scope): description"
```

**常用的 Commit Type：**
- `feat`: 新增功能 (Feature)
- `fix`: 修复 Bug (Bugfix)
- `docs`: 文档变更 (如 README.md 或这篇 git.md)
- `style`: 代码格式变化（不影响代码运行逻辑的修改）
- `refactor`: 重构代码（既不是增加 feature，也不是修复 bug）
- `perf`: 性能优化
- `test`: 增加测试代码
- `chore`: 构建过程、依赖修改或一些琐碎的日常维护

**Commit Message 示例：**
- `feat(gui_phone): add open-loop evaluation script into pipeline`
- `fix(server): resolve variable scope issue in status callback`
- `docs(git): update git workflow guide`

### 步骤 4: 推送分支
将本地特性分支推送到你自己的 `origin` 仓库。
```bash
git push origin feature/your-feature-name
```

### 步骤 5: 提交 Pull Request (PR)
在 GitHub 的网页端，从你的 `origin` 仓库分支 向团队或上游 (`upstream`) 仓库的目标分支提交 Pull Request。

---

## 3. 常见操作与注意事项 (⚠️ 安全须知)

为了配合机器人助手的严格安全限制（Defensive Editing & User Approval），请在使用 Git 时注意：

### 3.1 大范围变更前的主动备份
如需撤销不可逆修改，严禁不备份就执行 `git reset --hard` 或 `git clean -fd`。**如果需要恢复原始状态或放弃大量本地修改，务必确认没有遗漏的、暂存的最新实验数据！**
- 丢弃单个工作区文件的修改: 
  ```bash
  git restore <file>
  ```
- 将文件从暂存区撤出 (但不丢弃文件内容):
  ```bash
  git restore --staged <file>
  ```

### 3.2 处理合并冲突 (Merge Conflicts)
如果你的 PR 或在同步 `main` 分支时遇到冲突：
1. 建议使用 Rebase 的方式同步最新主分支：`git fetch upstream && git rebase upstream/main`
2. 找到发生冲突的文件并手动排查 `<<<<<<< HEAD` 等标记并决定保留的修改。
3. 解决冲突后使用 `git add <file>` 标记为已解决，然后执行:
   ```bash
   git rebase --continue
   ```