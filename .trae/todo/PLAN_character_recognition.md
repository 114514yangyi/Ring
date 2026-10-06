# 轨迹→字母识别 Implementation Plan(任务 2 第一阶段)

> **For agentic workers:** 逐任务实现,每任务 TDD:先写失败测试再实现;不执行 git commit(用户未授权)。
> **Spec:** `.trae/documents/SPEC_character_recognition.md`

**Goal:** 接触段轨迹切分为逐字符样本,训练 26 类字母分类器;GT 与重建轨迹均评估,用户无重叠为主。

**Tech Stack:** Python 3.10(conda vae)、PyTorch 2.2、numpy、unittest;运行方式同任务 1(`/tmp/opencode/ringcheck` 软链 + `PYTHONPATH`)。

## Global Constraints

- 六通道/数据约定沿用任务 1;标签只取 action 0/1,`wrong` 丢弃,字符统一大写(26 类)。
- 重建轨迹用 `models_trajectory_topology/x3/paper_trajectory.pt`(可在 CLI 指定);不修改轨迹模型代码。
- 样本切分必须过滤:时间范围外、无接触段、接触帧 < 8。
- 不提交 git;文档中文,代码风格对齐现有模块。

## Review Focus

1. 标签时间戳落在 clean 范围外(开头裁剪)必须跳过而非错位(测试覆盖)。
2. `wrong` 标签与多笔画字符的分组正确性(测试覆盖 1/2/3 段)。
3. 重采样/归一化的退化输入(零长度轨迹、单点)不除零(测试覆盖)。
4. 用户无重叠/随机划分互斥且确定性(测试覆盖)。
5. 重建轨迹与 GT 轨迹在评估中不可混淆(按 source 记录,report 分列)。

---

### Task 1: `character_dataset.py`(切分/特征)

**Files:** Create `character_dataset.py`; Test `test_character_classifier.py`

**Produces:**
- `CharacterSample(user, action, sample_id, label:int, char:str, runs:tuple[(start,end)])`
- `contact_runs(mask, min_frames=8) -> list[(start,end)]`
- `discover_character_samples(clean_root, raw_root, actions=(0,1), min_contact_frames=8) -> list[CharacterSample]`
- `resample_trajectory(points, n) -> np.ndarray[n,2]`(弧长重采样,退化输入返回重复点)
- `normalize_trajectory(points) -> np.ndarray`(首点归零 + bbox 对角线缩放)
- `extract_gt_trajectory(sample, board, resample_frames=64) -> np.ndarray`(逐段相对首点拼接)
- `build_gt_examples(samples, resample_frames=64) -> list[dict]`(含 trajectory/label/user/sample 元数据)

**Steps:** 写测试(合成 mask/标签文件)→ 跑到失败 → 实现 → 全绿 → 任务完成。

### Task 2: `character_model.py`(分类器)

**Produces:** `CharacterCNN`(2→32→64→128 Conv1d+BN+ReLU,全局池化,FC26);`build_character_summary()`;`load_character_model()`。
**Tests:** 前向形状/概率和、checkpoint 往返。

### Task 3: `train_character_classifier.py`(训练/评估)

**Produces:**
- `character_split(samples, strategy, seed=42) -> (train,val,test,info)`(随机按样本 / 用户无重叠)
- `predict_run_points(predictor, x_block, chunk_len=5000) -> list[np.ndarray]`(逐接触段积分,段内首点归零)
- `build_recon_examples(samples, run_points_by_sample, resample_frames=64) -> list[dict]`
- `train(args) -> report`;CLI:`--data-root --raw-root --reconstruction-checkpoint --split {random,user} --source {gt,recon,both} --epochs 60 --batch-size 128 --output-dir models_character`
- report:split、样本数、source、history、test 的 top1/top3/confusion(按 GT/recon 分列)
**Tests:** split 互斥/确定性;`topk_accuracy`/混淆矩阵函数;`predict_run_points` 常位移合成测试。
**Smoke:** `--source gt --epochs 2 --source gt` 生成 report。

### Task 4: 正式训练与评估

- 用户无重叠 `--source both`(混训)与 `--source gt`;随机划分对照;记录 best report。
- 可选:重建轨迹单独评估;文档更新(NOTES 结果、README 用法、STRUCTURE、TODO)。

### Task 5: 文档与回归

README/STRUCTURE/DESIGN/NOTES/TODO 同步;全量 unittest 回归。
