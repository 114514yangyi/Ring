# WritingRing 轨迹重建复现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在官方 `clean_data_delete_g` 数据上复现 WritingRing 的 TCN+LSTM 流式速度预测与接触段轨迹重建,并按论文指标评估。

**Architecture:** 数据层把每个 block 打包为 75 s(15000 帧)长片段,模型以 13 帧滑窗提取 TCN 特征(1×128),LSTM(128)跨窗流式预测窗口中点位移,训练按块做 TBPTT,仅接触帧计入 masked MSE;评估只在接触段积分预测位移并与 `board` 真值比较。

**Tech Stack:** Python 3.10(conda `vae`)、PyTorch 2.2、numpy、matplotlib(已有)、unittest。

**Spec:** `.trae/documents/SPEC_trajectory_reconstruction.md`

## Global Constraints

- 解释器:`/data/huyang/miniconda/envs/vae/bin/python`;仓库按包名 `writing_state` 导入。
- 测试与运行统一在临时挂载目录执行(父目录当前无软链):
  `mkdir -p /tmp/opencode/ringcheck && ln -sfn /data/huyang/trae_projects/Ring /tmp/opencode/ringcheck/writing_state`,工作目录 `/tmp/opencode/ringcheck`,`PYTHONPATH=/tmp/opencode/ringcheck`。
- 数据根:`/data/huyang/datasets/WritingRing/clean_data_delete_g/data`(不在仓库内;测试对真实数据用 `skipIf(not exists)` 保护)。
- **不执行 git commit**(用户未授权提交;每任务以测试通过为完成标志)。
- 新增依赖仅限 torch/numpy/pandas/matplotlib;标定任务可选用 `compress_pickle`,失败必须降级而不是阻塞主链路。
- 六通道顺序固定:`lin_acc_x/y/z, gyro_x/y/z`;采样率 200 Hz。
- 配置默认值以 Spec §4.1 为准:`window_frames=13`、`tcn_kernel=3`、`tcn_channels=(16,16,32,32,64,128)`、`feature_dim=128`、`lstm_hidden=128`、`dropout=0.2`、`board_mm_scale=(240.0, 169.5)`。
- 代码风格对齐 `paper_touch.py`/`paper_dataset.py`(英文 docstring、dataclass 配置、report.json 落盘)。

## Review Focus

1. 长度不一致/损坏样本(user_0/0/0 等):加载须跳过并在 report 计数,不得崩溃(Task 1)。
2. 极短样本(不足 13 帧)与接触段不足 5 帧:输出为空/无指标,不得 NaN 或除零(Task 1/3)。
3. GT bbox 对角线接近 0 的轨迹:归一化指标须有除零保护并统计跳过数(Task 3)。
4. TBPTT 分块:块长不整除序列长度时的余数块与 6 帧 margin 切片必须与整段前向等价(Task 2)。
5. 全 0 mask 或整段无接触:masked loss 分母为 0 时必须返回 0 loss(Task 4)。

---

### Task 1: 数据层 `paper_trajectory_dataset.py`

**Files:**
- Create: `paper_trajectory_dataset.py`
- Test: `test_paper_trajectory.py`

**Interfaces:**
- Consumes: 现有 `paper_dataset.discover_cleaned_samples` 的目录约定与 npy 命名(`user_*/{action}/*_x.npy` 等)。
- Produces:
  - `TrajectorySample(user, action, sample_id, x_path, board_path, mask_path, y_path, timestamp_path)`(frozen dataclass)
  - `discover_trajectory_samples(root: str|Path) -> list[TrajectorySample]`
  - `load_sample_arrays(sample, mmap=True) -> tuple[x, board, mask, y, timestamp]`,其中 x/board/mask/y 长度不一致时抛 `ValueError`
  - `sample_is_valid(sample) -> bool`
  - `SegmentRef(sample_index, start, length)`;`build_segment_refs(samples, segment_frames, strategy="pad_trim", min_chunk=1000) -> list[SegmentRef]`
  - `materialize_segment(sample, ref, segment_frames) -> (x[L,6], y[L,2], mask[L])`(不足 L 时头部补零)
  - `compute_normalization(samples, refs=None, segment_frames=None) -> (mean[6], std[6])`(仅训练数据)
  - `PaperTrajectoryDataset(samples, refs, mean, std, segment_frames)`,`__getitem__ -> (torch.FloatTensor x[L,6], torch.FloatTensor y[L,2], torch.FloatTensor mask[L])`

- [ ] **Step 1: Write the failing tests**

```python
class TrajectoryDatasetTests(unittest.TestCase):
    def test_materialize_pads_head_for_short_sample(self):   # N=100 < L=150
        # 前 50 帧 x,y,mask 全 0;后 100 帧等于源数据
    def test_materialize_symmetric_trim_for_long_sample(self):  # N=200 > L=150 -> start=25
    def test_build_chunk_refs_covers_all_and_drops_tiny_tail(self):  # N=20000,L=15000 -> [(0,15000),(15000,5000)]
    def test_invalid_length_sample_rejected(self):  # board 长度 != x 长度 -> sample_is_valid False
    def test_dataset_returns_normalized_segments(self):  # 形状 (L,6)/(L,2)/(L,) 且 mean≈0
    @unittest.skipIf(not DATA_ROOT.exists(), "dataset not present")
    def test_official_data_discovery_and_y_semantics(self):
        # 566 个样本;cumsum 语义与 board 的相关系数 > 0.99,并记录 delta anchor(0 或 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/opencode/ringcheck && PYTHONPATH=/tmp/opencode/ringcheck /data/huyang/miniconda/envs/vae/bin/python -m unittest writing_state.test_paper_trajectory -v`
Expected: FAIL / ImportError(模块不存在)

- [ ] **Step 3: Implement `paper_trajectory_dataset.py`**

关键算法:`build_segment_refs` 的 `pad_trim` 分支——`N <= L` 时 `SegmentRef(i, 0, N)`;`N > L` 时 `start=(N-L)//2, length=L`;`chunks` 分支按 `k*L` 连续切块、尾部余数 `< min_chunk` 丢弃。`materialize_segment` 头部补零到 L。`compute_normalization` 逐通道累计 mean/std(方差下限 `1e-8`)。

- [ ] **Step 4: Run tests to verify they pass**

Run: 同 Step 2,Expected: OK

- [ ] **Step 5: 任务收尾**

在 `test_paper_trajectory.py` 记录 `test_official_data_discovery_and_y_semantics` 输出的 anchor 结论(注释一行);不提交 git。

---

### Task 2: 模型与推理 `paper_trajectory.py`

**Files:**
- Create: `paper_trajectory.py`
- Test: `test_paper_trajectory.py`(追加)

**Interfaces:**
- Consumes: Task 1 的 `mean/std` 与数据形状约定。
- Produces:
  - `PaperTrajectoryConfig(sample_rate=200.0, window_frames=13, tcn_kernel=3, tcn_channels=(16,16,32,32,64,128), feature_dim=128, lstm_hidden=128, dropout=0.2, board_mm_scale=(240.0,169.5))`,属性 `window_offset=6`、`segment_frames=15000`,`validate()`
  - `stack_windows(x: Tensor[T,6]) -> Tensor[T-12,6,13]`
  - `TrajectoryTCN(config)`,`forward(M,6,13) -> (M,128)`
  - `WritingRingTrajectoryNet(config)`:`window_features(x[B,T,6]) -> (B,T-12,128)`;`forward(x, state=None) -> (out[B,T,2] 边界为 NaN, new_state)`
  - `chunk_slices(total: int, chunk_len: int, offset: int) -> list[tuple[int, int, int, int]]`(x_lo, x_hi, out_lo, out_hi;余数块正确)
  - `predict_sequence(model, x[T,6], chunk_len=None) -> np.ndarray[T,2]`(spec §4.1;流式分块前向,NaN 边界)
  - `load_trajectory_model(checkpoint, config=None, device="cpu") -> WritingRingTrajectoryNet`
  - `TrajectoryPredictor(model, config, mean, std, board_mm_scale)`,方法 `predict_deltas(x[T,6]) -> np.ndarray[T,2]`(归一化输入 + 整段前向)
  - `load_trajectory_predictor(checkpoint, device="cpu") -> TrajectoryPredictor`
  - `build_trajectory_summary(config) -> dict`(shape/参数量/窗口/offset)
  - `main()` CLI:`--json` 输出 summary;`--checkpoint` 加载校验

- [ ] **Step 1: Write the failing tests**

```python
    def test_stack_windows_alignment(self):        # x[t]=t -> 窗口中心 c 含 c-6..c+6
    def test_forward_shape_and_nan_edges(self):    # T=20 -> 有效索引 6..13,其余 NaN
    def test_lstm_state_is_carried_across_calls(self):  # 分两次 forward(state) 等价整段
    def test_chunk_slices_remainder(self):         # total=25, chunk=10, offset=6 -> 覆盖 [6,19)
    def test_chunked_inference_matches_full(self): # eval + dropout off, allclose 1e-5
    def test_checkpoint_roundtrip(self):           # save payload -> load -> 输出一致
```

- [ ] **Step 2: Run tests to verify they fail** — 同 Task 1 命令,Expected: FAIL

- [ ] **Step 3: Implement `paper_trajectory.py`**

关键算法:`window_features` 用 `F.unfold` 或 `(B,6,T+12)` 的 rolling 视图按 `window_frames` 切窗后批量过 TCN(实现自由,只要 `stack_windows` 语义一致);LSTM 逐窗消费特征,输出写回中心索引,两端 6 帧置 NaN。`chunk_slices` 保证每块的输出区间为 `[max(offset, s), min(T-offset, e))`,余数块不越界。

- [ ] **Step 4: Run tests to verify they pass** — Expected: OK

- [ ] **Step 5: 任务收尾** — 不提交 git。

---

### Task 3: 接触段积分与指标

**Files:**
- Modify: `paper_trajectory.py`(追加函数)
- Test: `test_paper_trajectory.py`(追加)

**Interfaces:**
- Produces:
  - `contact_segments(mask: np.ndarray, min_frames: int = 5) -> list[tuple[int, int]]`(极大连续 run)
  - `SegmentTrajectory(start, end, xy: np.ndarray[n,2])`
  - `integrate_contact_segments(deltas[T,2], mask, window_offset=6, min_frames=5) -> list[SegmentTrajectory]`(段内 `cumsum`,起点归零;NaN 预测帧按 0 处理并计数)
  - `trajectory_metrics(pred_xy, gt_xy, mm_scale=None) -> dict`(键:`normalized_mean`、`normalized_p50/p90`、`frac_gt_0.1`、`mm_mean`、`n_points`、`skipped_degenerate`)
  - `summarize_segment_metrics(rows: list[dict]) -> dict`(带权聚合 + 段数)

- [ ] **Step 1: Write the failing tests**

```python
    def test_integrate_constant_velocity_is_straight_line(self)
    def test_contact_segments_split_and_min_frames_filter(self)
    def test_normalized_metric_known_error(self)         # 已知偏移与 diag -> 手算值
    def test_degenerate_bbox_is_skipped_not_zero_division(self)
    def test_empty_mask_returns_empty_results(self)
```

- [ ] **Step 2: Run tests to verify they fail** — Expected: FAIL

- [ ] **Step 3: Implement(纯 numpy,公式:逐点距离/GT bbox 对角线;对角线 < 1e-6 记为 skipped)**
- [ ] **Step 4: Run tests to verify they pass** — Expected: OK
- [ ] **Step 5: 任务收尾** — 不提交 git。

---

### Task 4: 训练/评估入口 `train_paper_trajectory.py`

**Files:**
- Create: `train_paper_trajectory.py`
- Test: `test_paper_trajectory.py`(追加)

**Interfaces:**
- Consumes: Task 1 数据层、Task 2 模型/`chunk_slices`、Task 3 指标。
- Produces:
  - `random_split_samples(samples, seed, ratios=(0.7,0.1,0.2)) -> (train, val, test, split_info)`
  - `masked_mse_loss(pred[T,2], target[T,2], mask[T]) -> Tensor`(分母为 0 返回 0;NaN 预测行不计入)
  - `train(args) -> dict`,落盘 `paper_trajectory.pt` + `paper_trajectory_report.json`
  - `parse_args()` 参数:`--data-root`(默认本机绝对路径)、`--output-dir`(默认 `models_trajectory`)、`--split {random,user}`、`--epochs 100`、`--batch-size 16`、`--lr 1e-3`、`--chunk-len 3000`、`--long-sample-strategy {pad_trim,chunks}`、`--limit-samples`、`--seed 42`、`--device`、`--plot-dir`
- 训练循环 TBPTT:按 `chunk_slices` 逐块前向(块间 `detach` 隐状态)、每块 `loss.backward()` + `optimizer.step()`;验证/测试整段前向 + Task 3 指标;按验证集段级 `normalized_mean` 保存 best。
- report 字段:`checkpoint`、`data_root`、`split`、`sample_counts`、`segment_counts`、`normalization`、`config`、`board_mm_scale`、`history`、`val`、`test`。

- [ ] **Step 1: Write the failing tests**

```python
    def test_random_split_ratios_and_determinism(self)
    def test_user_split_has_no_user_overlap(self)   # 复用 paper_dataset.default_user_split
    def test_masked_mse_ignores_non_contact_and_nan(self)
    def test_masked_mse_empty_mask_returns_zero(self)
```

- [ ] **Step 2: Run tests to verify they fail** — Expected: FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run tests to verify they pass** — Expected: OK
- [ ] **Step 5: 冒烟验证(真实数据、小子集)**

Run: `cd /tmp/opencode/ringcheck && PYTHONPATH=/tmp/opencode/ringcheck /data/huyang/miniconda/envs/vae/bin/python -m writing_state.train_paper_trajectory --data-root /data/huyang/datasets/WritingRing/clean_data_delete_g/data --limit-samples 8 --epochs 1 --chunk-len 512 --output-dir /tmp/opencode/ringcheck/smoke_traj`
Expected: 打印 epoch 指标并生成 checkpoint/report.json;用 python 校验 report 含 `val.normalized_mean`。

---

### Task 5: board 单位标定(可降级)

**Files:**
- Create: `board_scale.py`
- Test: `test_paper_trajectory.py`(追加)

**Interfaces:**
- Produces:
  - `fit_scale_from_ranges(raw_xy: np.ndarray[n,2], clean_xy: np.ndarray[m,2], q=1.0) -> tuple[sx, sy]`(用 1%/99% 分位跨度比,跨度 < 1e-9 时返回 `(nan, nan)`)
  - `calibrate_sample(raw_block_dir, clean_sample, raw_chunk_glob) -> dict`(需 `compress_pickle`;失败时 `{"ok": False, "reason": ...}`)
  - `main()` CLI 输出 JSON;成功时给出 `mm_per_unit` 建议值
- 若标定不可行:保留 Spec 默认 `board_mm_scale=(240.0, 169.5)`,report 标注"按 Sensel Morph 尺寸假设"。

- [ ] **Step 1: Write the failing test** `test_fit_scale_recovers_known_ratio`(合成 raw×2 = clean → sx=sy≈2)
- [ ] **Step 2: Run tests to verify they fail** — Expected: FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run tests to verify they pass** — Expected: OK
- [ ] **Step 5: 尝试真实标定**

Run: `python -m writing_state.board_scale --raw-root /data/huyang/datasets/WritingRing/data --clean-sample /data/huyang/datasets/WritingRing/clean_data_delete_g/data/user_0/1/0`
Expected: 有 `compress_pickle` 时输出 JSON 标定结果;无则输出 `ok:false` 并给出安装提示,主链路不受影响。

---

### Task 6: 全量训练与评估(真实运行)

**Files:**
- Create(产物,不入库): `models_trajectory/`、`models_trajectory_user/`、`outputs/trajectory_plots/`

- [ ] **Step 1: 主结果(论文随机划分)**

Run: `python -m writing_state.train_paper_trajectory --data-root .../clean_data_delete_g/data --split random --epochs 100 --chunk-len 3000 --output-dir models_trajectory --plot-dir outputs/trajectory_plots`
Expected: report.json 的 `test.normalized_mean` 达到 0.073 量级;若明显更差,先检查数据对齐(anchor/单位)再调参。

- [ ] **Step 2: 严格对照(用户无重叠)**

Run:`--split user --output-dir models_trajectory_user`

- [ ] **Step 3: 曲线与可视化抽查**——打开 `outputs/trajectory_plots` 的代表性段,确认重建形状与 GT 一致;把两组 test 指标(归一化/mm/段数)记入 NOTES.md。

---

### Task 7: 文档与知识库同步

**Files:**
- Modify: `README.md`、`.trae/documents/STRUCTURE.md`、`.trae/documents/DESIGN.md`、`.trae/documents/NOTES.md`、`.trae/todo/TODO.md`

- [ ] **Step 1:** README 增加"轨迹重建"章节(用法、模型假设、指标结果)。
- [ ] **Step 2:** STRUCTURE 同步新文件与数据流;DESIGN 增加轨迹重建设计理念与里程碑勾选;NOTES 记录:论文方法核实、TCN 拓扑假设、TBPTT 差异、board 单位结论、随机 vs 用户划分对比。
- [ ] **Step 3:** TODO 任务 1 勾选 `(X)1.`;确认任务 2 保留。
- [ ] **Step 4: 全量回归**

Run: `cd /tmp/opencode/ringcheck && PYTHONPATH=/tmp/opencode/ringcheck /data/huyang/miniconda/envs/vae/bin/python -m unittest writing_state.test_paper_touch writing_state.test_writing_state writing_state.test_paper_trajectory`
Expected: 全部 OK;输出记录到 NOTES。
