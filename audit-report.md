# Research Codebase Audit Report — Ring / WritingRing 复现

- Project: Ring（WritingRing CHI 2025 复现：轨迹重建 + 字母识别 + 单词识别）
- Repository / commit: `/home/huyang/data/trae_projects/Ring`，HEAD = `7e9a90b`（初始化提交；轨迹/字符/单词全部新管线文件均为未跟踪状态，结果对应的是工作区代码而非该提交）
- Audit date: 2026-10-05
- Scope（审计范围）:
  - 轨迹重建管线：`paper_trajectory.py` / `paper_trajectory_dataset.py` / `train_paper_trajectory.py`
  - 字母识别管线：`character_dataset.py` / `character_model.py` / `train_character_classifier.py`
  - 单词识别管线：`word_dataset.py` / `word_recognition.py` / `word_ctc.py` / `train_word_ctc.py` / `evaluate_word_dtw.py`
  - 结果汇总与出图：`visualize_results.py` 及 `outputs/figures/metrics_summary.json`
  - 声明来源：`README.md`、`.trae/documents/{DESIGN,NOTES,STRUCTURE}.md`、`.trae/todo/TODO.md`、`ring_deck_materials/ring_report_source.md`、7 页汇报 PPT/讲稿
  - 支撑链路（抽样审计）：touch detector 路径、规则基线、Madgwick 探针、单元测试
- Coverage: Python 源文件 32/32 全部阅读或抽样（24 个深读、8 个抽样）；23 个 report JSON 全部程序化核对；92 个单元测试全部执行通过；3 组独立复算（噪声底、路径长度保真、严格跨用户下游评测）；未修改任何代码
- Report language: 中文

---

## 1. Executive summary

- Overall verdict: **CONDITIONAL → RESOLVED**（审计后当日已按第 8 节完成严格重跑，见 **§12**；M1 关闭，字母 86.30%、单词 64.17%、轨迹前端 0.159/5.54 mm 均为全测试集实测值）
- Findings: **0 CRITICAL, 1 MAJOR, 12 MINOR**
- Top problems（一行一条）:
  1. **M1 跨阶段评测泄漏**：下游"用户无重叠"端到端结果（字母 92.66%、单词 66.2%）所用的重建输入来自 x3 模型，而 x3 是随机划分训练，72–76% 的下游测试样本它见过；在这些样本上 x3 误差 0.078，在没见过的同类样本上 0.128。
  2. 多份文档与落盘产物不一致（ensemble 成员、最近邻基线数值、字母测试用户名单、样本数口径）。
  3. 噪声底 0.0377/1.28 为硬编码参照值，仓库内没有复算脚本（本报告已独立复算：0.0369/1.305，量级一致）。
- Results / claims affected:
  - 受 M1 影响：PPT 第 4/5/6 页、README 第 125–170 行、讲稿中所有"重建轨迹端到端、用户无重叠"的数字应视为**上界**；严格口径下未见过样本的子集结果为字母 88.2%（n=110）、单词词表对齐 60.0%（n=400）。
  - 不受影响:轨迹重建本身（x3 在自有 test split 上评估，无泄漏）、字母/单词的 GT 轨迹结果（94.77% / 66.69%）、touch detector 结果。

## 2. Coverage map

| Area | Files | Status | Reason if skipped |
|---|---|---|---|
| 轨迹重建 | `paper_trajectory.py`, `paper_trajectory_dataset.py`, `train_paper_trajectory.py` | audited（通读） | — |
| 字母识别 | `character_dataset.py`, `character_model.py`, `train_character_classifier.py` | audited（通读） | — |
| 单词识别 | `word_dataset.py`, `word_recognition.py`, `word_ctc.py`, `train_word_ctc.py`, `evaluate_word_dtw.py` | audited（通读） | — |
| 结果可视化 | `visualize_results.py`（1103 行） | audited（关键路径通读 + 全部指标独立复算） | 未整体重跑，避免覆盖 `outputs/figures` 现有产物 |
| Touch detector | `paper_touch.py`, `paper_dataset.py`, `train_paper_touch.py` | audited（关键路径） | — |
| 规则基线 | `detector.py`, `features.py`, `io.py`, `cli.py`, `realtime.py` | sampled | 不在任何报告结果链路上（README 已声明为兜底/回归用途） |
| 探针工具 | `madgwick.py`, `build_madgwick_dataset.py`, `board_scale.py` | sampled | 只用于 NOTES 中的输入预处理探针，不产生汇报数字 |
| 绘图辅助 | `board_plot.py`, `ring_plot.py`, `__init__.py` | sampled | 无指标逻辑 |
| 单元测试 | `test_*.py`（6 个文件，92 用例） | audited + executed（全部通过，2.6 s） | — |
| 结果产物 | 23 个 `*report.json` + `metrics_summary.json` | audited（逐字段核对） | — |
| 模型权重 / 数据 / 图像 | `*.pt`, `*.npy`, `*.png`, `*.pptx` 等 | skipped（二进制/大数据） | 模型权重以加载一致性校验代替 |

## 3. Findings

### CRITICAL

无。

### MAJOR

#### M1. 跨阶段评测泄漏：下游"用户无重叠"端到端结果的输入重建模型见过 72–76% 的测试样本

- Axis: B（数据泄漏）+ G（严谨性）+ E（声明与代码）
- Location:
  - `train_character_classifier.py:37-39`（`DEFAULT_RECONSTRUCTION = .../x3/paper_trajectory.pt`）
  - `train_word_ctc.py:44-46`（同一个 x3 默认值）
  - `visualize_results.py:85`（复算同样使用 x3）
  - `models_trajectory_topology/x3/paper_trajectory_report.json` → `"split": {"strategy": "random", "seed": 42}`
- Evidence（本次审计实测）:
  - 复现 x3 的随机 70/10/20 划分（seed 42）后统计：字母测试用户（16/18/19/20）的 113 个 block 中有 **86 个（76.1%）** 在 x3 训练集内；单词测试用户（17–20）的 97 个 block 中有 **70 个（72.2%）** 在 x3 训练集内。按下游样例计：字母 598/708（84.5%）、单词 906/1306（69.4%）位于"x3 见过"的 block 上。
  - 用同一套代码路径测量 x3 重建误差：见过样本 **0.0777**（1018 段）vs 未见过样本 **0.1275**（1532 段）；两组数据的协议噪声底几乎相同（0.0362 vs 0.0364），说明差异不是样本难度造成。
  - 用仓库现有模型在"x3 未见过"的子集上重算下游结果（其余流程完全不变）：
    - 字母（user_both，recon 输入）：整体 92.66%（n=708）→ 见过 93.48%（n=598）/ **未见过 88.18%（n=110）**
    - 单词（ctc_user_both，recon 输入）：整体 raw 60.64% / snapped 65.70% → 见过 64.02% / 68.21%，**未见过 53.00% / 60.00%（n=400）**
  - GT 输入对照（排除"子集本身难易不同"）：同样的两个子集改用 GT 轨迹输入时，字母 95.15% vs 92.73%（差 2.4 pt）、单词词表对齐 67.88% vs 64.00%（差 3.9 pt）；即未见过子集本身略难，但剩余 2.9–4.3 pt 的差距与"前端模型见过测试样本"一致。
  - 项目内部已记录该限制：`.trae/documents/NOTES.md:60` —"X3 轨迹模型用随机划分训练(含字符测试用户的数据)…严格跨用户端到端需用用户无重叠的轨迹模型重跑 recon（已记录,未执行）"；但 README:125-129/146-150/165-170、PPT 第 4/5/6 页与讲稿仍以"用户无重叠端到端"表述这些数字，未带该限定。
- Impact: 直接影响最显眼的两个结论——"字母 92.7% 高于论文 IME 88.7%"在严格口径下不再成立（约相当或略低）；"单词 66.2% 与论文开放词表 68.2/53.1 量级相当"在严格口径下要下修到约 60–62%。轨迹重建结论（0.1049 vs 论文 0.073）不受影响。
- Status: **CONFIRMED**
- Fix: 用"前端也从未见过测试用户"的协议重跑下游：以用户无重叠方式重训/微调一个与 x3 同级的轨迹模型（可用 `--split user`），用它对测试集生成重建轨迹，再重跑 `train_character_classifier.py` / `train_word_ctc.py` 的评估；在 README/PPT 中同时给出两种口径，或明确注明当前数字为"前端含测试用户随机样本"的上界。
- Retest: 重跑后核对 (a) 字母 recon top1、(b) 单词 recon raw/snapped、(c) 单词分 connected/unconnected，并与本报告的 88.2% / 60.0% 子集估计对照；差异应显著小于当前 92.66% / 65.7% 的乐观偏差。

### MINOR

#### m1. "最佳 ensemble" 的成员集合与页面复算不一致

- Axis: E
- Location: `visualize_results.py:67` `ENSEMBLE_MEMBERS = ("v1", "x3", "b2", "v4")`；`README.md:107` "x1+x3+b2+v4 … 0.100 / 4.01 mm（最佳总体）"；`.trae/documents/NOTES.md:78` 同。
- Evidence: 页面/讲稿引用的 0.1012 / 4.05 mm 来自 `visualize_results.py` 的 v1 版本 ensemble；README 引用的 0.100 / 4.01 来自 x1 版本；`TRAJECTORY_MODELS`（`visualize_results.py:61-66`）里没有 x1，因此 README 的"最佳总体"数字无法由仓库内可视化入口复现。
- Impact: 0.0009（norm）级别差异，不改变任何结论；但两个"ensemble"数字在文档中互换使用会误导复现者。
- Status: CONFIRMED
- Fix: 统一 ensemble 定义（建议把 x1 加入 `TRAJECTORY_MODELS` 并在页面标注成员），README 与 PPT 使用同一口径。
- Retest: 用统一入口复算一次，核对 0.1003/0.1012 中采用哪一个并全库同步。

#### m2. 最近邻基线数值与落盘产物不一致

- Axis: E + F
- Location: `README.md:146-150`、`NOTES.md:62` 声称 GT 闭集 63.6%、重建闭集 63.5%（"与 GT 几乎一致"）；`models_words/user/word_report.json` 实际为 GT closed 63.19%（n=1084）、recon closed **61.35%**，且其 config 为 `"metric": "dtw", "deskew": true`，与 README 给出的命令（`--metric euclidean`，deskew 默认关闭）不是同一次运行。
- Evidence: 上表报告 JSON 字段；两组数差 0.4 pt（GT）与 2.2 pt（recon），"几乎无损"的说法被唯一提交的产物否定。
- Impact: 不改变"CTC 66.2% 高于最近邻基线"的结论；影响复现者对基线的信任。
- Status: CONFIRMED
- Fix: 重跑并提交与 README 命令一致的 euclidean/no-deskew 报告，或把文档数字改为磁盘上 dtw+deskew 运行的值；同时说明分组数字（45%/61%）是含开放词表的全体口径，而头部 63.6% 是闭集口径。
- Retest: 用 `evaluate_word_dtw.py` 的同参数重跑一次核对。

#### m3. README 中字母识别的测试用户写错

- Axis: E
- Location: `README.md:125` "用户无重叠测试(user_17-20)"；实际 `models_character/user_both/character_report.json` 的 `test_users = [user_16, user_18, user_19, user_20]`。
- Evidence: 本次实测 `discover_character_samples` 共 3845 条、覆盖 20 个用户，**user_17 没有任何合格字母样本**，因此按 `default_user_split` 取排序后最后 20% 得到 16/18/19/20（`paper_dataset.py:65-74`）。PPT/讲稿写的是正确的 16/18/19/20。
- Impact: 纯文档错误；评测本身仍是用户无重叠（user_16 未参与训练）。
- Status: CONFIRMED
- Fix: README 改为 user_16/18/19/20，并说明 user_17 无字母样本。
- Retest: 无需重跑。

#### m4. 协议噪声底 0.0377 / 1.28 为硬编码参照，无复算脚本

- Axis: F + G
- Location: `visualize_results.py:92-93`（`PAPER = {..., "noise_floor_normalized": 0.0377, "noise_floor_mm": 1.28, ...}`）。
- Evidence: 本次按"完美积分官方 y"独立复算全部 566 样本、54383 个接触段（跳过 10 个退化段），得 **0.0369 / 1.305 mm**（点加权），与 0.0377/1.28 在尾数层面一致，说明数字可信；但仓库中没有任何脚本/入口复现它，属于"引用的关键参照不可自动复算"。
- Impact: 低；读者无法在仓库内验证论文对比中最关键的参照线。
- Status: CONFIRMED（数字本身）
- Fix: 把噪声底计算固化为脚本/子命令（输入 clean 数据，输出 norm/mm），并写入 report。
- Retest: 脚本输出与 0.0377/1.28 对照（或更新常量）。

#### m5. `y[t] = board[t+1] − board[t]` 的表述强于实测

- Axis: E
- Location: 所有 `paper_trajectory_report.json` 的 `y_anchor_note`（x3 报告第 74 行）："verified on official data: y[t] = board[t+1] − board[t]"；`test_paper_trajectory.py:202-211` 的注释与断言（只验证 corr>0.9）。
- Evidence: 全量 566 样本实测——接触段"内部帧"仅 5.1% 精确相等（平均差 6.4e-4、p99 2.6e-3）；每个接触段的最后一帧与 board 差分差异巨大（平均 0.754，因为抬笔后 board=0）。二者是"高相关近似"（帧级 corr 0.97、积分 0.9995），不是恒等式；该差异正是 0.037 噪声底的来源。
- Impact: 训练目标与评测真值存在系统差异；NOTES 已用"噪声底"解释其后果，但报告字段的措辞会让读者误以为目标无噪声。
- Status: CONFIRMED
- Fix: 改为 "y approximates board[t+1]−board[t] (frame corr 0.97, integrated 0.9995; differs at segment boundaries)"。
- Retest: 无需重跑。

#### m6. "566 个有效样本"应为"566 个发现样本、564 个可用"

- Axis: E
- Location: PPT 封面/讲稿"二十一个用户、五百六十六个有效样本"；各轨迹报告 `sample_audit` = `{"discovered": 566, "invalid_count": 2, "invalid_ids": ["user_0/0/0","user_7/0/0"]}`。
- Evidence: `train_paper_trajectory.py:369-376` 先过滤无效样本再划分，395+56+113 = 564。
- Impact: 纯口径表述；与 NOTES 的记录一致，仅汇报文案需修正。
- Status: CONFIRMED
- Fix: 统一写"566 个样本、564 个可用（2 个长度不一致被跳过）"。
- Retest: 无需重跑。

#### m7. 单词 raw/napped 数字在 PPT 中被标成"重建轨迹"，实际是 GT+重建合并口径

- Axis: E
- Location: 讲稿第 5 页"重建轨迹加词表对齐整体 66.2%…raw 61.2%"；`models_words/ctc_user_both/word_ctc_report.json` 顶层 test n=2612（gt 1306 + recon 1306），其 `gt`/`recon` 子块分别为 snapped 66.69%/65.70%、raw 61.79%/60.64%。
- Evidence: 本次用仓库模型在 recon 子集上重算得 raw 60.64% / snapped 65.70%，与报告 recon 子块一致。
- Impact: 数值差约 1 pt，不改变结论；但"重建轨迹"的表述让读者以为 66.2 是端到端重建口径。
- Status: CONFIRMED
- Fix: 写明"GT+重建合并口径 66.2%（其中重构子集 65.7%）"或直接引用 recon 子块。
- Retest: 无需重跑。

#### m8. 示例图是"定向挑选"的，6/8 正确率不是随机抽样

- Axis: E + G
- Location: `visualize_results.py:957-967`：先取 4 个 raw 命中、再取 4 个 raw 未命中（每组各 2），补足 8；经词表对齐后渲染为 6/8 正确。PPT/讲稿"八例里有六例对齐后正确"。
- Evidence: 代码选择逻辑；该图同时用于展示"对齐挽回错误"（rom→from、rilch→rich），因此必然高配错误样例。
- Impact: 示意目的合理，但 75% 高于 recon 子集的 65.7%，不应被当作抽样证据。
- Status: CONFIRMED
- Fix: 图注注明"为展示错误与对齐纠错，样例为定向挑选"；或补充一张随机样例图。
- Retest: 无需重跑。

#### m9. 模型选择使用了 test 指标，且全部结果只报单一种子

- Axis: G
- Location: `.trae/documents/NOTES.md:74-79`（B1→B2、X1→X3、ensemble 变体按 test 值比较后择优），`:45-49`、`:55-57`（字母/单词变体同理）。
- Evidence: x1 test 0.1059 vs x3 0.1049 选 x3；B2 0.1100 vs B1 0.1353；ensemble 多个变体取最好；所有报告 `seed=42` 且无多种子重复/误差棒。
- Impact: 被选中的数字带有"n 选 1"的乐观偏差，但候选间差距小（norm ≤0.001、字母 ≤0.9 pt、单词 ≤3.7 pt），不改变"距论文约 1.4x"的结论。
- Status: CONFIRMED
- Fix: 主表以 validation 选择为准，或同时给出全部候选与多种子均值±标准差。
- Retest: 固定 seed 集合（如 42/43/44）重跑最优候选。

#### m10. `predict_run_points` 的 `chunk_len` 参数是死参数；训练模式文档表述含混

- Axis: A + E
- Location: `train_character_classifier.py:100-103`（接收 `chunk_len` 但调用 `predictor.predict_deltas()`）；`paper_trajectory.py:307-311`（`predict_deltas` 不接受/不传 `chunk_len`）。
- Evidence: 参数从未生效；对 x3 这类双向模型本来也必须全序列前向（`paper_trajectory.py:247-251` 会显式拒绝分块），所以结果无影响。另外 x3 报告 `train.update_every = "batch"` 且双向强制 `chunk_len=15000`，实际是整段 BPTT，而 v4 是 `"chunk"`（TBPTT 每块一步）；文档统一描述为"TBPTT 分块"，未区分。
- Impact: 代码卫生与文档精度；无结果影响。
- Status: CONFIRMED
- Fix: 删除该死参数或真正透传；在 README/报告里按模型注明 BPTT/TBPTT 与 chunk 长度。
- Retest: 无需重跑。

#### m11. 字符/单词标签时间被 `int()` 截断，可能造成 ≤1 帧偏移

- Axis: A
- Location: `character_dataset.py:37-42`（`int(float(tokens[index]))`），`word_dataset.py` 复用同一函数。
- Evidence: 标签时间先截断为整数再 `searchsorted` 到 200 Hz 时间轴；亚秒小数被丢弃，最坏偏移 1 帧（5 ms）。被截断时刻若正好落在接触段起点前 1 帧，可能改变入段判定。
- Impact: 极小；单个字符/单词的 1 帧偏移不足以改变结果量级。
- Status: SUSPECTED（未构造受影响样本复现）
- Fix: 保留 float 时间再 searchsorted。
- Retest: 修改后重跑字母/单词各一次，核对 n 与 top1 是否变化。

#### m12. 多笔画字符/单词的"回原点"拼接约定未成文

- Axis: C + E
- Location: `character_dataset.py:150-160`、`word_dataset.py:126-139`：每个接触 run 各自减去自身起点再首尾拼接，抬笔位移被替换成"回到原点"的直线段，然后整体重采样。
- Evidence: 对 T、X、K 等多笔画字符会人为加入回原点的路径段；这可能与 T=85.0%、X=91.7% 的相对弱项相关（不能证明因果）。GT 与 recon 使用同一约定，故对端到端比较公平，但会影响绝对精度与"笔迹"语义。
- Impact: 建模选择，非错误；未在 NOTES 中说明。
- Status: CONFIRMED（约定存在）；因果影响 SUSPECTED
- Fix: 在 NOTES/README 记录该约定；可实验"笔画间 teleport 标记"编码作为对照。
- Retest: 若改变约定，需重跑字母与单词全部指标。

## 4. Claims verification（声明核对）

| # | Claim (source) | Code/artifact evidence | Status | Notes |
|---|---|---|---|---|
| 1 | Touch detector test acc 0.8829 / macro F1 0.8832 / event F1 0.9072，测试用户 17–20（README:25-34） | `models/paper_touch_report.json` test 完全一致；split test_users=[17,18,19,20] | CONFIRMED | — |
| 2 | 轨迹模型：13 帧窗、kernel=3、TCN→1×128、LSTM128、预测中点、6 帧前瞻 32.5ms（README:85） | `paper_trajectory.py:18-36,63-88,220-232`；`test_paper_trajectory.py` 形状/对齐用例通过 | CONFIRMED | 默认 valid TCN；报告主模型用 dilated 变体 |
| 3 | 主模型随机划分 0.124 / 5.08 mm，逐点 >0.1 占 46.3%（README:103-105） | `models_trajectory/paper_trajectory_report.json`（= v4 报告，md5 相同）norm 0.12416、mm 5.0793、frac_gt_0.1 0.4629 | CONFIRMED | — |
| 4 | X3 单模最佳 0.1049 / 4.15 mm（README:106） | x3 报告 norm 0.10488、mm 4.1518 | CONFIRMED | — |
| 5 | "最佳总体 ensemble 0.100 / 4.01 mm（x1+x3+b2+v4）"（README:107） | 无落盘产物；可视化入口用 v1 组合得 0.1012/4.05（`visualize_results.py:67`） | PARTIAL / 不可由现入口复现 | 见 m1 |
| 6 | 用户无重叠轨迹 0.214 / 8.53 mm（旧架构，README:108） | `models_trajectory_user/paper_trajectory_report.json` 一致 | CONFIRMED | 旧架构，非同级对照 |
| 7 | 路径长度保真：中位比 0.98、r=0.991（README/NOTES:99-101） | 本次全量复算（113 样本、11750 段）：0.978 / 0.9911 | CONFIRMED | 段数 11750 vs 报告 11748，尾数差异 |
| 8 | 协议噪声底 0.0377 / 1.28 mm（README:111、NOTES:25） | 本次复算 0.0369 / 1.305 mm；仓库无复算脚本（硬编码于 `visualize_results.py:92-93`） | CONFIRMED（近似） | 见 m4 |
| 9 | 字母 GT top1 94.8 / top3 99.6；重建 top1 92.7 / top3 98.9（README:128-129） | user_both 报告 test.gt 0.9477/0.9958、test.recon 0.9266/0.9887；本次复算 recon 92.66% | CONFIRMED | 分类器用户无重叠成立 |
| 10 | 字母"用户无重叠测试 user_17-20"（README:125） | 实际 test_users=16/18/19/20；user_17 无字母样本 | CONTRADICTED | 见 m3 |
| 11 | 字母"重建轨迹端到端、用户无重叠"（README/PPT/讲稿） | 前端 x3 随机划分含测试用户；严格子集 88.18%(n=110) | PARTIAL / 乐观偏差 | 见 M1 |
| 12 | 单词 GT+重建混训 raw 61.2 / 对齐 66.2；connected 57.8 / unconnected 75.2（README:168） | ctc_user_both 报告顶层 n=2612 一致 | CONFIRMED | 为 GT+recon 合并口径；recon 子集 60.64/65.70 |
| 13 | 单词 GT-only 54.9/62.3；随机划分 68.9/81.8（README:169-170） | ctc_user_gt 与 ctc_random_both 报告一致 | CONFIRMED | 均单种子 |
| 14 | 最近邻基线 63.6%（GT）/63.5%（recon），"几乎无损"（README:149-150） | 磁盘报告 GT closed 63.19%、recon closed 61.35%（dtw+deskew，配置与 README 命令不符） | CONTRADICTED（recon 项） | 见 m2 |
| 15 | "数据词表 561 词"（README:146） | 原始标签全量唯一词 = 561（去除 wrong 后 560）；当前评估用训练词表 416、过滤后数据词表 507 | PARTIAL | 需注明口径 |
| 16 | `y[t] = board[t+1] − board[t]`（各报告 `y_anchor_note`） | 高相关近似：内部帧仅 5.1% 精确相等；段尾差异大 | PARTIAL | 见 m5 |
| 17 | 论文参照值 0.073/1.64/80%、字母 88.7%、单词 68.2/53.1、3000 词表 84.4/74.2 | 仓库内无论文 PDF；数值转录于 NOTES:14/62-63，无法独立核验 | UNVERIFIABLE | 外部来源 |
| 18 | "266 个有效样本…": "566 个有效样本、21 用户"（PPT/讲稿） | 566 发现、564 可用（2 个无效） | PARTIAL | 见 m6 |
| 19 | 单词端到端"用户无重叠"（README:165-172） | 与字母同理，前端 x3 见过 69.4% 测试样例；严格子集 snapped 60.0% | PARTIAL | 见 M1 |
| 20 | 三级模型全部自训、无外部服务/预训练权重 | 代码中无下载/加载预训练逻辑；checkpoint 均为本地训练产物 | CONFIRMED | — |

## 5. Leakage assessment

| Check | Result | Evidence | Notes |
|---|---|---|---|
| 特征在预测时可用 | OK | 轨迹输入仅 IMU；识别输入仅轨迹 | — |
| 目标泄漏 | OK | 无标签参与输入构造 | — |
| 时序泄漏 | N/A | 数据为单段离线样本，无滚动统计 | — |
| 划分尊重分组（用户） | OK（user 口径） | `default_user_split`；字母/单词报告 test_users 与 train 无交集 | 字符测试用户因 user_17 无样本而变为 16/18/19/20 |
| 随机划分的样本级隔离 | OK | `random_split_samples` 按样本整体划分，无同一样本跨集 | 轨迹主口径为随机划分（论文协议），同用户样本会跨集，已披露 |
| 预处理统计只在训练集拟合 | OK | 轨迹 `compute_normalization(train_samples)`（`train_paper_trajectory.py:358`）；字符/单词为逐样本归一化 | — |
| 词表只在训练集构建 | OK | `train_word_ctc.py:218`：`sorted({example["word"] for example in train_examples})` | — |
| 数据增强只作用于训练 | OK | `augment=True` 仅用于 train DataLoader | — |
| 检查点选择不看 test | OK | 三个训练脚本均按 val 指标选 best（轨迹 val norm / 字符 val top1 / CTC val snapped） | — |
| 超参与模型选择不看 test | **FINDING** | NOTES:74-79 等按 test 值择优 | 见 m9 |
| 跨阶段：前端重建模型未见下游测试样本 | **FINDING** | x3 随机划分含 72–76% 下游测试 block | 见 M1；NOTES:60 已记录未执行 |
| 重复/近似重复样本跨集 | 未发现（抽查） | 随机划分按样本；user 划分按用户 | 未做全量近重复检测，列为残余不确定性 |
| 测试集直到最终才使用 | PARTIAL | 每个候选模型只在训练结束后评一次 test，但多候选共享同一 test 并据此择优 | 见 m9 |

## 6. Mock / fake / placeholder inventory

| Location | Kind | Test-only? | Result-affecting? | Action |
|---|---|---|---|---|
| — | 未发现 mock/fake/stub/hardcoded metric | — | — | `grep Mock/patch/NotImplementedError/TODO/FIXME` 在全部源码中零命中 |
| `visualize_results.py:88-99` `PAPER` 字典 | 外部参照常量（含噪声底） | 否 | 是（作为参照线绘制） | 非伪造：本报告已独立复算噪声底，量级一致；建议脚本化（m4） |
| `models_smoke/`, `models_probe/` | 冒烟/探测产物目录 | — | 否 | 报告的 checkpoint 均指向 `models/`、`models_trajectory_topology/`、`models_character/user_both`、`models_words/ctc_*`，未混用 |
| 单元测试 | 无对被测对象的 mock | 是 | 否 | 92 用例全部通过；但均为单元级，不覆盖端到端指标复现 |

## 7. Rigor and reproducibility

| Check | Result | Evidence | Notes |
|---|---|---|---|
| 随机种子（python/numpy/torch/cuda） | OK | 四个训练入口均调用 `set_seed`（`train_paper_trajectory.py:31-36` 等） | — |
| DataLoader 确定性 | OK | 全部 `num_workers=0` | 无 worker 竞争 |
| cudnn 确定性 | UNVERIFIED | 未设置 `torch.backends.cudnn.deterministic` | GPU 算子非确定路径未排除 |
| 报告记录 config/split/归一化统计 | OK | 各 report.json 含 config、split、mean/std、history | — |
| 报告记录 git commit / 数据版本 | **FINDING** | 仅记录路径，无 commit/hash | 结果无法绑定到代码版本；工作区大量未跟踪文件 |
| 依赖固定 | **FINDING** | 仓库无 `requirements.txt`（NOTES/STRUCTURE 已列为待办） | 环境为 conda `vae`（torch 2.2.0+cu121） |
| 指标方向与聚合 | OK | norm/mm 为误差；字母 top1/top3；单词 raw/snapped 整词匹配；`summarize_segment_metrics` 按点数加权 | 复算与报告一致 |
| 多种子方差 | **FINDING** | 全部 seed=42，单次结果 | 结论差距远大于候选间差异，但"最佳模型/ensemble"缺误差棒 |
| 检查点选择 | OK | val-only | — |
| 基线可比性 | PARTIAL | 最近邻与 CTC 同数据同划分；但与论文 IME 协议不可比（文档已注明） | 见 m2 |
| 手动步骤/魔法数 | PARTIAL | 噪声底、PAPER 数值为硬编码；`DEFAULT_*` 绝对路径 | 已文档化，建议脚本化 |
| 测试需要 `writing_state` 软链 | FINDING（环境） | 仓库无该软链；本次审计在 `/tmp` 建等效软链后 92 用例全通过 | README/NOTES 有说明，但开箱不可直接跑测试 |

## 8. Required repairs and retest conditions

1. **（最高优先）消除 M1 的前端暴露**：以用户无重叠方式重训/选一个与 x3 同级的轨迹前端，重生成测试集重建轨迹，重跑字母与单词评估；在 README/PPT 中并列发布"严格口径"与"当前口径"，并注明当前数字的偏差来源。重跑验证点：字母 recon top1、单词 recon raw/snapped、connected/unconnected 分组。
2. **统一文档口径**：ensemble 成员（m1）、最近邻基线与命令（m2）、字母测试用户（m3）、样本数（m6）、word 数字的 GT+recon 合并口径（m7）、y 语义措辞（m5）、词表口径（561/507/416）。
3. **把噪声底计算脚本化**并纳入 report（m4）。
4. **补充运行元数据**：git commit、数据目录 hash、依赖版本（requirements.txt）（第 7 节）。
5. **声明示例图为定向挑选**，并补一张随机样例图（m8）。
6. **按 val 口径重述模型选择**，或给出多种子均值±标准差（m9）。
7. **代码卫生**：删除 `predict_run_points` 死参数、修正标签时间截断、在 NOTES 记录多笔画拼接约定（m10–m12）。

在完成 1（严格重跑）之前，**字母/单词的端到端数字应仅作为上界使用**；轨迹重建、GT 口径识别、touch detector 的结果可以正常引用。

## 9. Residual uncertainty and coverage gaps

- 论文参照值（0.073/1.64/80%、IME 88.7%、68.2/53.1、84.4/74.2）无法在仓库内核验，未发现论文原文/补充材料副本；只能确认它们被一致地转录与引用。
- 严格子集估计（字母 88.18%，n=110；单词 snapped 60.0%，n=400）不是随机子集：GT 对照显示子集间存在 2.4 pt（字母）/3.9 pt（单词）的固有难度差；因此"完全未见前端"的真实值可能落在"GT 对照下界"与"当前子集点估计"之间，建议以第 8 节 1 的正式重跑为准。
- 未做全库近重复检测（跨划分重复轨迹）与训练过程确定性复核（cudnn、GPU 非确定性）。
- `visualize_results.py` 未整体重跑（避免覆盖交付产物）；其全部指标计算已用仓库函数独立复算，数值一致。
- 未验证模型权重与报告是否来自同一训练运行（无 run id/commit；只能核对 config/split/指标的内部一致性）。

## 10. Appendix

- Commands run（均为只读）:
  - `python scripts/scan_red_flags.py --root . --out /tmp/ring_audit/red_flags.json`
  - `python -m unittest writing_state.test_paper_trajectory ... test_writing_state`（92 用例，OK，2.6 s；经 /tmp 软链）
  - 噪声底复算：566 样本、54383 段 → 0.0369 / 1.305 mm
  - 路径长度复算：113 样本、11750 段 → 中位比 0.9781、r=0.9911
  - 严格跨用户下游复算：字符 92.66%→(seen 93.48% / unseen 88.18%)；单词 recon 65.70%→(68.21% / 60.00%)；GT 对照 95.15/92.73% 与 67.88/64.00%
  - x3 随机划分重叠统计：86/113、70/97 block；598/708、906/1306 样例
- Files sampled（抽样未逐行）: `detector.py`, `features.py`, `io.py`, `cli.py`, `realtime.py`, `madgwick.py`, `build_madgwick_dataset.py`, `board_plot.py`, `ring_plot.py`
- Environment notes: 审计用 Python 为 `vae` env（torch 2.2.0+cu121，CUDA 可用）与 `pptmaster` env（无 torch，用于 JSON/文本核对）；数据位于 `/data/huyang/datasets/WritingRing/`。未修改任何源代码；唯一新增文件为本报告。

---

## 11. 修订记录（2026-10-05，审计之后）

本报告记录的是审计时点的只读快照。审计后按用户要求"把 PPT 中的数据改正确"完成了**文档与汇报层面**的修正（未重训任何模型）：

- **M1 口径**：README、STRUCTURE、TODO、7 页结果版 PPTX/讲稿、14 页报告版 PPTX/讲稿与《讲稿与指标说明》统一改为"上界口径 + 严格未见子集估计"双数字（字母 92.7% → 严格 88.2%，n=110；单词 65.7% → 严格约 60.0%，n=400）；示例图标明"定向挑选"。
- **m1-m8**：ensemble 成员统一为 v1+x3+b2+v4（0.1012 / 4.05 mm；归档 x1 变体 0.1003 / 4.01 单独注明）；最近邻基线更正为落盘 dtw+deskew 的 GT 63.2% / recon 61.4%；字母测试用户更正为 16/18/19/20；样本数改为"566 个样本、564 个可用"；明确 66.2% 为 GT+重建合并口径、重建子集 65.7%（raw 60.6%）；`y` 与 board 差分表述改为"高相关近似（帧级 0.97、积分 0.9995）"。
- **产物同步**：`outputs/writingring_results_20261005/` 与 `outputs/writingring_report_20261005/`（pptx / pdf / 讲稿 / svg_source / 预览图）均已按修正后重导出；`fig_word_recognition.png`、`fig_letter_accuracies.png` 已重生成并同步到 `outputs/figures/`、`ring_deck_materials/figures/` 与两个 deck 项目。
- **仍未执行（保持 open）**：M1 的正式重跑（用户无重叠轨迹前端 → 下游端到端）；m4 噪声底复算脚本化；m9 去掉 test 选型与多种子复跑；m10-m12 的代码注释/死参数/拼接约定整理。

---

## 12. M1 正式重跑（2026-10-05，审计后当日完成）

审计给出的处置建议是"要么重跑前端、要么把数字标记为上界"。本节记录**重跑完成**后的结果，M1 由"已标记上界"升级为"已消除"。

- **重跑协议**：`train_paper_trajectory.py --split user --holdout-users 16,17,18,19,20`（前端 train user_0-13 / val 14-15 / test 16-20），架构与随机划分前端 x3 对齐（dilated TCN 32/64/128 + 2 层双向 LSTM，bidirectional 强制全序列 BPTT）。下游沿用同一 user split 重训：字母分类器 `models_character/user_both_ud`（test user_16/18/19/20）、单词 CTC `models_words/ctc_user_both_ud`（test user_17-20）。
- **跨阶段覆盖核对**：下游 test 用户 ⊂ 前端 holdout {16-20}；下游 val 用户（14/15/16）也不在前端训练集内。重跑前后端对下游测试样本的覆盖率为 **0%**（原 x3 为字母 84.5%、单词 69.4%）。
- **结果（全测试集，无子集估计）**：
  - 轨迹前端自身：0.1589 / 5.54 mm（test 129 样本 / 11070 段；逐点 >0.1 占 51.4%）。同架构随机划分 0.1048 / 4.13 mm，论文 0.073 / 1.64 mm。
  - 字母（n=708）：重建 top1 86.30% / top3 96.05%，GT 93.93% / 98.87%。上界口径 92.66% → **前端口径真实代价 −6.4 pt**；原"未见子集估计"88.2%（n=110）偏乐观，已被全测试集实测替代。
  - 单词（n=1306）：重建 raw 58.58% / 词表对齐 64.17%（连笔 56.49 / 断笔 72.45），GT 63.94% / 66.92%。上界口径 65.70% → **仅 −1.5 pt**，即词级任务对前端误差不敏感；字母等短笔画任务敏感。原估计 60.0%（n=400）同样被实测替代。
- **产物**：`models_trajectory_topology/b2u/`、`models_character/user_both_ud/`、`models_words/ctc_user_both_ud/`、`outputs/figures_strict/`（含 `letter_accuracy.csv`、`metrics_summary.json`）；`visualize_results.py` 新增 `--protocol strict`，并修正三处图注（原图注写死"random-split front-end → upper bound"，现已按协议动态生成）。
- **状态**：M1 关闭。仍开放的审计项：m4（噪声底复算脚本化）、m9（test 选型 / 多种子）、m10-m12（注释、死参数、拼接约定）。严格口径下前端仅训 300 epochs（x3 为 450），未做 lr/epochs 调优，属已知可继续提升空间。
