# STRUCTURE — 项目结构

> 此文件由 trae-init 初始化时依据项目扫描自动生成,并在此后的开发中持续维护。
> 若代码结构发生影响模块划分的改动,须同步更新本文件。

## 目录结构

```text
Ring/                          # 仓库根;运行时以模块名 writing_state 被导入(父目录需有软链 writing_state -> Ring)
├── __init__.py                # 包入口,导出规则基线的 DetectorConfig / DetectorEvent / DetectorState / WritingStateDetector
├── paper_touch.py             # 学习模型主路径:WritingRing touch detector 复现(窗口化、1-D ResNet、四分类、press/lift 事件解码、加载与导出)
├── paper_dataset.py           # 数据集层:clean_data 样本发现、滑窗标签、用户无重叠 train/val/test 划分、归一化、PyTorch Dataset
├── train_paper_touch.py       # 训练入口:训练/验证/测试与报告落盘
├── evaluate_paper_touch.py    # touch detector 独立评估入口:加载 checkpoint 的 split/mean/std,对 train/val/test 复评并落盘 report
├── paper_trajectory.py        # 轨迹主路径:TCN+LSTM 速度预测、流式分块推理、接触段积分、论文指标
├── paper_trajectory_dataset.py # 轨迹数据层:75s 长片段构造(补零/对称裁剪)、归一化、Dataset
├── train_paper_trajectory.py  # 轨迹训练入口:随机/用户划分、TBPTT、checkpoint 与 report
├── board_scale.py             # clean board 与原始 Sensel 触点的单位标定(compress_pickle 缺失时降级)
├── character_dataset.py       # 字母级切分:标签时间戳区间 + 接触段分组、弧长重采样/归一化、GT 样本构建
├── character_model.py         # 26 类字母 1D CNN 分类器 + checkpoint 加载/摘要
├── train_character_classifier.py # 字母分类训练入口:GT/重建混训、用户/随机划分、report(含混淆矩阵)
├── word_dataset.py            # 整词切分:标签区间分组、重采样/deskew/归一化、GT 与重建词样本
├── word_recognition.py        # 词识别基线:DTW、最近实例(欧氏/DTW 重排)与按组/词表指标
├── evaluate_word_dtw.py       # 词识别评测 CLI(report.json 落盘)
├── word_ctc.py                # CTC 词识别模型:CNN+BiLSTM、贪心解码、编辑距离词表对齐
├── train_word_ctc.py          # CTC 训练入口:GT/重建混训、词表对齐评估、report
├── madgwick.py                # Madgwick AHRS 去重力探针(任务 3 输入阶段,未接入主链路)
├── build_madgwick_dataset.py  # 用 raw+Madgwick 重建 clean 数据集的探针工具(对齐/写盘,未接入主链路)
├── build_clean_dataset.py     # raw→clean 重建:逆向作者的清洗管线(0.6/1.0 s 裁剪、4977.75 us 均匀网格、去重力、board 重采样、mask/y),自包含加载器 + --validate-against 验证
├── detector.py                # 规则基线:硬阈值 + 状态机(接触/抬笔/无效空中段)
├── features.py                # 规则基线的低维特征提取(加速度/角速度/加加速度/冲击)
├── io.py                      # CSV 读取与列名适配(SmartRing / OpenZen),线性加速度缺失时的离线兜底
├── cli.py                     # 规则基线离线回放命令行,导出 events / segments CSV
├── realtime.py                # 实时接入:RealtimeWritingStateHandler,桥接 raw frame 与 ImuBus
├── board_plot.py              # 触控板数据可视化(轨迹 + 压力)
├── ring_plot.py               # 戒指 IMU 数据可视化
├── visualize_results.py       # 结果可视化:轨迹重建对比/误差统计、字母混淆矩阵与识别样例、词识别对比与样例(输出 outputs/figures + metrics_summary.json)
├── test_paper_touch.py        # 学习模型路径单元测试(形状/重采样/结构/事件/checkpoint 加载)
├── test_writing_state.py      # 规则基线单元测试(合成序列回归)
├── test_paper_trajectory.py   # 轨迹路径单元测试(数据/模型/积分指标/划分/masked loss)
├── test_character_classifier.py # 字母识别单元测试(切分/特征/模型/划分/指标/重建组装)
├── test_word_recognition.py   # 词识别单元测试(切分/特征/DTW/最近实例指标)
├── test_word_ctc.py           # CTC 单元测试(编辑距离/贪心解码/词表对齐/模型/collate)
├── models/                    # touch detector 输出:paper_touch_resnet.pt(8 ep,0.8829)+ report.json;exp_30/exp_50/ 为更充分训练实验(0.9038/0.9092),exp_50 附独立评估 test_report.json
├── models_trajectory/         # 轨迹模型主训练输出(论文随机划分,V4 dilated+2层LSTM):paper_trajectory.pt + report.json
├── models_trajectory_topology/ # 拓扑/双向实验归档 v1、v3、v4(流式主模型)、u1(3层)、b2(双向)、x1/x3(B2 精调);b2u = 用户无重叠严格口径前端(+ eval15k/ 复评报告)
├── models_trajectory_user/    # 轨迹模型用户无重叠划分输出(严格对照,旧架构)
├── models_character/          # 字母分类输出:user_both/user_gt/random_both + user_both_ud(严格口径,前端 b2u),各含 character_cnn.pt + report.json
├── models_words/              # 词识别输出:user/(最近实例基线 report);ctc_{user_both,user_gt,random_both}/(CTC checkpoint+report);ctc_user_both_ud/(严格口径,前端 b2u)
├── models_probe/              # 探测性训练输出(同结构)
├── models_smoke/              # 冒烟训练输出(同结构)
├── README.md                  # 使用说明(模型、规则基线、CLI、实时、参数)
├── README_data.md             # WritingRing 数据集格式说明
└── .trae/                     # 项目知识库(结构/设计/思考/规则/任务)
```

## 模块职责速查

| 路径 | 职责 | 备注 |
| --- | --- | --- |
| `paper_touch.py` | 学习模型主路径:TouchWindowizer 窗口化、`WritingRingTouchResNet`(残差通道 8/16/32)、四分类 contact/air/lift/press、`PaperTouchDecoder` 只把 press/lift 解码为事件 | 输入 6 通道固定顺序 `lin_acc_x/y/z, gyro_x/y/z` |
| `paper_dataset.py` | 从 `{data_root}/user_*/{action}/*_x.npy + *_mask.npy` 发现样本;`default_user_split` 按用户划分;`window_labels` 按窗口首尾状态打标 | 数据根目录不在仓库内,训练时用 `--data-root` 指定 |
| `train_paper_touch.py` | 训练入口,输出 checkpoint 与 report.json(含 split、窗口计数、归一化、history、val/test 指标) | 默认 `--output-dir writing_state/models` |
| `evaluate_paper_touch.py` | touch detector 独立评估入口:从 checkpoint 读取 split/mean/std 复现同口径评估,可 `--report-out` 落盘 | `python -m writing_state.evaluate_paper_touch --checkpoint writing_state/models/exp_50/paper_touch_resnet.pt --split test --report-out writing_state/models/exp_50/test_report.json` |
| `paper_trajectory.py` | 轨迹模型主路径:`PaperTrajectoryConfig`(窗 13/TCN k=3/hidden 128)、`WritingRingTrajectoryNet`(TCN+LSTM)、`predict_sequence` 流式分块、`integrate_contact_segments` 接触段积分、`trajectory_metrics` 论文指标 | 输入 6 通道顺序同 touch 路径;输出为帧位移 (Δx,Δy),窗口中点延迟 6 帧 |
| `paper_trajectory_dataset.py` | 轨迹数据层:`discover_trajectory_samples`、`build_segment_refs`(pad_trim/chunks)、`compute_normalization`、`PaperTrajectoryDataset` | 训练数据语义 `y[t]=board[t+1]-board[t]` |
| `train_paper_trajectory.py` | 轨迹训练入口:随机 70/10/20 或用户无重叠划分、TBPTT 分块训练、mask 只计接触帧、report 落盘 | 默认 `--output-dir models_trajectory`;`--split user` 为严格对照 |
| `board_scale.py` | 用原始 Sensel 触点与 clean board 做单位标定 | 原始触点也是归一化坐标,物理 mm 无法反推,默认保守用 Sensel Morph 尺寸假设 |
| `madgwick.py` / `build_madgwick_dataset.py` | Madgwick 去重力与 raw↔clean 对齐探针 | 探针结论:官方 clean acc 非 Madgwick 输出(更像滑动均值去重力),未接入主链路;见 NOTES |
| `build_clean_dataset.py` | raw→clean 转换脚本:裁剪 ring∩board 窗口(头 0.6 s / 尾 1.0 s)、4977.75 us 均匀网格、gyro 原样截取、acc 去重力、Sensel 触点线性重采样为 board/mask/y | 与官方 clean 的一致性:gyro 逐位一致、board ~1e-4、mask 99–100%、y corr 0.96–0.998;acc 去重力与网格锚点不可精确还原;`--validate-against` 输出报告;详见 NOTES「原始数据 → 干净数据」 |
| `character_dataset.py` / `character_model.py` / `train_character_classifier.py` | 任务 2 字母识别:逐字符切分、1D CNN、GT/重建混训与评估 | 26 类大小写无关;严格口径端到端 top1 86.30%/top3 96.05%(n=708 全测试集,前端 b2u 未见测试用户),GT 93.93%;上界口径(前端 x3 随机划分)92.66%;论文 IME 88.7%(协议不同) |
| `word_dataset.py` / `word_recognition.py` / `evaluate_word_dtw.py` | 任务 2 词识别基线:整词切分、最近实例词表匹配与评测 | 用户无重叠闭集 top1 63.2%(GT)/61.4%(recon),dtw+deskew 落盘口径 |
| `word_ctc.py` / `train_word_ctc.py` | 任务 2 CTC 词识别:CNN+BiLSTM+CTC、词表对齐解码与训练 | 严格口径重建子集词表对齐 64.17%(connected 56.49/unconnected 72.45)、raw 58.58%,GT 66.92%,合并 n=2612 对齐 65.54%;上界口径 65.7%;随机划分 81.8% |
| `detector.py` | 规则基线状态机:`writing_start` / `pen_up` 事件、`valid_operation` 门控 | 仅作回归测试与无模型兜底,不代表论文最终 detector |
| `features.py` | 规则基线的逐帧特征(`motion_score`、`impact_score` 等) | 阈值参数在 `DetectorConfig` |
| `io.py` | `read_sensor_csv` / `iter_samples`,统一列名与单位 | 无 `lin_acc_*` 时用去整段均值的兜底,非最终预处理 |
| `cli.py` | `python -m writing_state.cli <csv>` 离线回放 | 可输出 events / segments CSV |
| `realtime.py` | `RealtimeWritingStateHandler.on_raw(frame, bus)` 实时处理并向 `writing_state`、`writing_state_frame` 发布 | 下游只应消费 `valid_operation=true` |
| `board_plot.py` / `ring_plot.py` | 触控板与戒指数据可视化 | 数据分析辅助 |
| `visualize_results.py` | 结果可视化入口:重跑 test 推理,生成轨迹对比与误差统计、字母混淆矩阵/逐类准确率/识别样例、词识别对照/解码样例/GT-重建对照卡片 | 输出 `outputs/figures/fig_*.png` + `letter_accuracy.csv` + `metrics_summary.json`;`--figures trajectory,character,word` 选子集;`--protocol strict` 切到严格口径(前端 b2u + `user_both_ud`/`ctc_user_both_ud`,建议配 `--output-dir outputs/figures_strict`);默认复用 x3 轨迹模型与 user_both 识别模型 |
| `test_writing_state.py` / `test_paper_touch.py` | 规则基线与学习模型的单元测试 | 见"配置与依赖"中的运行方式 |

## 关键入口

| 入口 | 作用 | 常用命令(在仓库父目录运行,该目录下需有软链 `writing_state -> Ring`) |
| --- | --- | --- |
| `writing_state.paper_touch` | 查看模型摘要 / 导出窗口等 | `python -m writing_state.paper_touch --json` |
| `writing_state.train_paper_touch` | 训练 touch detector | `python -m writing_state.train_paper_touch --epochs 8 --batch-size 512 --max-train-windows-per-class 30000 --max-eval-windows-per-class 15000 --output-dir writing_state/models` |
| `writing_state.evaluate_paper_touch` | touch detector 独立评估 | `python -m writing_state.evaluate_paper_touch --checkpoint writing_state/models/exp_50/paper_touch_resnet.pt --split test --max-windows-per-class 15000` |
| `writing_state.cli` | 规则基线 CSV 回放 | `python -m writing_state.cli data/xxx.csv --events-out outputs/writing_events.csv --segments-out outputs/writing_segments.csv` |
| `unittest` | 单元测试 | 在父目录运行 `python -m unittest writing_state.test_paper_touch writing_state.test_writing_state` |
| Python API | 加载模型 | `load_touch_classifier("writing_state/models/paper_touch_resnet.pt")` + `PaperTouchDetector` |
| `writing_state.train_paper_trajectory` | 训练轨迹模型 | `python -m writing_state.train_paper_trajectory --data-root /data/huyang/datasets/WritingRing/clean_data_delete_g/data --split random --epochs 500 --output-dir writing_state/models_trajectory` |
| `writing_state.paper_trajectory` | 查看轨迹模型摘要/校验 checkpoint | `python -m writing_state.paper_trajectory --json` |
| `writing_state.board_scale` | board 单位标定 | `python -m writing_state.board_scale --clean-sample .../data/user_0/1/0` |
| Python API(轨迹) | 加载轨迹模型 | `load_trajectory_predictor("writing_state/models_trajectory/paper_trajectory.pt")` + `predict_deltas / integrate_contact_segments` |
| `writing_state.visualize_results` | 生成结果图(轨迹/字母/单词) | `python -m writing_state.visualize_results --figures trajectory,character,word` |

## 配置与依赖

- 运行环境:conda 环境 `vae`(含 `torch 2.2.0+cu121`);base 环境无 torch。
- 依赖:torch、numpy、pandas;轨迹路径可选 `matplotlib`(绘图,缺失时跳过错图)、`compress_pickle`(board 标定,缺失时降级);仓库暂无 `requirements.txt`(待补充)。
- 配置方式:无独立 config 目录,参数为 dataclass(`PaperTouchConfig`、`PaperTrajectoryConfig`、`DetectorConfig`)与训练脚本 argparse 参数。
- 轨迹数据语义(已实测):`y[t] = board[t+1] - board[t]`;`board` 为归一化坐标(原始 Sensel 触点同样归一化),mm 误差按 Sensel Morph 240×169.5 mm 假设换算,结论写入 `report.board_scale_note`。
- 数据管线产物:`outputs/data_pipeline/`(raw→clean 检查图);重建数据集本身写到仓库外(如 `/data/huyang/datasets/WritingRing/clean_data_rebuilt/`)。
- 轨迹产物:`models_trajectory/`(随机划分主结果)、`models_trajectory_user/`(用户无重叠对照);结果图输出到 `outputs/figures/`(已 gitignore),指标汇总 `outputs/figures/metrics_summary.json`。
- 数据:训练数据默认 `writing_state/clean_data_delete_g/data`(不在仓库内);本机已下载到 `/data/huyang/datasets/WritingRing/clean_data_delete_g/data`(原始数据在同级 `data/`),目录结构 `user_*/{action}/*_x.npy + *_mask.npy`(另含 `_y/_board/_timestamp.npy`)。
- 模型输出:`models/`(含 `exp_30/`、`exp_50/` 实验归档)、`models_probe/`、`models_smoke/`(checkpoint + report.json,已纳入版本管理)。
- 无 log 目录;训练/测试指标写入 report.json。
