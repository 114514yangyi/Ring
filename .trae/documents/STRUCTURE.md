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
├── detector.py                # 规则基线:硬阈值 + 状态机(接触/抬笔/无效空中段)
├── features.py                # 规则基线的低维特征提取(加速度/角速度/加加速度/冲击)
├── io.py                      # CSV 读取与列名适配(SmartRing / OpenZen),线性加速度缺失时的离线兜底
├── cli.py                     # 规则基线离线回放命令行,导出 events / segments CSV
├── realtime.py                # 实时接入:RealtimeWritingStateHandler,桥接 raw frame 与 ImuBus
├── board_plot.py              # 触控板数据可视化(轨迹 + 压力)
├── ring_plot.py               # 戒指 IMU 数据可视化
├── test_paper_touch.py        # 学习模型路径单元测试(形状/重采样/结构/事件/checkpoint 加载)
├── test_writing_state.py      # 规则基线单元测试(合成序列回归)
├── models/                    # 主训练输出:paper_touch_resnet.pt + paper_touch_report.json
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
| `detector.py` | 规则基线状态机:`writing_start` / `pen_up` 事件、`valid_operation` 门控 | 仅作回归测试与无模型兜底,不代表论文最终 detector |
| `features.py` | 规则基线的逐帧特征(`motion_score`、`impact_score` 等) | 阈值参数在 `DetectorConfig` |
| `io.py` | `read_sensor_csv` / `iter_samples`,统一列名与单位 | 无 `lin_acc_*` 时用去整段均值的兜底,非最终预处理 |
| `cli.py` | `python -m writing_state.cli <csv>` 离线回放 | 可输出 events / segments CSV |
| `realtime.py` | `RealtimeWritingStateHandler.on_raw(frame, bus)` 实时处理并向 `writing_state`、`writing_state_frame` 发布 | 下游只应消费 `valid_operation=true` |
| `board_plot.py` / `ring_plot.py` | 触控板与戒指数据可视化 | 数据分析辅助 |
| `test_writing_state.py` / `test_paper_touch.py` | 规则基线与学习模型的单元测试 | 见"配置与依赖"中的运行方式 |

## 关键入口

| 入口 | 作用 | 常用命令(在仓库父目录运行,该目录下需有软链 `writing_state -> Ring`) |
| --- | --- | --- |
| `writing_state.paper_touch` | 查看模型摘要 / 导出窗口等 | `python -m writing_state.paper_touch --json` |
| `writing_state.train_paper_touch` | 训练 touch detector | `python -m writing_state.train_paper_touch --epochs 8 --batch-size 512 --max-train-windows-per-class 30000 --max-eval-windows-per-class 15000 --output-dir writing_state/models` |
| `writing_state.cli` | 规则基线 CSV 回放 | `python -m writing_state.cli data/xxx.csv --events-out outputs/writing_events.csv --segments-out outputs/writing_segments.csv` |
| `unittest` | 单元测试 | 在父目录运行 `python -m unittest writing_state.test_paper_touch writing_state.test_writing_state` |
| Python API | 加载模型 | `load_touch_classifier("writing_state/models/paper_touch_resnet.pt")` + `PaperTouchDetector` |

## 配置与依赖

- 运行环境:conda 环境 `vae`(含 `torch 2.2.0+cu121`);base 环境无 torch。
- 依赖:torch、numpy、pandas;仓库暂无 `requirements.txt`(待补充)。
- 配置方式:无独立 config 目录,参数为 dataclass(`PaperTouchConfig`、`DetectorConfig`)与训练脚本 argparse 参数。
- 数据:训练数据默认 `writing_state/clean_data_delete_g/data`(不在仓库内),目录结构 `user_*/{action}/*_x.npy` + `*_mask.npy`。
- 模型输出:`models/`、`models_probe/`、`models_smoke/`(checkpoint + report.json,已纳入版本管理)。
- 无 log 目录;训练/测试指标写入 report.json。
