# TODO

> 任务清单:每条任务以 "( )编号." 开头,完成后改为 "(X)编号.",不删除已完成条目。
> 背景段给出项目事实速查,动手前先读。

## 背景

**目标**:把 WritingRing 的书写接触/抬笔检测器工程化落地到实时书写系统(详见 `.trae/documents/DESIGN.md`)。

- 结构文件在 `.trae/documents/STRUCTURE.md`;模块职责、入口与数据约定以其为准。
- 模型/检测代码在 `paper_touch.py`(学习模型主路径),`detector.py` + `features.py`(规则基线兜底)。
- 训练入口在 `train_paper_touch.py`,模块名 `writing_state.train_paper_touch`;数据层在 `paper_dataset.py`。
- 数据读取在 `io.py`(CSV 列名适配,SmartRing / OpenZen);实时桥接在 `realtime.py`;离线回放在 `cli.py`。
- 绘图工具在 `board_plot.py`、`ring_plot.py`。
- 环境为 conda `vae`(torch 2.2.0+cu121);依赖 torch/numpy/pandas,仓库暂无 `requirements.txt`。
- 模块挂载:父目录软链 `writing_state -> Ring`;命令按 README 在父目录以 `python -m writing_state.*` 运行。
- 模型输出在 `models/`(主)、`models_probe/`(探测)、`models_smoke/`(冒烟),均为 checkpoint + report.json。
- 数据集默认 `writing_state/clean_data_delete_g/data`,目录结构 `user_*/{action}/*_x.npy` + `*_mask.npy`,不在仓库内。
- 单元测试为 `test_paper_touch.py`、`test_writing_state.py`;无 log 目录,指标写入 report.json。

你所做的所有改动要考虑对于上面所有内容的影响。

## 任务清单

> 尚未有任何任务。新任务由用户在下方追加,格式:( )编号. 任务描述
