# TODO

> 任务清单:每条任务以 "( )编号." 开头,完成后改为 "(X)编号.",不删除已完成条目。
> 背景段给出项目事实速查,动手前先读。

## 背景

**目标**:把 WritingRing 的书写接触/抬笔检测器工程化落地到实时书写系统(详见 `.trae/documents/DESIGN.md`)。

- 结构文件在 `.trae/documents/STRUCTURE.md`;模块职责、入口与数据约定以其为准。
- 模型/检测代码在 `paper_touch.py`(触控主路径),`detector.py` + `features.py`(规则基线兜底)。
- 训练入口在 `train_paper_touch.py`,模块名 `writing_state.train_paper_touch`;数据层在 `paper_dataset.py`。
- 轨迹重建在 `paper_trajectory.py`(TCN+LSTM、流式推理、接触段积分与指标)、`paper_trajectory_dataset.py`(75s 长片段/归一化)、`train_paper_trajectory.py`(随机/用户划分 + TBPTT);board 单位标定在 `board_scale.py`;测试在 `test_paper_trajectory.py`。
- 轨迹产物:`models_trajectory/`(随机划分)、`models_trajectory_user/`(用户无重叠)、`models_trajectory_topology/`(拓扑/双向实验归档);数据语义 `y[t]=board[t+1]-board[t]`,mm 按 Sensel Morph 240×169.5 mm 假设。
- 字母识别在 `character_dataset.py`(标签区间切分/重采样)、`character_model.py`(1D CNN)、`train_character_classifier.py`(GT/重建混训);产物 `models_character/{user_both,user_gt,random_both}`;测试 `test_character_classifier.py`。
- 数据读取在 `io.py`(CSV 列名适配,SmartRing / OpenZen);实时桥接在 `realtime.py`;离线回放在 `cli.py`。
- 绘图工具在 `board_plot.py`、`ring_plot.py`。
- 环境为 conda `vae`(torch 2.2.0+cu121);依赖 torch/numpy/pandas,仓库暂无 `requirements.txt`。
- 模块挂载:父目录软链 `writing_state -> Ring`;命令按 README 在父目录以 `python -m writing_state.*` 运行。
- 模型输出在 `models/`(主)、`models_probe/`(探测)、`models_smoke/`(冒烟),均为 checkpoint + report.json。
- 数据集在仓库外:`/data/huyang/datasets/WritingRing/clean_data_delete_g/data`(清洗版,训练用),原始版在同级 `data/`;目录结构 `user_*/{action}/*_x.npy` + `*_mask.npy`。
- 单元测试为 `test_paper_touch.py`、`test_writing_state.py`;无 log 目录,指标写入 report.json。

你所做的所有改动要考虑对于上面所有内容的影响。

## 任务清单

> 尚未有任何任务。新任务由用户在下方追加,格式:( )编号. 任务描述
> 新增任务时,须同时维护"背景"以及 DESIGN/STRUCTURE/NOTES 的对应描述。

(X)1. 【阶段性目标·本轮】复现 WritingRing(He et al., CHI 2025)的轨迹重建(字符识别不在本轮):
   - (a) 轨迹模型:复现 TCN+LSTM 流式指尖速度预测(输入去重力 acc+gyro;总窗 13 帧、TCN 窗长 3;LSTM hidden 128;预测窗口中点 (vx, vy));
   - (b) 训练:长片段(75 s / 15000 帧)流式 batch 训练、MSE、仅接触帧(mask=1)计入 loss;先复现论文设定(随机 70/10/20),另报用户无重叠划分作为严格对照;
   - (c) 轨迹重建与评估:仅对接触段积分速度重建 2D 轨迹,按论文指标(逐点归一化距离 + 无归一化 mm 误差)与 touchpad 真值对比;
   - (d) 交付:训练/评估入口、checkpoint + report.json、单元测试、README/STRUCTURE/DESIGN/NOTES 同步更新。
   结果(500 epochs,board 真值):随机划分 test 归一化 0.152 / 6.22 mm(论文 0.073 / 1.64 mm),用户无重叠 0.214 / 8.53 mm;指标差距归因见 NOTES。

(X)3. 【几乎完成·到可复现瓶颈】轨迹重建指标向论文对齐:试验更接近论文的 TCN 拓扑、整段 BPTT(或更长块/梯度累积)、lr 调度与更长训练,目标 test 归一化误差 ≈0.073、mm ≈1.64。目标未达成,但全部可复现路线已探索完毕。
   进展(2026-10-03,训练方式/超参阶段已筛完):最佳 G2 = TBPTT3000+每 chunk 一步+dropout0+constant,test 0.149 / 6.10 mm(旧 0.152/6.22);cosine、全 BPTT batch8 未胜出。诊断:完美 `y` 噪声底 0.038/1.28mm(论文已近底),当前逐帧 R² 0.83/0.88 → 需 ~0.98。
   进展(架构阶段):① ~~Madgwick 探针~~(官方 clean 非 Madgwick,不切换输入);② ~~TCN 拓扑~~ 最佳 V4(dilated 32/64/128+2层LSTM)= 0.1242 / 5.08 mm;③ ~~其他方案~~ 逐样本归一化、轨迹辅助损失、3 层 LSTM(0.1261)、全 BPTT 均未超过 V4;④ 免训练 ensemble V1+V3+V4+U1 = 0.1117 / 4.57 mm。
   最终结论(双向延长训练后):单模最佳 X3 = 0.1049/4.15(离线双向,B2 精调 450ep),最佳总体 ensemble x1+x3+b2+v4 = 0.1003/4.01;论文流式口径最佳 V4 = 0.1242/5.08。距论文 0.073 约 1.37x,噪声底 0.038/1.28mm;单模/ensemble 均进入平台,再翻倍训练预计收益 <5%。全部可复现路线已探索完毕,剩余差距疑似来自论文未公开的 TCN/训练实现细节或评估口径。状态:达到当前仓库可复现瓶颈,待决定结题或等待新信息。

(X)2. 【完成·字母+单词】轨迹→字符识别:利用原始 `*_timestamp.txt` 的逐字符时间戳+标签切分轨迹,训练轨迹→26 字母分类器(论文用的 Google IME 无法复现,需自训替代),报告字母识别率与混淆矩阵;再扩展单词(连笔/断笔、词表匹配)。
   字母阶段结果(用户无重叠 test user_16/18/19/20,user_17 无合格样本):GT-only 训练 GT 93.2%/99.0%;GT+重建混训 GT 94.8%/99.6%、重建轨迹端到端 92.7%/98.9%(上界口径:重建前端 x3 为随机划分,含 84.5% 测试样本);随机划分混训 GT 93.8%/recon 92.7%;混淆集中在 L/I、I/Z、T/Z 等形近字母。
   【2026-10-05 严格端到端重跑完成】前端 b2u(user_0-13 训练、16-20 测试)+ 字母分类器 `models_character/user_both_ud`:全测试集 n=708,重建轨迹 top1 **86.30%/top3 96.05%**,GT 93.93%/98.87%;前端未见使字母 top1 相对上界下降 6.4 pt(原 88.2% 子集估计偏乐观)。逐类最低 L 54.5%、I 63.0%,满分 C/Y。
   产物:`models_character/{user_both,user_gt,random_both}`;代码 `character_dataset.py`/`character_model.py`/`train_character_classifier.py`。
   单词阶段基线(2026-10-03,2026-10-05 按落盘报告更正):414 词块、8364 标签、561 唯一词;整词轨迹 + 最近训练实例(dtw+deskew,闭集 1084):用户无重叠 test n=1306,GT 闭集 top1 63.2%(connected 44.5% / unconnected 61.0%)、重建 61.4%;产物 `models_words/user`。对照论文 IME:unconnected 68.2 / connected 53.1(3000 词表 84.4/74.2),口径不同。
   单词 CTC 阶段(2026-10-03,已完成;2026-10-05 补上界口径):CNN+BiLSTM+CTC + 编辑距离词表对齐;用户无重叠 GT+重建混训 raw 61.2%/词表对齐 66.2%(GT+重建合并口径;重建子集 raw 60.6%/对齐 65.7%,上界——重建前端 x3 含 69.4% 测试样例,严格未见子集约 60.0%,n=400;重建子集 connected 57.5/unconnected 74.5;严格口径全测试集实测见下行条目),GT-only 54.9%/62.3%,随机划分 68.9%/81.8%;产物 `models_words/ctc_{user_both,user_gt,random_both}`。对照论文 IME(3000 词表 84.4/74.2)口径不同,量级相当。
   【2026-10-05 严格端到端重跑完成】前端 b2u + `models_words/ctc_user_both_ud`:全测试集 n=1306,重建子集 raw 58.58%/词表对齐 **64.17%**(connected 56.49/unconnected 72.45);GT 子集 raw 63.94%/对齐 66.92%;合并(n=2612)raw 61.26/对齐 65.54。相对上界仅 −1.5 pt,**词级任务对前端误差不敏感**;unconnected 72.5/connected 56.5 略高于论文开放词表 IME(68.2/53.1)。
   可选后续:词表约束 beam search、更长训练/多种子 ensemble;严格跨用户端到端已用用户无重叠前端 b2u 完成重跑(前端 0.159/5.54 mm、字母 86.30%、单词 64.17%)。详见 NOTES 文末「严格端到端重跑」。

(X)4. 【完成·数据管线】确认训练数据为官方 clean 版并逆向 raw→clean 转换规则:裁剪窗口(头 0.6s/尾 1.0s)、4977.75us 均匀网格、gyro 逐位截取、acc 去重力(近似)、Sensel 触点重采样为 board/mask/y。交付 `build_clean_dataset.py`(含 `--validate-against` 一致性报告,gyro Δ=0、board 2e-4、mask 99%、y corr 0.997)与 `outputs/data_pipeline/raw_to_clean_overview.png`;README/NOTES/STRUCTURE 已同步。遗留:acc 去重力与网格锚点不可精确还原。
