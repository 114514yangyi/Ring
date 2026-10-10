# NOTES — 思考与问题记录

> 记录开发过程中的问题、思考与决策。格式:日期 + 问题/思考 + 结论/决定。
> 此文件随时间追加,不删除历史条目。

## 2026-09-30

- 问题/思考:仓库目录名为 `Ring`,但 README、测试与所有运行命令都按 `writing_state` 包名导入;base 环境无 torch,仓库也没有 `requirements.txt`。
- 结论/决定:初始化 `.trae` 知识库;模块挂载方式确定为在父目录建软链 `writing_state -> Ring`(不重命名仓库、不改导入);运行环境记录为 conda `vae`(torch 2.2.0+cu121),依赖清单待后续补充。

## 2026-10-03

- 任务:复现 WritingRing 轨迹重建(TCN+LSTM 流式速度预测 + 接触段积分),字符识别留作后续。论文为 CC-BY 但 ACM 直连 403,正文细节通过检索逐段核实。
- 论文方法事实:Madgwick(β=0.041)姿态估计去重力;13 帧窗、TCN kernel=3、TCN 输出 1×128、LSTM hidden=128、预测窗口中点速度;75 s/15000 帧流式训练(短补零长对称裁剪),batch 16、Adam 1e-3、MSE、500 epochs、随机 70/10/20,仅接触帧计 loss;评分为归一化逐点距离/轨迹 bbox 对角线(均值 0.073,80% <0.1;无归一化 0.164 cm);消融:碎片化 0.116→流式 0.073、无 TCN 0.095→0.073;字符识别用 Google IME(字母 88.7%、单词 68.2%)。
- 数据语义(实测):`y[t] = board[t+1]-board[t]`(帧级相关 0.97,积分后 0.9995);【2026-10-06 更正:`y` 不是网格差分——严格复算 corr(y, board[t+1]-board[t])≈0.06–0.11,`y` = 原始 board 帧率速度插值到网格×网格步长(相关 0.96–0.998);积分口径仍成立,详见文末「原始数据 → 干净数据」】;`board` 与原始 Sensel 触点均为归一化坐标,物理 mm 无法从原始数据反推,mm 指标按 Sensel Morph 240×169.5 mm 假设;2 个样本 x/timestamp 长度不一致(user_0/0/0、user_7/0/0)已跳过。
- 实现决策:TCN 拓扑论文未公开,默认 6 层 valid 卷积(16/16/32/32/64/128,同通道加残差);训练用 TBPTT 分块(块长 3000,块间 detach 隐状态、每块 step),与论文整段 BPTT 有差异;接触段内 NaN 预测按有限子段切分并记录 lost_frames;env `vae` 原缺 matplotlib 与 compress_pickle,已分别安装(mathplotlib 缺失时绘图降级)。
- 运行环境:GPU0(空闲 4090),batch 32、chunk 3000,约 6-7 s/epoch。
- 结果(2026-10-03,board 真值评估,均 500 epochs):随机划分 test 归一化 0.152(误差 >0.1 的逐点占比 57.3%,p50=0.152)/ 6.22 mm;用户无重叠 test 0.214 / 8.53 mm。预测幅度比 pred/gt 中位 0.914(模型未坍塌、有真实跟踪),段级中位误差 ~0.10。与论文 0.073 / 1.64 mm 仍有约 2x / 3.8x 差距。
- 代码审查(独立代理)修复:评估真值由 `y` 改为 touchpad `board`(SPEC 要求;修正后指标略升),report 增加 sample_audit/y_anchor/board_mm_scale,新增 `--init-checkpoint` + `--epochs 0` 评估模式。

### 任务 3 训练方式/超参筛选(2026-10-03)

- 实验(随机划分 seed 42,指标为 board 真值 val/test):A = TBPTT3000+每 batch 一步 100ep best val 0.297;B = 全 BPTT(15000)+batch32 100ep 0.296;G1 = TBPTT3000+每 chunk 一步+cosine 500ep test **0.159 / 6.43 mm**;G2 = 同 G1 但 dropout 0 test **0.149 / 6.10 mm**(现为主模型);F1 = 全 BPTT+batch8+cosine 500ep(314 截断)best val 0.179。
- 结论:改成"每 batch 一步"后收敛显著慢于旧"每 chunk 一步"(旧 500ep 等效更新多 ~5x);cosine 与 dropout0 合计收益 <0.01,训练方式/超参阶段到顶。
- 上限诊断:完美预测官方 `y` 的协议噪声底(逐段 L)为 norm 0.0377 / 1.28 mm——论文 0.073 / 1.64 mm 已接近该底;当前模型逐帧 R² 仅 0.83/0.88、幅度比 0.94、绝对误差 5.8 mm,距底约 4.5x。
- 判定与下一步(架构/输入阶段):要达 0.073 需提高逐帧精度至 R²≈0.98,候选:①复现 Madgwick 姿态归一化并与官方 clean acc 对照(探针,便宜);②TCN 拓扑(扩张残差/flatten 变体);③全 BPTT+梯度裁剪的充分长训练。
- 代码增量:`--lr-schedule {constant,cosine}`、`--update-every {batch,chunk}`(默认 batch,chunk 即旧语义);全序列 BPTT 用 `--chunk-len 20000` 实现。

### Madgwick 输入探针(2026-10-03,任务 3 架构/输入阶段)

- 对齐:每个 clean 样本 ↔ 同名 `{id}_ring_0.bin`,FFT 互相关 + 整数偏移精修(gyro 逐轴相关可达 1.0000,前置裁剪约 8-13 s);探针工具 `madgwick.py`、`build_madgwick_dataset.py`(带测试,暂未接入主链路)。
- 结论:官方 clean acc 与论文所述 Madgwick(β=0.041)去重力不吻合——canonical(修正项乘 dt)corr 0.30;漏 dt 版本 corr 0.65;β 越小越接近 clean(β=0.005 → 0.87);最接近的是 0.5 s 滑动均值去重力(corr 0.90)。发布版 clean 更像"局部均值/高通"预处理,而非 Madgwick。
- 判定:暂不重建 Madgwick 输入数据集(结果不确定、成本高,且官方 clean 很可能就是作者训练所用输入);架构阶段优先做 TCN 拓扑变体。

### 任务 3 TCN 拓扑筛选(2026-10-03)

- 对照实验(随机划分,500 epochs,chunk-step,dropout 0,batch 32):valid 16..128(基线 G2)0.1488 / 6.10 mm;valid 加宽 32..128(V2)0.1430 / 5.81;dilated 16/32/64(V3)0.1334 / 5.40;dilated 32/64/128(V1)0.1299 / 5.24;**dilated 32/64/128 + 2 层 LSTM(V4)0.1242 / 5.08(新主模型)**;V1+V3+V4 预测位移平均 ensemble(免训练)0.1142 / 4.65 mm。cosine 调度反而更差(V5 0.154);4 层更宽 dilated(V6)未跑完(0.135@263,已终止)。
- 结论:架构方向有效(0.149 → 0.124,ensemble 0.114),但距论文 0.073 仍有 ~1.6-1.7x;官方 y 噪声底 0.038 / 1.28 mm。剩余差距很可能来自论文未公开的 TCN/训练细节或评估口径,继续加大容量的边际收益递减。
- 新增能力:`--tcn-style {valid,dilated}`、`--tcn-channels`、`--lstm-layers`;拓扑 checkpoint 归档于 `models_trajectory_topology/{v1,v3,v4}`(ensemble 复现:三个 checkpoint 的预测位移逐帧平均后积分评估)。
- 注意:用户无重叠对照仍是旧架构(0.214 / 8.53 mm),与新主模型不可直接对比,如需可比需按 V4 重训。

### 任务 3 其他方案探索(2026-10-03,续)

- 逐样本输入归一化(P2):明显更差(500ep 未跑完即止,早期 best 0.31)。
- 轨迹级辅助损失(P1,窗 50/权重 0.5):0.1408 / 5.21,不如 V4;长窗(200)更差。
- 3 层 LSTM(U1):0.1261 / 5.20,与 V4 持平略差;已归档 `models_trajectory_topology/u1`。
- 全 BPTT batch8(U2):曲线明显落后,216ep 终止(best 0.147)。
- 免训练 ensemble:V4+U1 = 0.1149 / 4.73;V1+V3+V4+U1 = **0.1117 / 4.57 mm**。
- 结论:所有单模型方案收敛在 0.124-0.141,ensemble 到 0.112;距论文 0.073 仍 ~1.5x,距噪声底 0.038 约 3x。剩余差距大概率来自论文未公开的 TCN/训练实现细节或评估口径,在本仓库可复现的范围内已到瓶颈。

### 任务 2 第一阶段:轨迹→字母识别(2026-10-03)

- 切分:标签时间戳区间 + 接触段分组(多笔画合并),26 类大小写无关,`wrong` 丢弃、裁剪外标签跳过;真实数据 3845 个字符样本、20 用户、26 类(每类 89-176)。
- 分类器:小 1D CNN(2→32→64→128,全局池化→FC26),弧长重采样 64 点 + 首点归零 + bbox 对角线归一化;训练增广(旋转 ±10°/缩放 0.9-1.1)。
- 结果(用户无重叠,test 用户 user_17-20):
  - GT-only 训练:GT 轨迹 top1 **93.2%** / top3 99.0%;
  - GT+重建混训:GT **94.8%/99.6%**,重建轨迹(端到端,X3 轨迹模型)**92.7%/98.9%**;随机划分混训:GT 93.8%、recon 92.7%。【2026-10-05 更正:重建端到端为上界口径——X3 前端随机划分含 84.5% 的测试样例;X3 未见子集估计 top1 = 88.2%(n=110)偏乐观;正式重跑(用户无重叠前端 b2u)全测试集严格 top1 = 86.30%,见文末「严格端到端重跑」】
  - 主要混淆对(L→I、I→Z、T→Z、B→P)为经典形近字母。
- 与论文对照:论文 Google IME 在实时研究中对重建轨迹字母识别 88.7%;我们自训分类器离线端到端 92.7%(上界口径)/86.30%(严格口径,全测试集 n=708),**协议不同**(离线、26 类、自有分类器 vs IME、实时、大小写处理后),仅作量级对照。
- 诚实限制:X3 轨迹模型用随机划分训练(含字符测试用户的数据),字符分类器本身未见测试用户;严格跨用户端到端需用用户无重叠的轨迹模型重跑 recon(已记录,未执行)。【2026-10-05 已完成:见文末「严格端到端重跑(user-disjoint 前端 b2u)」,字母严格 86.30%、单词严格 64.17%】
- 第二阶段(单词:连笔/断笔、词表匹配)基线(2026-10-03):414 词块、8364 词标签、561 唯一词(数据自带词表,论文用外部 3000 词表);按标签区间切分整词轨迹(重采样 128 点、bbox 归一化),用"最近训练实例"做识别(标准欧氏;尝试 DTW 重排与 deskew 均无提升)。
  - 用户无重叠 test(n=1306,其中 1084 词在训练词表内):GT 闭集 top1 63.6% / top5 61.0%(overall top5);重建轨迹闭集 63.5%,与 GT 几乎无差(长轨迹对重建误差不敏感);connected top1 45%、unconnected 61%。【2026-10-05 更正:落盘 dtw+deskew 报告为 GT 闭集 63.2%(n=1084)、重建 61.4%;63.5 来自旧 euclidean 运行,以磁盘报告为准】
  - 对照论文:unconnected 68.16 / connected 53.14(开放词表,Google IME),3000 词表下 84.36/74.17;我们的词表更小(561)但无语言模型,量级接近但未达。
  - 下一步候选:CTC 字母序列模型 + 词表 beam/edit-distance 解码(专门识别器,论文作者也建议)。
- CTC 词识别(2026-10-03):CNN(2→64→128)+ 双向 2 层 LSTM(128)+ 逐帧 27 类 softmax,CTC loss + 贪心解码 + 编辑距离词表对齐(词表=训练词集,561 词);300 epochs、batch 64、Adam 1e-3、梯度裁剪 5.0、训练增广(±10° 旋转/0.9-1.1 缩放)。
  - 用户无重叠 test(n=1306):GT-only raw 54.9%/snapped 62.3%(connected 54.1/unconnected 71.0);**GT+重建混训 raw 61.2%/snapped 66.2%(GT+重建合并口径;重建子集 raw 60.6%/snapped 65.7%,其中 connected 57.5/unconnected 74.5)**,其中 GT 子集 snapped 66.7%、recon 子集 65.7%【上界口径:X3 前端含 69.4% 测试样例,严格未见子集约 60.0%(n=400)】;随机划分混训 raw 68.9%/snapped 81.8%(connected 76.1/unconnected 87.7)。
  - 对照:最近邻基线闭集 63.2%(dtw+deskew 落盘;connected 44.5/unconnected 61.0);CTC 在分组上提升 +13-14pts。论文 IME:开放词表 68.2(unconnected)/53.1(connected),3000 词表 84.4/74.2;口径不同(词表规模、语言模型),量级相当。
  - 调试教训:初次 60 epochs 纯 CNN 坍塌为重复字母,且 `nn.CTCLoss` 需传 log-softmax(传原始 logits 时首步 loss 为负);40 样本 1000 步可 100% 过拟合验证实现正确。产物 `models_words/ctc_{user_both,user_gt,random_both}`。
  - 可继续提升:词表约束 beam search、更长训练、多种子 ensemble;当前已交付。

### 任务 3 离线双向最后尝试(2026-10-03)

- 动机:离线评估允许双向上下文;作为"可达上限"探针(偏离论文流式设计)。新增 `--bidirectional`(自动强制全序列前向、禁止分块)。
- 结果:B1(dilated 32/64/128 + 双向 1 层 LSTM,300ep)0.1353 / 5.38;B2(同 + 双向 2 层 LSTM,batch4,300ep)**0.1100 / 4.36 mm**,单模型最佳,且训练末段仍在下降(ep200 0.1170 → ep288 0.1133 → test 0.1100)。
- ensemble:B2+V4 = 0.1050 / 4.23;V1+V3+V4+U1+B2 = 0.1053 / 4.29。
- 判定:双向带来实质增益(0.124 → 0.110),但仍距 0.073 ~1.4x;checkpoint 归档 `models_trajectory_topology/b2`。
- 延长训练(从 B2 init,450 epochs,batch 8):X1(lr 1e-3)test 0.1059 / 4.21;X3(lr 2e-4)**0.1049 / 4.15**(单模最佳);两条都在 ~ep224 后进入平台,再翻倍训练预计收益 <5%。
- 最终 ensemble(x1+x3+b2+v4 位移平均,免训练)**0.1003 / 4.01 mm**;x3+b2+v4 = 0.1014 / 4.06。
- 总判定:全部可复现路线(训练语义/调度/dropout、输入预处理、TCN 拓扑/深度/宽度、LSTM 深度/双向、辅助损失、归一化、全 BPTT、多模型 ensemble)探索完毕,单模 0.105、ensemble 0.100,距论文 0.073 约 1.37x,距噪声底 0.038 约 2.6x。进一步逼近需要论文未公开的 TCN/训练实现细节或评估口径澄清;当前仓库可复现范围内已到瓶颈。
- 差距归因与后续:TCN 拓扑为默认假设(论文未公开);TBPTT 分块(块长 3000、块间 detach、每块一次优化步)替代整段 BPTT;未做 lr 调度。后续任务 3 将试更接近论文的拓扑/整段回传/更长训练。

## 2026-10-01

- 问题/思考:训练数据 `clean_data_delete_g` 在本机不存在;官方来源为 Hugging Face `dBHz/WritingRing`,直连 huggingface.co 不通,`/data/dataset` 当前用户无写权限。
- 结论/决定:经 `hf-mirror.com` 镜像下载并解压到仓库外 `/data/huyang/datasets/WritingRing/`:`clean_data_delete_g/`(清洗版,441 MB,21 用户 566 样本)+ `data/`(原始版,12 GB,606 组 `*_ring_0/1.bin + *_board_*.gz + *_timestamp.txt`),两个 zip 保留。两文件 SHA-256 与官方 LFS oid 一致。训练用 `--data-root /data/huyang/datasets/WritingRing/clean_data_delete_g/data`(该数据发现的 566 样本与已训练模型的 train/val/test 用户划分完全一致)。

### 清洗版 vs 原始版差异(实测,user_0/0 为例)

- 时间轴:原始 ring 时间戳 70% 重复(10247 帧仅 3099 个唯一值),不可直接当采样时钟;清洗版按 200 Hz 均匀网格重采样,并裁剪到与触控板重叠的区间(样本开头约 8 s、结尾约 1 s 被裁)。
- 去重力:原始 `|acc|` 均值 9.93 m/s²(含重力),清洗版 acc 各轴均值≈0;gyro 与原始逐位一致(最大差 0.0)。与滑动均值/Butterworth 高通的相关性仅 0.84~0.95,说明不是简单去均值/线性高通,精确去重力算法未公开。
- 触控板:原始 `*_board_*.gz` 是 compress_pickle 序列化的 `FrameData` 列表(~130 Hz,含 x/y/force/area 等);清洗版 `board.npy` 是接触点 x/y 线性插值到 200 Hz 网格(与最近原始点差 std≈0.001~0.003,无一个完全相等),无接触处置 0;`mask.npy` 与 board 非零完全一致;`y.npy` ≈ board 的逐帧增量(相关 0.998),不是绝对轨迹。
- 样本量:606 → 566,丢弃 40 组;字符/单词文本标签只保留在原始 `*_timestamp.txt`,清洗版无字符标签。2 个样本存在 `x` 与 `timestamp` 长度不一致的瑕疵(user_0/0/0、user_7/0/0)。

## 2026-10-05

- 任务:为用户讲解当前轨迹识别/字符识别能力并可视化结果(轨迹图与识别图)。
- 新增 `visualize_results.py`(入口 `python -m writing_state.visualize_results`):重跑测试集推理,输出 `outputs/figures/` 下 6 张图与 `metrics_summary.json`;`--figures trajectory,character,word` 可选子集。
- 重算指标(轨迹:随机划分 test 113 样本、11748 个接触段,与既有 report 一致,尾数差异来自分段最小帧数与 NaN 处理):
  - V4(流式,论文口径)norm 0.1238 / 5.06 mm;b2 0.1098 / 4.33;x3(离线双向,下游识别使用)0.1048 / 4.13;ensemble(v1+x3+b2+v4 位移平均)0.1012 / 4.05。
  - x3 逐段误差中位 0.081;逐段均值 <0.1 的段占比:x3 64.7%、ensemble 66.0%(论文"80% 轨迹 <0.1",口径不完全可比)。逐点 <0.1 占比:x3 61.8%。
  - 路径长度保真:x3 重建/真值路径长度中位比 0.98、相关 r=0.991(幅度未坍塌;直方图长尾与个别段误差 >0.3 对应快速连笔/多笔画段)。
  - 字母(user 无重叠,重建轨迹):top1 92.66% / top3 98.87%(与 `models_character/user_both` report 完全一致)。【2026-10-05 口径更正:此为上界口径(x3 前端随机划分,见过 84.5% 测试样本);严格口径全测试集实测 86.30%(n=708),见文末「严格端到端重跑」】
  - 单词(user 无重叠,重建轨迹,CTC raw):60.64%(与 `models_words/ctc_user_both` report 的 recon raw 0.6064 一致);词表对齐 66.2% 取 report。
- 图:`fig_trajectory_examples`(8 个测试段 GT vs 重建,按误差分位取样)、`fig_trajectory_quality`(逐段误差直方图/路径长度散点/误差 CDF/与论文-噪声底对比)、`fig_character_recognition`(26x26 混淆矩阵 + 逐类准确率 + 汇总)、`fig_character_examples`(24 个字母轨迹与识别结果)、`fig_word_recognition`(与最近邻基线、论文数值对照)、`fig_word_examples`(8 个词轨迹与解码文本)。
- 差距结论(用户问答口径;字母/单词下文的 92.7%/66.2% 为上界口径,严格口径见文末「严格端到端重跑」:字母 86.30%、单词 64.17%):轨迹流式主模型 0.124 vs 论文 0.073(1.7x),离线双向 0.105 / ensemble 0.101(约 1.4x),mm 口径 4.05 vs 1.64(约 2.5x);协议噪声底 0.038 / 1.28 mm,论文已接近该底。字母 92.7%(重建、用户无重叠)高于论文 IME 88.7%,但协议不同(自训 26 类分类器 vs Google IME)。单词词表对齐 66.2% 与论文开放词表 IME(68.2/53.1)量级相当,低于其 3000 词表 + 语言模型(84.4/74.2);主要差距来自论文未公开的 TCN 拓扑/训练细节与 IME 语言模型,而非复现流程错误。
- 图与 `metrics_summary.json` 在 `outputs/`(已 gitignore),可随时重跑再生成。
- 追加(同日):新增两张用户要求的图——`fig_word_examples_gt_vs_pred.png`(8 个测试词的卡片:GT 轨迹 | 重建轨迹 | 真实词 | 识别词,含 raw decode 与词表对齐结果)与 `fig_letter_accuracies.png`+`letter_accuracy.csv`(26 字母逐类 top1/top3,GT 与重建两列)。逐类结果:26 字母中重建轨迹 top1 最低为 L 59.1%(n=22)与 I 66.7%(n=27),其次 V 81.8%(n=11)、T 85.0%(n=20)、A/B/D 88-90%;C/E/M/P/S/W 为 100%。整体重建 top1 92.66%、GT 94.77%,与 `models_character/user_both` report 一致。【2026-10-05 口径更正:上界口径;严格口径逐类见 `outputs/figures_strict/letter_accuracy.csv`,最低 L 54.5%、I 63.0%,满分 C/Y】

### 全库审计与汇报数据修正(2026-10-05,research-codebase-audit)

- 任务:对整个研究代码库做端到端审计(数据加载→训练→评估→文档声明一致性),产出 `audit-report.md`(0 CRITICAL / 1 MAJOR / 12 MINOR;92 个单元测试全过、23 个 report JSON 逐字段核对)。
- MAJOR·M1 跨阶段评测泄漏:下游"用户无重叠"端到端结果(字母 92.66%、单词 65.7%)所用的重建输入来自随机划分训练的轨迹前端 x3,其训练集覆盖字母测试 block 86/113(样例 598/708,84.5%)、单词测试 block 70/97(样例 906/1306,69.4%);x3 见过样本误差 0.078、未见样本 0.128。结论:这些端到端数字应记为**上界**;按 x3 未见子集复算,字母 top1 88.18%(n=110)、单词词表对齐 60.0%(n=400)。
- 其余修正:m1 统一 ensemble 成员为 v1+x3+b2+v4(0.1012 / 4.05 mm,页面口径);m2 最近邻基线按落盘 dtw+deskew 报告改为 GT 闭集 63.19% / recon 61.35%(原文档 63.6/63.5 来自另一次 euclidean 运行);m3 字母测试用户为 16/18/19/20(user_17 无合格样本);m4 噪声底 0.0377/1.28 经独立复算为 0.0369/1.305(仓库内仍无复算脚本);m5【2026-10-06 更正】`y` 与网格差分并非高相关(严格复算 0.06–0.11);`y` 是原始帧率速度重采样,段尾差异来源见文末「原始数据 → 干净数据」;m6 数据应为"566 个样本、564 个可用";m7 66.2% 是 GT+重建合并口径(重建子集 65.7%、raw 60.6%);m8 单词示例图为定向挑选(6/8 非随机抽样)。
- 处置(用户要求"把 PPT 中的数据改正确",未重训模型):`visualize_results.py` 只改文案口径不改计算;重生成 `fig_word_recognition.png`(65.7/57.5/74.5,raw 60.6)与 `fig_letter_accuracies.png`(标题标注上界);7 页结果版 deck 与 14 页报告版 deck 全部按上界口径改写并重新导出 PPTX/PDF;讲稿与指标说明、README、STRUCTURE、TODO 同步。
- 待办/遗留:严格"前端也未见"的端到端重跑已于 2026-10-05 当日完成(前端 b2u、下游 `user_both_ud`/`ctc_user_both_ud`,见文末记录);噪声底计算尚未固化为可复跑脚本;模型选择使用过 test 指标、结果均为单种子,已在审计报告中记录。

### 严格端到端重跑(2026-10-05,user-disjoint 前端 b2u)

- 动机:审计 M1 指出下游端到端数字(字母 92.66%、单词 65.7%)使用的重建前端 x3 是随机划分训练,其训练集覆盖 84.5%/69.4% 的下游测试样本;此前的"未见子集估计"(88.2%/60.0%)只是下界性质的近似。本次按用户要求做正式重跑,让前端也严格未见测试用户。
- 做法:①`train_paper_trajectory.py` 新增 `--holdout-users`,强制把 user_16-20 放进 user-split 的 test;②用 b2u 配置训练前端(`--bidirectional --tcn-style dilated --tcn-channels 32,64,128 --lstm-layers 2 --dropout 0.0`,300 epochs,lr 1e-3,batch 4;因 bidirectional 会强制全序列 BPTT,实际 chunk=15000,与 x3 的回传方式一致,差异只在划分与 lr/epochs);③下游沿用同一 `--split user`,并以 `--reconstruction-checkpoint b2u` 重跑字母分类器与单词 CTC;④`visualize_results.py` 新增 `--protocol strict`。
- 划分(实测确认):前端 train user_0-13 / val 14-15 / test 16-20(388/47/129 样本);字母分类器 test user_16/18/19/20(n=708,user_17 无合格字母样本)、val 14/15;单词 CTC test user_17-20(n=1306)、val 15/16。下游 test 用户全部落在前端 holdout 内,**且下游 val 用户(14/15/16)也不在前端训练集**,无任何跨阶段重叠。
- 结果 A|轨迹前端本身(test user_16-20,129 样本 / 11070 段 / 871,639 点):归一化 **0.1589**(p50 0.158、p90 0.287,逐点 >0.1 占 51.4%)/ **5.54 mm**。同架构随机划分 x3 = 0.1048 / 4.13 mm,论文 0.073 / 1.64 mm。即"换到真正未见用户"的代价约 +0.054 归一化(+1.4 mm);同时它比旧架构的 user-disjoint 结果(0.2137 / 8.53 mm)好 26%,说明架构改进在严格协议下同样有效。
- 结果 B|字母识别(n=708,全测试集,前端未见):重建轨迹 top1 **86.30%** / top3 96.05%;GT 子集 top1 93.93% / top3 98.87%。上界口径为 92.66/98.87(前端见过 84.5%),故**前端口径的真实代价是 −6.4 pt**,此前 88.2% 的子集估计偏乐观(该子集是 x3 没见过的那部分样本,但 x3 在这部分上本就偏弱,不能代表干净前端)。逐类:满分 C/Y,最低 L 54.5%(n=22)、I 63.0%(n=27)、A 72.0%、F 78.6%、D/P 79.3%、R 81.3%、V 81.8%。对照论文 IME 88.7%(协议不同)。
- 结果 C|单词识别(n=1306,全测试集,前端未见):重建子集 raw 58.58% / 词表对齐 **64.17%**(连笔 56.49% / 断笔 72.45%);GT 子集 raw 63.94% / 对齐 66.92%(连笔 59.44 / 断笔 75.00);GT+重建合并口径(n=2612)raw 61.26 / 对齐 65.54。上界口径重建子集为 65.70(连笔 57.5/断笔 74.5),**仅差 1.5 pt**。
- 关键结论:①**词级任务对前端误差不敏感**(−1.5 pt),字母等短笔画任务敏感(−6.4 pt),这与"最近邻基线上 GT 与 recon 几乎无差"的旧观察一致;②严格口径下 unconnected 72.5 vs 论文开放词表 68.2、connected 56.5 vs 53.1,均略高于论文开放词表 IME,但明显低于其 3000 词表 + 语言模型(84.4/74.2),差距来源依旧是词表规模与 IME 语言模型;③轨迹前端自身的严格口径误差(0.159)仍落在旧的多架构实验区间内,说明"未见用户"是比"换架构"更大的影响因素。
- 产物:`models_trajectory_topology/b2u/paper_trajectory.pt|_report.json`(+ `b2u/eval15k/` 复评报告)、`models_character/user_both_ud/`、`models_words/ctc_user_both_ud/`、`outputs/figures_strict/`(含 `letter_accuracy.csv`、`metrics_summary.json`)。
- 复现命令:见 README 各节的"严格口径端到端重跑"代码块;图用 `python -m writing_state.visualize_results --protocol strict --figures trajectory,character,word --output-dir outputs/figures_strict`。
- 遗留:仍为单种子;前端训练用了 300 epochs(lr 1e-3)而非 x3 的 450 epochs(lr 2e-4),严格口径下再调 lr/epochs 可能还有少量收益;词表约束 beam search 未做。


### 原始数据 → 干净数据:转换规则逆向与重建脚本(2026-10-06)

- 任务:确认训练所用数据是否为官方 clean 版、说明作者把原始采集数据转成 clean 的方式,并给出 raw→clean 转换脚本。
- 结论 A(用的是哪份数据):训练/评估全部用官方 clean 发布版 `clean_data_delete_g`(21 用户 / 566 样本,实际使用 564);原始版 `data/`(12 GB,606 组 `*_ring_0.bin` + `*_board_*.gz` + `*_timestamp.txt`)此前仅用于:①逐字符/逐词标签时间戳切分(字母/单词识别);②Madgwick 去重力探针与 raw↔clean 对齐实验;③board 单位标定。轨迹/字母/单词的全部结果均基于 clean 数据,不受本次脚本影响。
- 结论 B(作者管线,逐条实测复现,证据样例 8 个用户):
  1. **窗口**:clean 窗口 = board 录制区间去掉头部 0.6 s、尾部 1.0 s 后与 ring 录制取交集;落在窗口内的原始 ring 行数即 clean 长度 T(实测与官方 T 相差 0–7 帧 / 万帧)。
  2. **时间网格**:均匀,固定步长 **4977.75 µs(≈200.894 Hz)**,所有样本 dt 完全相同;官方 timestamp 与 ring 行时间戳相差 −0.07…−0.15 s(逐样本不同),说明官方网格锚在 board 会话时钟上;raw 里没有两设备时钟关系,该锚点无法精确还原(脚本默认锚在 ring 行,`--anchor-shift-us` 可平移,`--validate-against` 会拟合出最贴合的偏移)。
  3. **gyro**:= 截取的原始 ring 行,**逐位一致**(max|Δ| = 0,8/8 样本)。
  4. **lin_acc**:= 原始 acc 去重力;与论文声称的 Madgwick(β=0.041)不吻合;最接近的简单模型是 ~1 Hz 零相位高通(per-axis 相关 0.55–0.98,逐样本波动),**无法逐位复现**(作者实现未公开)。
  5. **board**:= 原始 Sensel **每帧第 0 个触点**位置线性插值到网格;重建与官方平均距离 ~1e-4 归一化单位(≈0.02 mm),8/8 样本。
  6. **mask**:= 1 当网格点两侧原始帧都有触点(AND 规则);与官方一致率 99–100%,IoU 0.98。
  7. **y** = **原始帧率**的触点位移(逐帧位移 ÷ 帧间隔)插值到网格、再乘网格步长;与官方 y 相关 0.96–0.998。**更正**:此前记录的"y[t]=board[t+1]-board[t],帧级相关 0.97"不成立——严格复算 corr(y, diff(board)) ≈ 0.06–0.11(6 个样本);因为 board 是插值后的位置、y 是原始帧率速度重采样,段尾 y 是原速度而差分跳到 0(段尾差异的真正来源)。两者积分后都等于段位移,故"积分 0.9995"仍成立。
- 脚本 `build_clean_dataset.py`(自包含:自研 gzip+pickle 兼容加载器,不依赖作者仓库/compress_pickle;含 `--validate-against` 逐样本一致性报告):
  - 快速验证:`python -m writing_state.build_clean_dataset --output-root /tmp/clean_rebuilt --users user_0 --actions 0 --samples 1,2,3 --validate-against /data/huyang/datasets/WritingRing/clean_data_delete_g/data`
  - 全量重建:`python -m writing_state.build_clean_dataset --output-root /data/huyang/datasets/WritingRing/clean_data_rebuilt/data --workers 4`
  - 验证(8 用户 × 1 样本):gyro max|Δ|=0;board mean dist 4e-5…4e-4;mask 一致率 0.987–1.000(IoU 0.97–0.99);y corr 0.96–0.998(相对误差 ~4%);acc corr 0.55–0.98;拟合 anchor 偏移 −72…−149 ms。
  - 全量 `user_0`(32 样本)重建 0 失败(≈30 MB,官方同用户 28 MB),产物可被 `paper_trajectory_dataset.discover_trajectory_samples/load_sample_arrays` 直接读取。
- 可视化:`outputs/data_pipeline/raw_to_clean_overview.png`(a: 原始 6 通道;b: 单笔画 raw 帧→clean 网格点;c: 重建 vs 官方 lin_acc;d: 轨迹叠加)。
- 遗留:acc 去重力实现与网格锚点无法从公开 raw 精确还原;脚本给出可复现近似 + 量化验证,若后续拿到作者预处理脚本可直接替换这两步。

## 2026-10-06

### 两条工作线合并(远端 main + 本地轨迹/识别线)

- 背景:远端 `main` 有独立提交 `2f572a5`(信息"1",作者 Assass1nHeart),与本地 `trajectory-reconstruction`(`72c7abf`)从 `1e88d91` 分叉。
- 审阅对方改动(纯 touch detector 一条线):新增 `evaluate_paper_touch.py`、`models/exp_30/`、`models/exp_50/`(与基础模型同划分同口径:test user_17–20,97 样本 / 60000 窗口;30/50 epoch test acc 0.9038 / 0.9092,event macro F1 0.9235 / 0.9242),README 插入"更充分训练实验"段落,`paper_touch.py` 仅改 2 行 docstring。
- 一致性核对:本线未改 `paper_touch.py`/`paper_dataset.py`/`train_paper_touch.py`;`evaluate_paper_touch.py` 依赖的 `load_touch_model`/`evaluate`/dataset 符号全部存在;三个 checkpoint(8/30/50 ep)均可用 `load_touch_classifier` 加载并前向;`git merge-tree` 预演 0 冲突。
- 处理:合并提交 `6063edf`;整理 README(8/30/50 epoch 三个结果并列,写明同划分/同窗口上限)与 STRUCTURE(补 `evaluate_paper_touch.py`、`models/exp_*` 条目)。
- 推送:分支与 `main` 均推到 `origin-ssh`(github.com:114514yangyi/Ring);`origin` 保留 ghfast.top 只读镜像用于 fetch。

### 自采 200 Hz 数据可用性评估(2026-10-06,队友采集)

- 任务与数据:队友用实验室戒指(LPMS-B2,200 Hz)采集 `/home/huyang/data/trae_projects/raw/200hz`(4 条件 × 26 字母,1580 对 `*_imu.csv` + `*_gt_raw.csv` + `*_gt_100hz.csv`,`_meta/manifest.csv` 提供 per-trial `delta`),问"现有代码能否用于这份数据、能否给出识别结果"。
- 新脚本:`evaluate_real_data.py`(端到端评测:manifest delta 对齐 → 与训练同口径预处理 1 Hz 高通去重力 + gyro → 轨迹前端 `models_trajectory_topology/b2u` + 字母分类器 `models_character/user_both_ud` → 轨迹指标/字母 top1-top3/混淆矩阵/逐字母);`analyze_real_data.py`(质量诊断 + 4 张配图)。产物在 `outputs/real_data_eval/`(gitignore)。
- 结论 A(管线可用):格式全部兼容,1580/1580 trial 跑通、0 跳过;GT 侧字母识别 top1 **86.90%** / top3 92.59%(去掉 T/X 为 **94.17%**)——说明像素 GT → 归一化 → 分类器这条链路在该数据上成立。
- 结论 B(模型不可迁移):IMU 重建轨迹字母识别 top1 **4.24%** / top3 11.08%,预测几乎全部塌缩到同一类(混淆矩阵 I 列占绝大多数);轨迹归一化误差 **0.378**(尺度对齐后 0.252)vs 论文 0.073,路径长度比 0.701,逐点 >0.1 占 90.6%。四个条件一致(3x3 lifted/resting 4.0/4.3%,5x5 lifted/resting 4.6/4.2%)。
- 对照实验(排除"换个口径就能用"):①`--imu-source device`(设备自带 lin_acc)top1 4.49%,不变;②`--align motion`(按运动包络尖峰对齐,论文式)使 GT 侧掉到 53.0%(窗口被裁),重建仍 4.51%;③输入增益 1/2/3/5 + Procrustes 旋转尺度对齐(此前实验)top1 均 ≤7%,与最优旋转角 std 78° 说明不是简单佩戴方向问题。
- 诊断证据(IMU 侧为何失败):
  1. **幅度不匹配**:落笔时间窗内 6 通道输入 std / checkpoint 训练 std = ax/ay/az 0.12–0.14、gx 0.34、gy 0.12、gz 0.26(整段口径 0.15–0.41),即真实信号比训练分布小 3–8 倍;A/B 两类输入口径与训练管线逐行同口径(1 Hz butter 高通 + gyro ×π/180),对比有效。
  2. **对齐不可靠**:manifest delta 下 IMU 运动包络 vs 平板笔速的零延迟相关中位 **−0.155**;最佳滞后中位 +0.80 s、p90 +1.45 s,而录制余量(IMU 时长 − GT 时长)中位仅 0.50 s,**72.2% 的 trial 最佳滞后超过可用余量**(物理上不成立)→ 运动式尖峰对齐在该数据上给不出可信偏移;另有 63.3% 的 trial 运动峰值不落在落笔窗口内(300 trial 抽样),落笔段运动包络中位仅为非落笔段的 0.72 倍。
  3. **IMU 流本身健康**:时间戳单调(1580/1580)、实测 200.5 Hz、`frame_id` 零跳变(无丢帧)→ 不是录制/格式问题。
  4. **六轴自洽性差(辅助证据)**:d(acc 方向)/dt 与 −ω×acc 方向的 R² 中位仅 **0.004**(raw/10 Hz/20 Hz 低通三种口径 0.03–0.05);四元数重力方向与加速度计方向 mean|cos| 最优约定 xyzw/R = 0.74、其余约定 0.41–0.53。该项受平动加速度干扰,只作辅助指标。
  5. **书写风格差异(新发现)**:该数据集 26 个字母全部为**一笔连写**(每个 trial 恰 1 个 down 事件),T、X 一笔写出的形状与训练集多笔写法不同,导致 GT 也被判成 E(56/61)、K(39/61),两字母 GT 准确率 0/61——这是 GT 侧 86.9% 而非 94% 的全部来源,与技术链路无关。
- 【2026-10-06 晚更正:本节结论 B 已被推翻,真正根因是画布尺度污染 + 对齐判据错误 + 采集覆盖率不足,见文末「自采 200 Hz 数据重训」。】
- 结论:代码可用(读得进、跑得通、指标齐全),但模型不能直接用于这份数据;失败原因指向**设备/佩戴/书写条件造成的域偏移 + 对齐不可靠**,而非录制损坏。该结果**不作为成果汇报**,建议:①采集时增加静止段与显式同步(平板↔戒指),或改用 NTP/同源时钟;②用这批数据微调/重训轨迹前端(现有 checkpoint 直接迁移无效);③把"输入幅度比、运动-落笔一致性"做成采集质检门限;④如要复现论文量级,需按论文条件(多笔自然书写、时间戳可精确对齐)重采。

### 自采 200 Hz 数据重训:三条根因 + 端到端结果(2026-10-06 晚)

> 本节修正上一节「自采 200 Hz 数据可用性评估」的结论 B(此前判定"模型不可迁移/域偏移")。重训证据表明真正根因是:**画布尺度污染、对齐判据错误、采集覆盖率不足**;修掉前两条后同一份数据可用。

- 输入幅度不是问题:自采去重力后 std `[0.30,0.33,0.35,0.33,0.18,0.31]` vs 官方 b2u `[1.90,1.99,2.02,0.66,1.02,0.79]`。差异来自去重力算子(自采 1 Hz 零相位高通比官方彻底),量纲正确,**不改**。
- 根因 1|画布尺度污染:两组采集设备/画布不同(手机 894×390、956×370(fk)vs 平板 1080×682(yjx)),旧口径 `pixel / width,height` 使同一字母在两组间幅度差 **2.8 倍**,而 IMU 是物理单位 → 单模型无法同时拟合,测试集(全是 yjx)直接负 R²。
  修复:用"字母必须填满 3×3 / 5×5 cm 书写框"做逐设备标定,`bbox_height_px ≈ px_per_mm × box_mm` 过原点最小二乘估 `px_per_mm`;`board = pixel × mm_per_px / 240`(1.0 = 240 mm,对齐论文 Sensel Morph)。实测 **fk 0.176409 mm/px、yjx 0.250143 mm/px**(3 cm/5 cm 两组自洽)。线性探针 test R² **−1.43 → +0.47**(相关 0.65–0.70)。
- 根因 2|对齐判据必须是"笔尖速度尖峰":4 判据对照(240 trial、160/80 划分、81-lag 岭回归探针 test R²):A 仅用 run 内部速度 **−0.054**;B 全网格位置→梯度→掩码(保留落笔/抬笔尖峰)**+0.559**;C 纯矩形落笔窗 −0.040;D 同 B 但用原始像素幅度 **+0.559**。B/D 即论文所述 spike alignment,已写进 `build_lab_dataset.py`(`pen_speed_profile()` + `estimate_speed_lag()` 两级搜索:粗 0.02 s、细 0.005 s)。
  对齐质量实测:逐 trial 最优残余位移中位 ≈1 帧;全测试集最优统一位移 +0.09 s 只是把少数离群 trial 往右拉,**会伤害主体**(中位 R² 0.60 → 0.43),故不做全局位移。
- 根因 3|采集覆盖率:落笔时长中只有 **65%** 落在戒指录制窗内(p5 0.09 / p25 0.47 / p50 0.65 / p75 0.98;**≥0.95 占 27%**,**<0.5 占 29%**,n=1580)。逐字母覆盖率与识别率同向(最低 F 0.52、K 0.53、B 0.55、L 0.57、H 0.59、G 0.60)。
  另一类问题(**2026-10-07 更正**):建集后仍有 176/1516(11.6%,除 I 外)样本退化成一条直线——例如 `00012_b_fk_1` 覆盖率 0.30,构建轨迹只剩 110 点。核对原始数据后确认**这不是平板的问题**:同一样本的平板原始轨迹有 569 点、宽高比 0.46(完整字形),退化是建集时按录制窗裁剪造成的,与覆盖率同源。B 的 6 个原始真值判错样本(B→D)是完整轨迹(300–486 点),属分类器对连笔 b 的风格混淆,与左右翻转无关(全库仅 19 个样本翻转后变好,且集中在 O/X/K/I 这类近对称字母)。
- 数据集:`/data/huyang/datasets/RingLab/lab200_v4`(1516 样本、70.5 万帧、落笔 38.9 万帧;物理 mm 空间 + spike 对齐)、`lab200_v5`(覆盖率 ≥0.85 的 504 样本)、`/data/huyang/datasets/RingLab/coverage_all.csv`(1580 trial 覆盖率表)。
- 轨迹重训结果(v4 test 303 段,论文同口径逐点加权):
  | 模型 | val 归一化 | test 归一化 | test mm |
  |---|---|---|---|
  | 零基线 | — | 0.4456 | 14.52 |
  | 平均速度基线 | — | 0.3037 | 9.88 |
  | a 从零 300 ep | 0.2036 | 0.2707 | 7.01 |
  | b 从 b2u 微调 | 0.2278 | 0.2915 | 7.89 |
  | d 窗宽 25 从零 | 0.2281 | 0.2947 | 8.06 |
  | **f 微调 + 辅助损失 1.0** | **0.1917** | **0.2573** | **6.57** |
  | g 微调 + 辅助损失 10 | 0.1953 | 0.2647 | 6.46 |
  | e 条件(用户组)无重叠 | 0.3785 | 0.8037 | 19.13 |
  | v5 高覆盖子集(101 段):从零 0.219 / 微调 **0.183** | | | |
  逐 trial 中位 **0.164 / 5.37 mm**;覆盖率 ≥0.9 子集(93 段)0.195;剔除退化样本(GT 对角 <8 mm)后逐点 0.222。**mm 误差在各覆盖率分组基本不变(6.7 mm 左右)**,说明误差主要来自逐帧幅度/波形精度,而非"缺一半字母"。
- 字母识别(70/10/20,train 1061 / val 152 / test 303,与轨迹同划分):GT 输入 **76.2%** top1 / 88.8% top3(300 ep);重建轨迹端到端 **51.2%** / 74.3%(前端 f)。逐字母最弱:GT 侧 B 18%、G 39%、K/L 44%、N 45%;端到端最弱 A 33%、B 9%、C 30%、I 10%。端到端错例集中在形近对(C→D、O→S、A→G、Q→O),即**轨迹误差已小到肉眼贴合(中位 0.16),但字母身份对细节敏感**。
- 与论文/官方数据对照:论文 0.073 / 1.64 mm、字母 88.7%(Google IME);本仓库官方 clean 数据复现 0.105 / 4.15 mm、字母 86.3–93.9%。自采数据距论文 3.5×(逐点)/2.2×(中位)。瓶颈已从"模型不可迁移"变成**采集质量**(覆盖率、部分笔画丢失)与样本量(1061 vs 官方数千)。
- 下一步建议:①采集时保证录制窗覆盖整个书写(先起录、抬笔后再停);②确认平板完整记录整笔(不要只记到第一笔);③每字母每条件样本量翻倍;④把"覆盖率 + 形状质检"做成采集门限。
- 产物:`build_lab_dataset.py`(新增 `--min-coverage`、`--scale-json`、逐 trial `coverage`/`pen_seconds` 列)、`train_lab_character.py`、`visualize_lab_results.py`;模型 `models_lab/traj_{a_scratch,b_finetune,d_wide,e_crossuser,f_aux1,g_aux10,v5_scratch,v5_finetune}`、`models_character/lab_{gt,gt_long,both_a,both_f}`;图与汇总 `outputs/lab_eval/`。

### 自采数据"轨迹颠倒"排查:方向无问题,是绘图约定(2026-10-06 深夜)

- 现象: `outputs/lab_eval/gt_letters_overview.png` 等图里的字母看起来左右/上下颠倒。
- 判据 1|模型当裁判: 用官方 clean 数据训练的分类器 `models_character/user_both_ud` 直接判原始像素 GT(n=1580):
  | 输入约定 | top1 |
  |---|---|
  | 原始 `(x, y)` | **0.825** |
  | 上下翻转 `(x, -y)` | 0.119 |
  | 左右翻转 `(-x, y)` | 0.066 |
  | 180° `(-x, -y)` | 0.051 |
  → 存储方向与官方 clean 完全一致,任何翻转都会毁掉精度,数据本身没有颠倒。
- 判据 2|坐标系: 平板/浏览器是 CSS 像素(origin 左上、y 向下),matplotlib 默认 y 向上,因此绘图必须 `axis.invert_yaxis()`;官方代码(`visualize_results.py` 326/676/909/1046、`train_paper_trajectory.py` 323)一直这么做,本轮新写的 `visualize_gt_letters.py` / `visualize_lab_results.py` 漏了,已补并重出 `gt_letters_overview/gallery.png`、`trajectory_examples.png`、`recognition_examples.png`。
- 证据图: `outputs/lab_eval/orientation_explained.png`(同批轨迹两种画法对照)、`orientation_conventions.png`(同一 trial 四种约定,橙色=落笔点)、可复跑脚本 `outputs/lab_eval/orientation_check.py`。
- 顺带完成的体检(相比官方 clean 的真正差异):
  1. 覆盖率(最严重): p5/p25/p50/p75/p95 = 0.09/0.47/0.65/0.98/1.00,≥0.95 仅 27%、<0.5 占 29%;把被录制窗截断的 v4 真值喂官方分类器:n=1516 整体 0.465,**覆盖率 ≥0.9 组 0.822(n=460)**、0.6–0.9 组 0.521(n=447)、**<0.6 组 0.154(n=609)**;未截断的原始真值 0.825。→ 低覆盖样本"标签 vs 可用信号"基本不对应。
  2. 书写风格/笔顺: 1580/1580 恰 1 个 down 事件(全一笔连写),官方每字符平均 1.55 个接触段;一笔写的 T/X 被判成 E / Y·K,GT 侧 T 0%、X 0%、F 9%、H 48%、K 55%,其余 20 字母 ≥95%。
  3. 单位/画布: CSS 像素非物理单位,fk 0.176409 / yjx 0.250143 mm/px;fk 出现 3 种画布(894×390/956×370/956×390),但同一字母像素尺寸跨画布一致(O 160×181 vs 171×180 px),说明"每设备一个 mm/px"的标定安全;官方 240×169.5 mm 各向异性,y 拉伸 1.416 实测只差 0.8 个点,可忽略。
  4. 时钟: per-trial `delta` 跨 −856 s ~ −46460 s(≈12.9 h),必须逐 trial 用;`received_at` 抖动 std 中位 6.6 ms(最大 9.6 ms),只能粗对齐,不可替代设备时间戳。
  5. 分布不均: fk 1040(4 条件)/ yjx 540(仅两个 3×3 条件),第二人无 5×5。
- 通过项(GT 内部自洽): 419,919 步里 |步长 − dx/dy| 仅 1 处不一致(l_yjx_1 第 2 点 6.02 px),最大偏差 0.0000;7 处首点时间戳重复(非问题);IMU 1580/1580 无丢帧(frame_id 步进恒 2)、时间戳单调、中位 199.80 Hz、无缺失文件。

### 全样本铺开检查(2026-10-07)

- 新脚本 `visualize_all_samples.py`,产物在 `outputs/lab_eval/`:`all_samples_overview.png`(1580 个样本一张图,每行一个字母,按覆盖率排序,绿/黄/红=录全/缺一点/缺一半以上)、`all_samples/<字母>.png`(26 张逐字母样本表,每格标注 trial + 覆盖率 + 平板记录点数)、`coverage_bands_per_letter.png`(逐字母堆叠占比)、`all_samples_summary.csv`。
- 覆盖率分档(n=1580):录全(≥0.9)**464(29%)**、缺一点(0.6–0.9)447(28%)、缺一半以上(<0.6)**669(42%)**。
- 最差字母(录全占比):B 11.7%、K 11.7%、M 18.3%、F 21.2%、G/H 21.3%、L 21.7%;最好:Z 48.3%、V 45.0%、X 42.6%、Y 41.0%、R 40.3%。
- 注意:覆盖率表 `coverage_all.csv` 的 `trial` 字段**在不同条件下会重名**(1580 行只有 539 个唯一名),合并必须用 `(condition, trial)`;按 trial 名单独合并会把 B/K 这类最差的字母算成 29%,是错的。此前 v4 分档(≥0.9 → 0.822,n=460)用 (condition, trial) 复核后不变。

### 重训能否解决问题?——覆盖率是天花板,换训练数据不是(2026-10-07)

- 问题:用户问"能不能在新的数据上重新训练去解决(覆盖不足 / 风格差异)"。
- 实验(脚本 `/tmp/labtest/clean_retrain.py`,报告 `models_character/lab_v4clean_gt/character_report.json`):
  在 lab200_v4 的同一 split(seed 42)上,把训练集按覆盖率 ≥0.85 过滤(1061 → **349** 条)重训一版分类器,
  与现有全量训练的 `models_character/lab_gt_long`(1061 条)对照,在**同一批测试样本**上比较:
  | 测试集 | 全量训练 lab_gt_long | 只用干净样本重训(349 条) |
  |---|---|---|
  | 录全样本(test∩cov≥0.85,n=100) | **99.0%** | 97.0% |
  | 全部 test(n=303) | **76.2%** | 52.8% |
- 结论 1|**覆盖率不是模型的问题**:在录全的 100 个测试样本上,现有模型已经 99.0%,此前最弱的 B/K/G/L/N 全部 100%。
  同一模型在全量 test 上只有 76.2%,差额全部来自被录制窗截断的样本。
- 结论 2|**只留干净样本重训没有收益**(349 vs 1061 条):干净测试集 97% vs 99%(差异在噪声内),全量测试集 53% vs 76%(训练数据少了 2/3)。
  → 正确做法是**补采**(把覆盖率提上去 + 增加样本量),在"新旧合并"的数据上重训;不要为了干净而丢掉样本。
- 结论 3|**风格问题重训已经解决**:官方 clean 模型在自采 GT 上 T 0% / X 0% / F 9% / H 48% / K 55%;
  用自采数据重训后(lab_gt_long)变成 T 86% / X 100% / F 86% / H 57% / K 44%。
- 结论 4|逐字母"录全占比"与"重训后识别率"相关 **r=0.674**,覆盖率是识别率的主要瓶颈。
- 仍存在的真实差距在**轨迹重建**:录全子集 0.195 vs 论文 0.073(mm 误差 6.7 vs 1.64),这部分不能只靠覆盖率解释,需要更多数据与设备/佩戴标定。

### 数据缺陷报告 PDF(2026-10-07)

- 产物:`outputs/lab_eval/RingLab_200Hz_dataset_report.pdf`(A4 12 页,中文,封面预览 `RingLab_200Hz_report_cover.png`),生成脚本 `make_dataset_report.py`(matplotlib PDF 后端,无需 LaTeX/浏览器;`--preview` 可导出逐页 PNG)。
- 结构:封面与结论先行 → 一、数据构成与评测口径 → 二、缺陷总览(严重度排序表) → 三、覆盖率不足(含逐字母分档图、全样本总览、覆盖率→识别率因果表) → 四、建集退化样本(新增图 `build_truncation_B.png`:同一批 B 样本平板原始 vs 建集裁剪后) → 五、书写风格/笔顺 → 六、坐标系与设备标定(含方向澄清) → 七、时钟偏移 → 八、采集分布不均 → 九、通过项 → 十、重训实验 → 十一、行动建议 → 附录 A/B(文件脚本清单、复现命令)。
- 新增配图 `outputs/lab_eval/build_truncation_B.png`,由 /tmp/labtest/trunc_fig.py 生成。
- 报告口径:缺陷 2 已按 2026-10-07 的核对结果改写为"建集裁剪造成,非平板漏记";重训实验结论(99.0% / 97.0% / 76.2% / 52.8%)一并写入第十章。

### 对齐验证:覆盖率低不是时间戳错位造成的(2026-10-07 晚)

用户三问:数据是不是 clean?要不要按官方 `build_clean_dataset.py` 那套处理?覆盖率低会不会是时间戳对不上?

- **结论 1|不是 clean**:按 `coverage_all.csv`(尖峰对齐口径)覆盖率中位 **0.65**,录全(≥0.9)只有 **29.4%**、缺一半以上(<0.6)**42.3%**(n=1580)。官方 clean(566 样本)是"录制窗比书写长"的产物,我们这批是"书写比录制窗长"。
  反推的官方流程(`build_clean_dataset.py` docstring 8 条):裁 `[board_start+0.6s, board_end−1.0s]` → 4977.75 µs 均匀网格 → gyro 原样搬运 → 加速度 ~1 Hz 高通去重力 → board 重采样 → mask 由接触帧夹逼 → y = 帧率位移。**直接套到自采数据上只会再切掉 1.6 s**,必须重采(先起录 0.5 s、抬笔后多录 ≥1 s)才谈得上 clean。
- **结论 2|时间戳错位不是根因,但决定"看起来好不好"**:只用 manifest `delta`(纯钟差)算覆盖率 → ≥0.9 占 **93.0%**、<0.6 占 **0%**(假象);换成构建用的尖峰对齐 → ≥0.9 掉到 29.4%。**63.5% 的 trial 因尖峰对齐把覆盖率下调 >0.2**(修正量中位 **0.95 s**、p90 1.64 s、max 2.51 s),即 IMU 上电 t=0 相对录制启动的抖动。
- **三判据(脚本 `/tmp/labtest/{physical_align,probe_align,coverage_truth,align_verify_fig}.py`,图 `outputs/lab_eval/alignment_verification.png`、`alignment_three_ways.png`)**:
  1. 落笔冲击:戒指运动最陡上升点与落笔瞬间 |Δt| 中位 **0.03 s**(69% ≤0.3 s);manifest delta 口径中位 **1.11 s**(0% ≤0.3 s)。
  2. 线性探针(6 通道窗特征 → 2D 笔尖位移,ridge,360 trial):尖峰对齐 test **R²=0.44**;manifest delta **−0.02**(全局时移扫描在 lag=0 附近全负,修不回来)。
  3. 截断方向(尖峰口径):落笔起点 − 录制起点中位 **+0.94 s**(只有 2% 早于录制起点);落笔终点 − 录制终点中位 **+0.55 s**(**75% 超出录制终点**)。运动段"贴着窗口尾巴"的 trial 占 **86%**、贴着头占 41%。低覆盖样本 669 个里 **653 个(97.6%)** 属"运动被窗口切断/运动段远短于落笔时长",仅 16 个存疑。
  → **截断几乎全在尾巴**(抬笔后停录太早),开头基本没截。
- **结论 3|`received_at` 与"14 分钟偏差"**:`received_at` 抖动 std 中位 6.6 ms(最大 9.6 ms),只能做粗对齐;per-trial `delta` 跨 −856 s ~ −46460 s 必须逐 trial 用,大偏差本身不影响结果(它是常数偏移,尖峰对齐会把它吸收掉),影响结果的是 IMU t=0 相对录制启动的 **1 s 量级抖动**,这一项必须用数据驱动(尖峰)对齐,不能用钟差。
- 采集规则(直接给队友):**起录后停 0.5 s 再落笔,抬笔后继续录 ≥1 s**;保留静止段用于对齐与质检。

### PPT 增补:新数据集结果与差异说明(2026-10-07)

- 目标:在组会结果版 deck(原 7 页)上加入自采 200 Hz 数据的结果与差异说明,措辞不涉及对采集工作的评价。
- 做法:按 ppt-master 流程在 `~/.agents/projects/writingring_results_20261005` 手工新增两页 SVG(`08_lab_data.svg`、`09_lab_notes.svg`),同步 `spec_lock.md`(images / page_rhythm / core_message)与 `design_spec.md`(§VIII 图片表、§IX Slide 08/09),全部原有页脚 `xx / 07` → `xx / 09`;校验器 9 页 0 error,导出 9 页 PPTX。
- 页面内容:P08 = 新设备数据结果(轨迹 0.257 / 6.57 mm、字母 GT 76.2% / 端到端 51.2%、录全样本 99.0%);P09 = 差异来源(录制窗口起止设置、一笔连写 vs 官方分笔、样本量与字母间均衡)+ 采集侧下一步。
- 产物:`outputs/writingring_results_20261005/`(pptx / preview.pdf / deck_overview.png / 讲稿 md+docx+pdf / speaker-notes.md,旧 7 页版在 `backup/`)。
- 讲稿第三版:新增第 8、9 页逐页讲稿与"五、新设备数据指标"(覆盖率定义与计算、录全子集、尖峰对齐、自采随机划分与官方严格口径不可直接比较),附录 A 增补 3 行。

### 未录全样本清单 PDF(2026-10-07)

- 需求:用户要"所有没有录全的样本"。
- 口径:覆盖率 < 0.90 → **1116 个**(71%);其中 <0.60 有 669(60%)、0.60–0.90 有 447(40%);未录全样本覆盖率中位 0.55。
- 产物:`outputs/lab_eval/not_full_samples.pdf`(A4 横向 58 页:封面 + 逐字母汇总表 + 56 页图册,每页 5×4=20 个小格),小格内深蓝 = 窗内(模型看到的)、红虚线 = 窗外(没录到)、橙点 = 落笔点;标题给出字母 · 样本名 · 条件 · 覆盖率(书写时长 → 录到时长)。排列按字母 A–Z、组内覆盖率升序。
- 生成脚本 `/tmp/evidence/build_pdf.py`(读 manifest + `coverage_all.csv` 的尖峰对齐 lag,复算每个 trial 的窗内/窗外笔画);逐样本数据 `outputs/lab_eval/truncation_evidence.csv`(1580 行);预览图 `not_full_samples_cover.png`、`not_full_samples_table.png`、`not_full_samples_page-3.png`。
- 注意:严格口径是 <0.90,但显示两位小数时 0.897 会显示成 0.90。

### 对齐口径复核(用户质疑"窗口没对齐"时的再验证,2026-10-07 晚)

- 背景:覆盖率"仅 27% 录全"这一结论遭到质疑,怀疑是平板轨迹与 IMU 时间没对齐。复核结论:**对齐口径是对的,不是对齐问题**。
- 新证据(脚本 `/tmp/align3/{probe_sweep,probe_refine,trunc_side}.py`,图 `outputs/lab_eval/alignment_probe_sweep.png`、`coverage_verified.png`、`align_window_vs_pen.png`,数据 `outputs/lab_eval/alignment_probe_sweep.json`):
  1. **口径扫描**(400 trial、留出 1/3,30 维 IMU 窗特征 → 笔尖速度 ridge):`offset = 钟差 + delta − alpha×lag` 中 alpha=0(纯钟差)→ R² **0.01**;alpha=1(当前口径)→ R² **0.47**;alpha=1.25/1.5 只剩 0.08/0.06。**只有当前口径能把环信号映射到笔尖运动。**
  2. **全局平移扫描**:在当前口径上再加常数 c,峰值精确落在 **c=0**;c=±0.1 s 即掉到 0.32/0.33,c=±0.2~0.4 s 掉到 0.10~0.13。→ 当前对齐的残余误差 < 0.1 s。
  3. **"把窗口往安静段平移换覆盖率"是几何假象**:覆盖率高(0.99)的摆法对应 R²=0.01,即那段时间环信号与笔尖运动无关;覆盖率必须与探针一起看。
  4. 截断方向复核(alpha=1):落笔点距窗口起点中位 **+0.94 s**、**73.9%** 的样本结尾被切(平均丢 0.58 s)、仅 **1.1%** 开头被切;`W−P` 中位 **+0.49 s**(窗口总时长够长,是整体偏早)。
- **manifest `delta` 的真实含义**(此前未记录):实测 `delta = 第一个 GT 触点时刻 − 首个 IMU 帧 received_at`,`max|误差| = 0.5 ms`(n=1580)。即 delta 把"落笔点"钉在录制窗口起点上,它同时吸收了钟差与内容偏移;尖峰 lag 再把内容偏移去掉。同一 clock_state 内 delta 跨度 1.1–2.2 s,而平板 `performance.timeOrigin` 在页面加载后锁定,钟差在同一次会话内必须是常数 → **这 1–2 s 的簇内离散就是内容偏移,不是"钟漂"**。
- 结论口径:对队友表述为"录制起止时机整体偏早约 0.75 s(早开录 0.94 s、早停录 0.58 s)",不涉及对齐/代码问题。

### 重采数据(new 批次)核查与建集(2026-10-09)

- 数据:`/data/huyang/trae_projects/new`(队友重采,**550 trial** = fk 260 + yjx 290,26 字母目录,无 condition 子目录)。每 trial 三件套 `_imu.csv` / `_gt_raw.csv` / `_gt_100hz.csv`;IMU 时长中位 **4.38 s**(旧批 2.16 s),GT 时长中位 2.29 s。
- 新增列:`mask`(算法一,lin_acc 最大尖峰)、`mask_v2`(算法二,陀螺仪模长 vs 平板笔速互相关 + 三条约束)、`mask_v3`(按位置先验在两者中二选一);另有 `sensor_to_host_offset`(逐 trial 常数,`received_at = timestamp + 该值`)。说明见 `new/README.md`,复核清单 `new/mask_v3_review_list.csv`(8 条)。
- **格式坑**:新 CSV 带 UTF-8 BOM,`evaluate_real_data._read_csv` 默认 utf-8 会让首列名变成 `\ufefftimestamp` → 已把默认编码改为 `utf-8-sig`(对无 BOM 文件无影响)。
- **体检结论(脚本 `verify_lab_alignment.py`,产物 `outputs/lab_eval/new_data_check/`)**:
  - 用 `mask_v3` 反推:覆盖率中位 **0.997**、**100% 的样本 ≥0.95**、`W<P` 为 **0**;接触段位置 start 中位 **0.19**、end 中位 **0.73**(与采集方 19%~73% 的先验一致)。**旧批"只有 27% 录全"的问题已解决。**
  - 两种独立对齐互证:我方尖峰对齐 vs `mask_v3` 差值中位 **−0.006 s**、`|差|` 中位 **0.012 s**、>0.15 s 仅 **8.2%**;`mask_v2` 同量级;`mask`(峰值法)离散更大(19.3%)。
  - 我方独立估计在 45 条上与 `mask_v3` 差 >0.15 s,且都撞到 `search=(-1.0,2.5)` 的边界 → 这批以 **`mask_v3` 为锚**更稳。
- **尺度**:新批画布统一 **1080×682**(旧批 fk 是手机 894/956 宽)。新 yjx bbox 中位 124 px ≈ 旧平板 yjx 122 px → 沿用平板标定 **0.250143 mm/px**(=4.0 px/mm,与旧批 3cm/5cm 双框拟合一致)对两个来源统一使用;新 fk 写得更大约 1.3 倍(162 vs 124 px),属书写习惯差异而非画布差异。注意 fk 由此与旧批(手机 0.176409)不同尺度,**旧模型不可直接迁移**。
- 代码改动:`build_lab_dataset.py` 新增 `--align tablet_mask`(用 `new/` 的 mask 列定位落笔点 + ±0.2 s 尖峰精修)、`USER_GROUPS` 增加 `("fk","new")→user_6`/`("yjx","new")→user_7`、`estimate_device_scales` 跳过无书写框条件;`evaluate_real_data.read_imu_masks()` 读 mask 列;新 manifest 落在 `new/_meta/manifest.csv`(condition='new')。
- 数据集:`/data/huyang/datasets/RingLab/lab200_new_v1`(**550 样本**、48.96 万帧、落笔 25.76 万帧;覆盖率中位 1.000、≥0.95 占 100%;精修 lag 中位 +0.010 s)。
- 待办:轨迹重训(`models_lab_new/traj_r_aux1`,随机划分 seed42)+ 字母分类器重训;`new_manifest.csv` 的 `split` 列为空,需自行划分。

#### 重采批次训练结果(2026-10-09)

- 配置与旧批口径一致:随机划分(seed 42,70/10/20)、dilated TCN 32/64/128 + 2 层 BiLSTM、13 帧窗、aux 1.0、300 epoch、lr 3e-4 cosine、segment 8 s(新批最长 7.5 s,不会再被中心裁掉)。
- **轨迹** `models_lab_new/traj_r_aux1`:val 最优 0.1646(ep280);test **0.1798 / 6.24 mm**(110 段)。对比旧批 v4 的 0.2573 / 6.57 mm,**归一化误差 −30%**。论文 0.073 / 1.64 mm,官方 clean 复现 0.105 / 4.15 mm。
- **字母**(`models_character_new/`):GT 输入 `lab_gt_long`(300ep,lr 1e-3)test **top1 100% / top3 100%**(n=110,26 类全对);GT+重建混训 `lab_both`(150ep)GT **100%**、**端到端重建 top1 89.09% / top3 93.64%**。
- 对比:旧批 GT 76.2% / 端到端 51.2%;论文 Google IME 字母 88.7%。→ **端到端已略高于论文 IME 口径**,但注意本批是"同人随机划分",且只有 2 位书写者(fk 260 / yjx 290),严格跨人结论仍需 fk→yjx 留一用户实验。
- 逐字母端到端最低:J 43%(n=7)、D 67%(6)、F 67%(3)、H 67%(3)、K 75%(4)、N 75%(4)、P 83%(6)、C 86%(7);其余 18 个字母 100%。
- 产物:数据集 `/data/huyang/datasets/RingLab/lab200_new_v1`;图 `outputs/lab_eval_new/`(trajectory_examples / letter_accuracy / recognition_examples / confusion_recon / error_vs_coverage)。
- 未做:单词识别(新批只有单字母)、覆盖率高不再需要 v5 高覆盖子集(≥0.95 已占 100%)。

### SmartRing 数据运行记录(2026-10-10)

- 用户提供原始数据: `/data/fan/SmartRing/data/raw/new`。其内容与重采批次格式一致(26 个字母目录、550 个 trial、三件套 CSV)，但清单位于根目录 `new_manifest.csv`，没有 `_meta/manifest.csv`；本次仅建立兼容软链 `_meta/manifest.csv -> ../new_manifest.csv`，未修改 CSV 内容。
- 使用环境: `/data/fan/conda/writingring-gpu`，PyTorch 2.10.0+cu128，RTX 4090。使用既有设备标定 fk=0.176409、yjx=0.250143 mm/px，并以 `--align tablet_mask` 建集。
- 建集产物: `/data/fan/SmartRing/data/processed/lab200_new_20261010`；550/550 样本成功，489649 帧、257648 接触帧，user_6=260、user_7=290，无跳过项。
- 轨迹训练产物: `outputs/run_smartring_new_20261010/traj_r_aux1`；随机 seed 42、70/10/20、dilated TCN 32/64/128 + 2 层 BiLSTM、aux=1、300 epoch、lr 3e-4 cosine。test=110 段，归一化误差 **0.12153**，mm 误差 **3.6200**。
- 字母训练产物: `outputs/run_smartring_new_20261010/character_both`；GT+重建混训 150 epoch。test=110，GT top1/top3 **100%/100%**，重建端到端 **92.73%/97.27%**。
- 本批仍是两位书写者的随机划分，不等同于严格跨用户泛化；数据集只有单字母样本，因此未运行单词识别。
