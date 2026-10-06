# Writing State Detector

这个目录负责书写状态分段，不负责 a-z 字符分类。当前包含两个路径：

- `detector.py`：原先的硬阈值 + 状态机 baseline。
- `paper_touch.py`：WritingRing 风格的学习模型复现骨架，不使用固定阈值。

## WritingRing 复现路径

论文中的 touch detector 使用：

- `0.1 s` 固定窗口；
- 每个窗口 `20 x 6`，即 20 帧线性加速度和角速度；
- 四分类：`contact`、`air`、`lift`、`press`；
- 三个 1-D ResNet 残差块，输出通道为 `8 / 16 / 32`；
- 每个卷积后使用 BatchNorm 和 Dropout；
- 只把 `press` 和 `lift` 解码成书写起止事件。

代码入口：

```powershell
python -m writing_state.paper_touch --json
```

当前已经基于 `clean_data_delete_g` 训练出 touch detector：

```text
模型: writing_state/models/paper_touch_resnet.pt
报告: writing_state/models/paper_touch_report.json
测试集: user_17, user_18, user_19, user_20
test accuracy: 0.8829
test macro F1: 0.8832
test press/lift event F1: 0.9072
```

```python
from writing_state.paper_touch import (
    PaperTouchConfig,
    PaperTouchDetector,
    load_touch_classifier,
)

config = PaperTouchConfig()
classifier = load_touch_classifier("writing_state/models/paper_touch_resnet.pt", config=config)
detector = PaperTouchDetector(classifier, config=config)
```

训练命令：

```powershell
python -m writing_state.train_paper_touch `
  --epochs 8 `
  --batch-size 512 `
  --max-train-windows-per-class 30000 `
  --max-eval-windows-per-class 15000 `
  --output-dir writing_state\models
```

建议先做一个更充分的训练实验，把每次实验输出到独立目录：

```powershell
python -m writing_state.train_paper_touch `
  --epochs 50 `
  --batch-size 512 `
  --max-train-windows-per-class 30000 `
  --max-eval-windows-per-class 15000 `
  --output-dir writing_state\models\exp_50
```

如果没有 CUDA，可以显式指定 CPU：

```powershell
python -m writing_state.train_paper_touch `
  --epochs 50 `
  --device cpu `
  --output-dir writing_state\models\exp_50
```

训练完成后独立测试：

```powershell
python -m writing_state.evaluate_paper_touch `
  --checkpoint writing_state\models\exp_50\paper_touch_resnet.pt `
  --split train `
  --max-windows-per-class 15000

python -m writing_state.evaluate_paper_touch `
  --checkpoint writing_state\models\exp_50\paper_touch_resnet.pt `
  --split val `
  --max-windows-per-class 15000

python -m writing_state.evaluate_paper_touch `
  --checkpoint writing_state\models\exp_50\paper_touch_resnet.pt `
  --split test `
  --max-windows-per-class 15000 `
  --report-out writing_state\models\exp_50\test_report.json
```

`--max-windows-per-class 0` 会评估指定集合中的全部窗口。比较不同训练轮数时，
请保持数据划分、窗口上限和随机种子一致；最终模型应以 `test` 的
`event_f1`、尤其是 `lift` 和 `press` 的 F1 为主要参考。

模型输出的时间窗口标签为：

- `writing_start`：指尖开始接触书写平面，开始一个有效书写段。
- `pen_up`：从有效书写段离开书写平面。它同时结束当前有效段。
- `invalid_air_start`：可选的、在还没有有效接触时检测到的空中运动。

`pen_up` 之后，到下一次 `writing_start` 之前的动作不参与字符识别。这样同一个状态机既能识别首字符起笔/末字符收笔，也能识别两字符之间的抬笔。

## 当前实现

### 规则 baseline

`WritingStateDetector` 是一个可实时运行的规则基线：

1. 从 `lin_acc`、`gyro`、加加速度和角速度变化提取低维特征。
2. 使用接触/抬笔冲击和持续运动证据触发候选状态。
3. 用连续帧确认时间和滞回避免单点噪声误报。
4. 通过 `valid_operation` 明确把空中段标成无效。

它只用于当前阶段的回归测试和无模型兜底，不代表论文中的最终 touch detector。

## 轨迹重建(TCN+LSTM)

复现论文的指尖速度预测与接触段轨迹重建(轨迹→字符识别为后续任务)。方法要点:

- 输入为去重力 acc + gyro(官方 `clean_data_delete_g` 的 acc 已去重力);
- 13 帧滑窗经 6 层 valid TCN(kernel=3,通道 16/16/32/32/64/128)压成 1×128 特征,LSTM(hidden=128)跨窗流式预测窗口中点帧位移 (Δx, Δy);预测使用 6 帧未来数据,延迟 32.5 ms;
- 训练用 75 s(15000 帧)长片段:不足头部补零、超长对称裁剪;仅接触帧(mask=1)计入 MSE;TBPTT 分块回传(默认块长 3000);
- 推理只在接触段积分预测位移;评估为论文归一化逐点距离(除以该段 GT bbox 对角线)与无归一化 mm 误差。

```powershell
python -m writing_state.train_paper_trajectory `
  --data-root /data/huyang/datasets/WritingRing/clean_data_delete_g/data `
  --split random --epochs 500 --batch-size 32 --chunk-len 3000 `
  --output-dir writing_state\models_trajectory
```

`--split user` 为用户无重叠严格对照;`--long-sample-strategy chunks` 可切换等长切块。board 物理单位标定见 `board_scale.py`(实测原始 Sensel 触点也是归一化坐标,默认按 Sensel Morph 240×169.5 mm 假设换算 mm)。

当前复现结果(500 epochs,随机 70/10/20;board 真值评估,test):

```text
模型: writing_state/models_trajectory/paper_trajectory.pt
报告: writing_state/models_trajectory/paper_trajectory_report.json
随机 70/10/20,500 epochs(dilated TCN+2 层 LSTM):
                        归一化误差均值 0.124(论文 0.073),误差 >0.1 的逐点占比 46.3%,
                        无归一化误差 5.08 mm(论文 1.64 mm)
离线双向 X3(2层双向 LSTM,B2 精调 450ep): 0.105 / 4.15 mm(单模最佳,模型 report 口径)
v1+x3+b2+v4 位移平均 ensemble(免训练):    0.1012 / 4.05 mm(最佳总体,可视化 entry 复算口径)
用户无重叠 严格口径(旧架构,500 epochs):  0.214 / 8.53 mm
用户无重叠 严格口径(当前架构,b2u,300 epochs,user_0-13 训练 / 16-20 测试):
                                        0.159 / 5.54 mm(129 样本,11070 段;逐点误差 >0.1 占 51.4%,p50 0.158)
```

最佳总体结果距论文仍有约 1.4x 差距;官方 `y` 的协议噪声底为 0.038 / 1.28 mm,论文已接近该底。训练方式/拓扑/双向/ensemble 的完整实验矩阵与归因见 `.trae/documents/NOTES.md`。归档中另有一组 x1+x3+b2+v4 = 0.1003 / 4.01 mm(`models_trajectory_topology/x1`),但 `visualize_results.py` 的 ensemble 成员为 v1/x3/b2/v4,页面的 0.1012 / 4.05 mm 按该口径给出,两者均低于 0.105。

## 轨迹→字母识别(任务 2 第一阶段)

用逐字符标签切分轨迹并训练 26 类字母分类器(替代论文的 Google IME),同时支持 touchpad 真值(GT)与轨迹重建输出(端到端)。

```powershell
python -m writing_state.train_character_classifier `
  --data-root /data/huyang/datasets/WritingRing/clean_data_delete_g/data `
  --raw-root /data/huyang/datasets/WritingRing/data `
  --split user --source both --epochs 60 `
  --output-dir writing_state\models_character\user_both
```

严格口径端到端重跑(重建输入来自用户无重叠前端 b2u,前端训练集不含 user_16-20):

```powershell
python -m writing_state.train_character_classifier `
  --data-root /data/huyang/datasets/WritingRing/clean_data_delete_g/data `
  --raw-root /data/huyang/datasets/WritingRing/data `
  --split user --source both --epochs 60 `
  --reconstruction-checkpoint writing_state\models_trajectory_topology\b2u\paper_trajectory.pt `
  --output-dir writing_state\models_character\user_both_ud
```

用户无重叠测试(user_16/18/19/20;user_17 无合格字母样本):

```text
GT 轨迹:                    top1 93.9% / top3 98.9%
重建轨迹(严格口径,前端未见): top1 86.3% / top3 96.1%(n=708,全测试集)
重建轨迹(上界口径,前端 84% 测试样本见过): top1 92.7% / top3 98.9%
```

口径说明:字母分类器本身与测试用户无重叠。上界口径的重建输入来自随机划分训练的轨迹前端 x3(其训练集含约 84% 的字母测试样本);严格口径改为用户无重叠前端 b2u(训练集 user_0-13,测试 user_16-20),字母测试集完整覆盖 708 个样本,不再需要子集估计。两者相差 6.4 pt,即前端未见测试用户带来的真实代价。

论文 Google IME 实时字母识别 88.7%(口径不同,仅量级对照):严格口径 86.3% 略低于该值,上界口径 92.7% 高于该值。

## 轨迹→单词识别(任务 2 第二阶段基线)

按标签时间区间切分整词轨迹(重采样 128 点、bbox 归一化),用"最近训练实例"做词表匹配(标准欧氏;DTW 重排与 deskew 实测无提升)。

```powershell
python -m writing_state.evaluate_word_dtw `
  --data-root /data/huyang/datasets/WritingRing/clean_data_delete_g/data `
  --raw-root /data/huyang/datasets/WritingRing/data `
  --split user --source both --metric dtw --deskew `
  --output-dir writing_state\models_words\user
```

用户无重叠(user_17-20;test n=1306,闭集 1084;训练词集 416 词):

```text
GT 轨迹:   闭集 top1 63.2%(全体口径 connected 44.5% / unconnected 61.0%)
重建轨迹:  闭集 top1 61.4%
```

注:以上为 `--metric dtw --deskew` 的落盘报告(`models_words/user/word_report.json`);此前文档中的 63.6 / 63.5 来自另一次 euclidean 运行,与磁盘产物不一致,现已按落盘报告更正。

### CTC 词识别(专门识别器)

CNN + 双向 LSTM + CTC 预测字母序列,再用编辑距离对齐词表(词表=训练词集):

```powershell
python -m writing_state.train_word_ctc `
  --data-root /data/huyang/datasets/WritingRing/clean_data_delete_g/data `
  --raw-root /data/huyang/datasets/WritingRing/data `
  --split user --source both --epochs 300 `
  --output-dir writing_state\models_words\ctc_user_both
```

严格口径端到端重跑(重建输入来自用户无重叠前端 b2u):

```powershell
python -m writing_state.train_word_ctc `
  --data-root /data/huyang/datasets/WritingRing/clean_data_delete_g/data `
  --raw-root /data/huyang/datasets/WritingRing/data `
  --split user --source both --epochs 300 `
  --reconstruction-checkpoint writing_state\models_trajectory_topology\b2u\paper_trajectory.pt `
  --output-dir writing_state\models_words\ctc_user_both_ud
```

用户无重叠结果(test n=1306):

```text
严格口径(前端 b2u 未见 user_17-20;test n=1306,全测试集):
  重建子集: raw top1 58.6% | 词表对齐 top1 64.2%(connected 56.5% / unconnected 72.5%)
  GT 子集:  raw top1 63.9% | 词表对齐 top1 66.9%(connected 59.4% / unconnected 75.0%)
  合并口径(n=2612): raw 61.3% | 对齐 65.5%
上界口径(前端 x3 见过 69% 测试词):
  重建子集: raw 60.6% | 对齐 65.7%(connected 57.5% / unconnected 74.5%);合并 66.2%
GT-only:     raw top1 54.9% | 词表对齐 top1 62.3%
随机划分:     raw top1 68.9% | 词表对齐 top1 81.8%
最近邻基线:  词表对齐 63.2%(闭集,GT 轨迹;connected 44.5% / unconnected 61.0%)
```

口径说明:严格口径下前端 b2u 训练集为 user_0-13,测试用户 17-20 完全未参与前端训练;重建子集词表对齐 64.2% 与上界 65.7% 仅差 1.5 pt——词级任务对重建误差不敏感(与最近邻基线观察一致),而字母任务相差 6.4 pt,说明短笔画更依赖前端精度。

论文 Google IME:unconnected 68.2 / connected 53.1(其 3000 词表下 84.4 / 74.2);口径不同,仅量级对照——严格口径 72.5 / 56.5 已超过论文开放词表 unconnected 与 connected,但仍低于其 3000 词表 + 语言模型。细节见 `.trae/documents/NOTES.md`。

## 结果可视化

一条命令重跑测试集推理并生成全部结果图(轨迹对比、误差统计、字母混淆矩阵与识别样例、词识别对照与样例):

```powershell
python -m writing_state.visualize_results --figures trajectory,character,word
```

严格口径的图与逐字母表输出到 `outputs/figures_strict/`(加 `--protocol strict` 即可复现,使用 b2u 前端与 `user_both_ud` 下游模型);
上界口径(论文随机划分前端)输出到 `outputs/figures/`:

```text
fig_trajectory_examples.png   测试段 GT vs 重建轨迹(按误差分位取样)
fig_trajectory_quality.png    逐段误差直方图 / 路径长度保真 / 误差 CDF / 与论文和噪声底对比
fig_character_recognition.png 26x26 混淆矩阵 + 逐类准确率 + 汇总
fig_character_examples.png    字母轨迹与识别结果(对/错样例)
fig_word_recognition.png      词识别与最近邻基线、论文数值对照
fig_word_examples.png         词轨迹与 CTC 解码文本
fig_word_examples_gt_vs_pred.png  8 个测试词卡片:GT 轨迹 | 重建轨迹 | 真实词 | 识别词
fig_letter_accuracies.png     26 字母逐类识别率(GT vs 重建,柱上标数值)
letter_accuracy.csv           26 字母逐类 top1/top3 明细
metrics_summary.json          本次复算的全部指标
```

`--figures` 可选 `trajectory,character,word` 子集;`--device` 可指定 `cpu`/`cuda`。

## CSV 回放

支持 SmartRing 记录格式：

```powershell
python -m writing_state.cli data\without-tablet\Up\2\imu_data_up2.csv `
  --events-out outputs\writing_events.csv `
  --segments-out outputs\writing_segments.csv
```

也支持 OpenZen 常见列名，例如 `TimeStamp (s)`、`LinAccX (g)`、`GyroX (deg/s)`。

## 实时接入

```python
from writing_state.realtime import RealtimeWritingStateHandler

writing_handler = RealtimeWritingStateHandler()

def on_raw(frame):
    writing_handler.on_raw(frame, bus)

bus.subscribe("raw", on_raw)
```

它向 `writing_state` topic 发布：

```json
{
  "type": "pen_up",
  "timestamp": 12.34,
  "state": "air_invalid",
  "confidence": 1.0,
  "valid_operation": false,
  "reason": "pen_up_confirmed",
  "segment_id": 2
}
```

同时向 `writing_state_frame` topic 发布每一帧的门控状态。字符识别模块应只消费
`valid_operation=true` 的帧或时间段；`air_invalid`、`air_idle` 都应丢弃：

```json
{
  "type": "writing_state_frame",
  "timestamp": 12.34,
  "state": "air_invalid",
  "valid_operation": false,
  "segment_id": null
}
```

## 参数调节

默认配置假设：

- 采样率约 100 Hz；
- `lin_acc` 以 g 为单位；
- gyro 使用项目当前 LPMS 数据的单位；
- 已经存在 `lin_acc_x/y/z`，否则只做“减去整段均值”的离线兜底。

在没有标注数据前，优先调节 `DetectorConfig` 中的：

- `contact_impact_threshold`
- `lift_impact_threshold`
- `writing_motion_max`
- `air_motion_threshold`
- `contact_confirm_seconds`
- `lift_confirm_seconds`

## 输入约定

`paper_touch.py` 的六个通道顺序固定为：

```text
lin_acc_x, lin_acc_y, lin_acc_z, gyro_x, gyro_y, gyro_z
```

论文在预处理阶段先去除重力，再把六轴数据送入 touch detector。当前项目已有
`ImuFrame.lin_acc`，因此默认直接使用 `lin_acc + gyro`；如果只有原始加速度，
适配器会暂时回退到 `acc + gyro`，这只是接口兜底，不是最终训练预处理。

## 数据管线:原始数据 → clean 数据集

本仓库所有训练/评估结果使用的都是官方清洗版 `clean_data_delete_g`(21 用户 /
566 样本,实际使用 564),不是自行转换的数据。原始版 `/data/huyang/datasets/WritingRing/data`
(12 GB,606 组 `*_ring_0.bin` + `*_board_*.gz` + `*_timestamp.txt`)仅用于逐字符/逐词
标签时间戳、去重力探针与 board 单位标定。

作者的 raw→clean 处理规则(逐样本实测逆向,8 个用户样本复现)为:

1. **裁剪窗口**:board 录制区间去掉头部 `0.6 s`、尾部 `1.0 s`,与 ring 录制取交集;落在
   窗口内的原始 ring 行数即 clean 样本长度 `T`(与官方相差 0–7 帧/万帧)。
2. **时间网格**:均匀网格,固定步长 `4977.75 µs`(≈200.894 Hz),所有样本一致。
3. **gyro**:截取的原始 ring 行,与官方**逐位一致**(max|Δ| = 0)。
4. **lin_acc**:原始加速度去重力。论文声称 Madgwick(β=0.041)但与官方数据不符,精确
   算法未公开;脚本用 ~1 Hz 零相位高通近似(与官方各轴相关 0.55–0.98)。
5. **board**:Sensel 每帧第 0 个触点位置线性插值到网格(与官方平均距离 ~1e-4 归一化
   单位,≈0.02 mm)。
6. **mask**:网格点两侧原始帧都有触点时为 1(AND 规则;与官方一致率 99–100%,IoU 0.98)。
7. **y**:原始帧率触点位移重采样到网格 × 网格步长(与官方相关 0.96–0.998);它对应的是
   速度/位移增量,不是 `board` 的逐帧差分(严格复算后者相关仅 0.06–0.11)。

转换脚本 `build_clean_dataset.py`(自包含 gzip+pickle 加载器,不依赖作者仓库):

```powershell
# 快速验证:重建 3 个样本并与官方 clean 逐字段比对
python build_clean_dataset.py `
  --output-root /tmp/clean_rebuilt `
  --users user_0 --actions 0 --samples 1,2,3 `
  --validate-against /data/huyang/datasets/WritingRing/clean_data_delete_g/data

# 全量重建(566 样本,约几分钟)
python build_clean_dataset.py `
  --output-root /data/huyang/datasets/WritingRing/clean_data_rebuilt/data `
  --workers 4
```

重建一致性(8 用户 × 1 样本;`user_0` 全量 32 样本 0 失败):

```text
gyro  max|Δ| = 0(逐位一致)
board 平均距离 4e-5 … 4e-4 归一化单位
mask 一致率 0.987–1.000,IoU 0.97–0.99
y    相关 0.96–0.998,相对误差 ~4%
acc  相关 0.55–0.98(近似步,不可逐位还原)
```

遗留不确定性:官方网格锚点在 board 会话时钟上,与 ring 行时间戳相差 −0.07…−0.15 s
(逐样本不同,raw 中无两设备时钟关系);脚本默认锚在 ring 首行,`--anchor-shift-us` 可
平移,`--validate-against` 会拟合最贴合的偏移。产物可被
`paper_trajectory_dataset.discover_trajectory_samples/load_sample_arrays` 直接读取。

可视化对照见 `outputs/data_pipeline/raw_to_clean_overview.png`,完整逆向记录与证据见
`.trae/documents/NOTES.md`「原始数据 → 干净数据」章节。

## 参考

Zhe He et al. *WritingRing: Enabling Natural Handwriting Input with a Single IMU Ring*. CHI 2025. DOI: `10.1145/3706598.3714066`.
