# SPEC — WritingRing 轨迹重建复现(任务 1)

> 状态:待评审(2026-10-03)
> 关联:`.trae/todo/TODO.md` 任务 1;论文 He et al., CHI 2025, DOI `10.1145/3706598.3714066`
> 本文档经用户评审通过后,再据此产出实现计划(`writing-plans`),评审前不写产品代码。

## 1. 目标与范围

**交付**:在官方 `clean_data_delete_g` 数据上复现论文的轨迹重建方法——TCN+LSTM 流式预测指尖速度,只在接触段(`mask=1`)积分重建 2D 轨迹,并按论文指标与 touchpad 真值对比,产出 checkpoint + report + 可视化。

**非目标(本轮不做)**:
- 轨迹→字符识别(任务 2;论文用 Google IME,不可复现);
- 实时部署、模型量化、端侧移植;
- Madgwick 去重力(clean 数据已去重力,论文预处理仅在有原始数据/部署时需要);
- touch detector 训练(已有 `paper_touch.py`;本轮离线评估直接用官方 mask 作为接触真值)。

**成功标准**:
1. test 归一化误差均值达到论文同一量级(论文 0.073),mm 误差同数量级(论文 1.64 mm);
2. 同时给出用户无重叠划分结果作为严格对照(预期略差,须如实报告);
3. `report.json` 可复现实验(含 split、超参、逐 epoch 指标、测试指标、单位标定);
4. 单元测试通过;README/STRUCTURE/DESIGN/NOTES 同步更新。

## 2. 论文方法要点(已核实)

- **输入**:去重力后的三轴加速度 + 三轴角速度(6 通道)。论文用 Madgwick(β=0.041)估计姿态后减去重力。
- **触控状态**:ResNet 四分类 20×6@0.1s(8/16/32 通道,BN+Dropout),给出 press/lift 门控;`paper_touch.py` 已复现。
- **轨迹模型**:13 帧滑窗(TCN kernel=3)→ TCN 压缩为 1×128 特征 → LSTM(hidden=128,隐状态跨帧流式携带)→ 线性层(ReLU)→ 预测窗口中点速度 (vx, vy)。延迟 6.5 帧(32.5 ms),即**使用 6 帧未来数据,非严格因果**;推理时连续滑窗,仅接触帧输出有效。
- **训练**:75 s(15000 帧)长片段(不足头部补零、超长对称裁剪)、batch=16、Adam lr=1e-3、MSE、500 epochs、随机 70/10/20;**仅接触帧计入 loss**。
- **评估**:逐点距离 / 该轨迹 GT bbox 对角线;论文均值 0.073(>80% <0.1);无归一化 0.164 cm。
- **消融**:碎片化→流式 0.116→0.073(-36.7%);无 TCN→有 0.095→0.073(-23.1%)。本轮不承诺完整复现消融,但在训练脚本预留开关。

## 3. 数据与假设

- 路径:`/data/huyang/datasets/WritingRing/clean_data_delete_g/data/user_*/{action}/*_{x,y,mask,board,timestamp}.npy`;样本=一个 block,21 用户 566 样本。
- 字段:`x (N,6) float32`(去重力 acc xyz + gyro xyz,@200 Hz)、`board (N,2)`(触控板坐标插值到 200 Hz,空接触为 0)、`mask (N,)∈{0,1}`、`y (N,2)`(逐帧位移,作为速度监督目标;实现时验证 `y[i]` 与 `board[i+1]-board[i]` 的确切换算,以 report 记录)。
- 已知瑕疵:2 个样本 `x` 与 `timestamp` 长度不一致(user_0/0/0、user_7/0/0),加载时跳过并计数。
- 时间对齐:clean 时间戳为 200 Hz 均匀网格,board 已按论文校准的时间偏移对齐;`y` 以官方值为准。
- **board 单位未标定**:clean `board` 为归一化/缩放坐标(实测范围 x≈[0,0.63]、y≈[0,0.68])。mm 指标需用 Sensel Morph 物理尺寸(240×169.5 mm)与原始 `*_board_*.gz`(需 `compress_pickle`)回归标定;归一化指标不受影响。标定结果写入 report,供论文 1.63 mm 对照。
- 划分:主结果按论文随机 70/10/20(按样本);对照实验用现有 `default_user_split`(用户无重叠)。归一化统计只用训练集。

## 4. 模块设计

| 文件 | 职责 | 复用/约定 |
| --- | --- | --- |
| `paper_trajectory.py`(新) | 配置、滑窗、TCN+LSTM 网络、流式推理、接触段积分、指标、checkpoint 加载、CLI 摘要 | 命名与风格对齐 `paper_touch.py`;6 通道顺序不变 |
| `paper_trajectory_dataset.py`(新) | 样本发现、长度校验、长片段构造(补零/对称裁剪)、归一化、`Dataset` | 复用 `paper_dataset.discover_cleaned_samples` |
| `train_paper_trajectory.py`(新) | 训练/验证/测试入口 + report 落盘 | 对齐 `train_paper_touch.py` 结构 |
| `test_paper_trajectory.py`(新) | 单元测试 | 对齐 `test_paper_touch.py` |
| `.trae/*` + `README.md` | 文档同步 | 完成后更新 |

### 4.1 关键接口

```python
@dataclass(frozen=True)
class PaperTrajectoryConfig:
    sample_rate: float = 200.0
    window_frames: int = 13          # 论文总窗长
    tcn_kernel: int = 3              # 论文 TCN 窗长
    tcn_channels: tuple[int, ...] = (16, 16, 32, 32, 64, 128)  # 6 层 valid 卷积,13→1
    feature_dim: int = 128           # TCN 输出特征维度
    lstm_hidden: int = 128
    dropout: float = 0.2
    window_offset: int = 6           # 预测对应窗口中点;非因果 6 帧前瞻
    board_mm_scale: tuple[float, float] = (240.0, 169.5)  # 待标定

class WritingRingTrajectoryNet(nn.Module):
    # forward: (B, T, 6) -> (B, T, 2) 每个时刻对应窗口中点位移(前/后 6 帧无输出)
    ...

def predict_sequence(model, x_seq, chunk_len=None) -> np.ndarray:
    # 流式(可 TBPTT 分块)前向,返回与输入等长的 (T,2),无输出的边沿用 NaN
    ...

def integrate_contact_segments(deltas, mask, config) -> list[dict]:
    # 接触段 = 极大连续 mask==1 run,长度 >= min_segment_frames(默认 5)
    # 不做 gap 合并;段内 cumsum(deltas) 得轨迹(起点归零);返回 start/end/xy
    ...

def trajectory_metrics(pred_xy, gt_xy, mm_scale) -> dict:
    # normalized_mean(逐点距离/GT bbox 对角线)、mm_mean、分位数、占比
    ...
```

### 4.2 网络结构(论文未公开 TCN 细节,以下为受描述约束的默认假设)

- 6 层因果 valid `Conv1d(k=3, stride=1, padding=0)`:`13→11→9→7→5→3→1`,通道 `(16,16,32,32,64,128)`,每层 Conv→BN→ReLU→Dropout,通道相同的相邻层加残差;
- 输出 1×128 特征;LSTM(128→128)对逐帧特征序列流式处理;`Linear(128→128)+ReLU → Linear(128→2)`;
- 训练用 TBPTT 分块(默认 3000 步,块间 detach 但保留隐状态)控制显存;`report.json` 记录该实现差异与块长。

## 5. 训练与评估协议

- **目标**:`y[t]`(官方逐帧位移);预测头输出对应伸缩中点,与 `y` 对齐后做 mask MSE:`Σ mask_t·||pred_t-y_t||² / Σ mask_t`。
- **长片段**:默认按论文——≤15000 帧头部补零,>15000 帧对称裁剪;提供 `--long-sample-strategy chunks`(等长切块、块间连续)作为补充实验。
- **超参**:batch=16、Adam lr=1e-3、weight_decay=0、epochs 默认 100 + val 早停(论文 500;在 report 记录差异),seed 固定。
- **划分**:主结果 random(70/10/20 样本级);对照 user-disjoint。两份结果分别落盘 `models_trajectory/` 与 `models_trajectory_user/`。
- **模型选择**:以验证集"段级归一化误差均值"最小保存 best checkpoint。
- **评估**:在 test 上整 block 前向 → 接触段积分 → 段级指标(主)与整块指标(辅);另报告按 action(字母/连笔词/断笔词)与用户的分布;可视化少量重建 vs GT 轨迹图(PNG,输出到 output 目录,不入库)。
- **单位标定**:对标 `board` 与原始 Sensel 坐标(必要时安装 `compress_pickle`),得到 mm/单位刻度;若无法可靠标定,mm 指标标注"按 Sensel Morph 尺寸假设"。

## 6. 测试与验证(`test_paper_trajectory.py`)

1. 长片段构造:短样本头部补零位置正确、长样本对称裁剪长度正确;
2. 窗口/输出对齐:输出第 t 项对应输入窗 `[t-6, t+6]`,前/后 6 帧为 NaN;
3. mask 语义:非接触帧不进入 loss(合成数据上 loss 梯度为 0);
4. 积分:合成常速度位移 → 直线,终点等于 cumsum;接触段切分正确;
5. TBPTT 等价性:分块前向(块间 detach)与整段前向在数值容差内一致;
6. 模型前向形状;checkpoint 保存/加载往返一致;
7. 指标:构造已知误差的轨迹,验证 normalized/mm 数值正确。

## 7. 风险与开放问题

| 风险 | 应对 |
| --- | --- |
| TCN 具体拓扑未公开(仅"窗长 3、输出 128") | 按 44 KB 量级与描述约束默认实现,通道可配;在 report 说明假设 |
| board 单位标定不确定 | 归一化指标为主;mm 同时给标定值与假设值 |
| 15000 步 BPTT 显存/耗时(GPU 被占用) | TBPTT 默认 3000;先小规模冒烟(样本子集、少量 epoch)再全量 |
| 论文随机划分存在用户泄漏,数字偏乐观 | 主报随机划分以对齐论文,同时报用户无重叠 |
| 2 个样本长度瑕疵 | 加载时跳过并在 report 计数 |
| 非因果 6 帧前瞻 | 评估按论文口径;文档写明实时延迟 32.5 ms |

## 8. 交付物与后续

- 交付:`paper_trajectory.py`、`paper_trajectory_dataset.py`、`train_paper_trajectory.py`、`test_paper_trajectory.py`、训练产物与 report、文档更新。
- 后续(任务 2):逐字符标签切分 → 轨迹→26 字母分类器 → 单词(连笔/断笔、3000 词表+DTW)。

## 9. 参考

- Zhe He et al. *WritingRing: Enabling Natural Handwriting Input with a Single IMU Ring*. CHI 2025. DOI: `10.1145/3706598.3714066`.
- 数据集:https://huggingface.co/datasets/dBHz/WritingRing (CC 许可;clean_data_delete_g.zip 与 data.zip 均在本机 `/data/huyang/datasets/WritingRing`)。
- 既有实现:`paper_touch.py`(touch detector)、`paper_dataset.py`(样本发现与划分)。
