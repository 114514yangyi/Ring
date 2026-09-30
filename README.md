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

## 参考

Zhe He et al. *WritingRing: Enabling Natural Handwriting Input with a Single IMU Ring*. CHI 2025. DOI: `10.1145/3706598.3714066`.
