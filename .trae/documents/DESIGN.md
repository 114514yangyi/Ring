# DESIGN — 项目目标与设计理念

## 一句话目标

把 WritingRing 的书写接触/抬笔检测器工程化落地到实时书写系统:从单 IMU 戒指数据实时判断笔尖是否接触书写平面,输出有效书写段,供下游字符识别使用。

## 设计理念

- 学习模型为主路径:按论文复现 touch detector——0.1 s 固定窗口、每窗 20x6、四分类 `contact` / `air` / `lift` / `press`、三个 1-D ResNet 残差块(通道 8/16/32),只把 `press` 和 `lift` 解码成书写起止事件。
- 实时优先:检测器需可在线运行,通过 `realtime.py` 接入 `ImuBus`,发布 `writing_state`(事件)与 `writing_state_frame`(逐帧门控),下游只消费 `valid_operation=true` 的帧或时间段。
- 规则基线兜底:`detector.py` 的硬阈值 + 状态机用于回归测试与无模型兜底,不作为最终方案;两条路径保持接口一致,便于替换。
- 接口约定固定:六通道顺序 `lin_acc_x, lin_acc_y, lin_acc_z, gyro_x, gyro_y, gyro_z`;`pen_up` 之后到下一次 `writing_start` 之间的动作不参与字符识别。
- 复现与落地并重:模型训练与评估离线进行(按用户无重叠划分 train/val/test),实时链路直接加载训练好的 checkpoint。

## 技术栈与运行环境

- 语言/框架:Python + PyTorch。
- 环境:conda `vae`(torch 2.2.0+cu121),依赖 numpy、pandas。
- 依赖文件:暂无(无 `requirements.txt`,待补充)。
- 模块挂载:仓库目录名为 `Ring`,代码与测试按 `writing_state.*` 导入;在父目录创建软链 `writing_state -> Ring` 后按 README 命令运行。
- 训练数据:WritingRing 清洗后数据 `clean_data_delete_g/data`,目录结构 `user_*/{action}/*_x.npy` + `*_mask.npy`,默认路径 `writing_state/clean_data_delete_g/data`(不在仓库内)。

## 里程碑

- [ ] 实时链路端到端验证:戒指原始帧 -> touch detector -> `writing_state` 事件稳定输出。
- [ ] 模型路径与规则基线在相同数据上的一致性/回归对比。
- [ ] 与下游字符识别模块联调,确认 `valid_operation` 门控有效。
