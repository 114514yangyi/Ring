# Ring 项目交接入口

最后核对：2026-10-12 09:00 CST。新 AI 会话从这里开始；完整阅读顺序见下方。

## 先读

1. [HANDOFF_PROTOCOL.md](documents/HANDOFF_PROTOCOL.md)：跨 AI 的状态维护与交接规则。
2. [STRUCTURE.md](documents/STRUCTURE.md)：项目结构与模块职责。
3. [DESIGN.md](documents/DESIGN.md)：项目目标和设计理念。
4. [NOTES.md](documents/NOTES.md)：按日期追加的研究过程、决策和结果。
5. [RULES.md](documents/RULES.md)：项目约束和完整必读清单。
6. [TODO.md](todo/TODO.md)：用户维护的当前任务列表。
7. 按当前任务再读相关计划、代码、日志和报告。

## 当前状态

- `TODO.md` 当前没有未完成任务条目；不要自行把 DESIGN 中尚未完成的长期里程碑改写成 TODO。
- 最近记录的项目实验截至 2026-10-11：SmartRing 新批次双向轨迹模型及单向 LSTM 对照、下游字母识别均已完成。关键结果、checkpoint 路径、评测限制在 `NOTES.md` 的「SmartRing 数据运行记录」和「SmartRing 单向 LSTM 对照」两节。
- 2026-10-12 09:00 CST 检查时没有发现 Ring 仓库相关训练/评测进程。运行状态可能变化，新会话开始工作前应重新检查。
- `DESIGN.md` 仍列有实时链路端到端验证、模型路径与规则基线对比、下游 `valid_operation` 门控联调等未完成长期里程碑；它们尚未被用户加入任务清单。

## 接下来

目前没有获用户确认的活动任务。先读上面的文档；若用户没有指定新工作，向用户确认希望推进哪个长期里程碑。用户确定任务后，由用户在 `TODO.md` 中添加条目，再按项目规则实施并持续维护本 handoff。
