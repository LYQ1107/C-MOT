# V3.1 controlled-ablation known issues

本轮只完成受规格授权的四个 controlled ablations；以下问题按规格保留，不进入本轮归因。

- `DEFERRED_AFTER_CONTROL_EXPERIMENTS`: negative_fraction 当前实现与配置命名问题。
- `DEFERRED_AFTER_CONTROL_EXPERIMENTS`: effective_lambda aggregate 报告命名问题。
- `DEFERRED_AFTER_CONTROL_EXPERIMENTS`: `cmot/pseudo/track_filter.py::_limit_segment_length` 的 segment score 截取问题；本轮未修改、未重新筛选 PL。
- `DEFERRED_AFTER_CONTROL_EXPERIMENTS`: motion 及其它未授权 architecture/阈值变体。

以上冻结项不应被解释为本轮控制实验已经修复。
