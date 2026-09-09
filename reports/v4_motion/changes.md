# C-MOT V4 历史轨迹动力学

本报告只记录实际执行状态、basename和SHA256；数据、GT、权重和服务器私有路径未进入仓库。

- parent: `R-QPLSEG-S1`, checkpoint SHA256 `e5157b270878bab010954c7642f8eb3cb5562e80b3a3e9073748e52237a216e1`。
- S2 corrected policy: `qpl_segment_v3`; selector `qpl_segment_v3_highscore_center`。
- QPL v2/v3 selected-segment、frame和source-video overlap见 `results.json`。
- B0、history-AGN、history-COND共用同一 current/replay/QPL/sampler/eval绑定；motion mode是唯一预期差异。
- 实际接入接口：`cmot.continual_v4` 配置解析/传参、`qpl_segment_v3` 视图校验、分数/帧号/视频号确定性截断、fraction 负采样、聚合损失归一化和有限梯度审计。
- 实际接入历史动力学：OVTR track `Instances` 的 detached history ring buffer、真实 timestamp/dt、`CategoryConditionedMotionPrior`、inverse-sigmoid reference 更新、合法 GT/GT replay motion loss 与运行时覆盖统计；未接入 GRU 外的额外模型模块。
- 训练步数均按正式 optimizer steps 记录为 600；评价范围标为 `pilot`，不称完整 benchmark。
- `motion_audit.json` 和 `artifact_matching_audit.json` 记录实际 head、motion pairs、参数梯度/更新及公共数据绑定；原始预测和 TrackEval 仅以 basename/SHA 引用。
- 下载与新依赖：`NOT_RUN`；本轮未下载新数据或大依赖。
