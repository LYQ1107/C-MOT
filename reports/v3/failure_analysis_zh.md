# C-MOT V3 失败与限制记录

只保留实际状态，不用估计值补齐。

- `S1/R-QPL-KD-S1`: `SKIPPED_EQUIVALENT`；原因：runtime KD audit found zero valid replay-aligned objects; no independent KD result is counted
  - observed_optimizer_steps=600，原始产物仅作为等价审计证据。
- `S2/R-QPL-KD-S2`: `SKIPPED_EQUIVALENT`；原因：S1 KD had zero valid replay-aligned objects; S2 KD branch is equivalent to its QPL parent and is not retrained
- `S2/zero_S2_R-QPL-KD-S1_expanded_vocabulary`: `NOT_RUN`；原因：parent S1 checkpoint was not complete
- `S2/zero_S2_R-QPL-KD-S1_previous_vocabulary`: `NOT_RUN`；原因：parent S1 checkpoint was not complete

本轮 KD 实际有效对象为 0；R-QPL-KD-S1 的观察 run 不计为独立 KD 结果，R-QPL-KD-S2 未训练。
新类与全部已见类指标均按 TrackEval 原始输出记录；负 MOTA 或无提升不作阈值修饰。
