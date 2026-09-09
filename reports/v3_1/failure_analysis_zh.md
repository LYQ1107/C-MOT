# C-MOT V3.1 失败与限制记录

只保留实际状态，不用估计值补齐。

- `S2/R-QPLSEG-PF-S2`: `NOT_RUN_CONDITION_NOT_MET`；原因：stability/plasticity conditions were not both met
- cap 修复前的中间尝试因片段上限校验暴露出选择器错误，在 9 steps 后停止并隔离，未计入正式结果；修复后正式 S1 从 step 1 重跑。

KD 是否有效以每个新命名 QPLSEG run 的实际 gate 统计为准；若被 fail-fast 阻断，保留具体 gate 状态，不改写成等价结果。
新类与全部已见类指标均按 TrackEval 原始输出记录；负 MOTA 或无提升不作阈值修饰。
