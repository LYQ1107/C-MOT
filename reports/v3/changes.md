# C-MOT V3 修复与实验摘要

本报告只引用脱敏哈希、basename和真实执行状态；未运行项保留为 `NOT_RUN`。

- 公共执行路径接入了 V3 配置解析、stable video/frame/track UID、分来源 GT/PL 监督、质量门控 PL、一对一 PL 匹配、合法 GT 回放、跨来源分层采样、推理阈值分离、重复查询抑制和 checkpoint→prediction→TrackEval 绑定。
- 运动分支仅实现并接通 `one_step_residual_v2`；默认基线 motion 关闭，未声明 GRU、MoE、LoRA、Prompt或新检测器已运行。
- J3 仅为 `diagnostic_joint`，不向 CIL 提供 checkpoint、PL、memory或校准参数。
- 原始数据、标注、权重、完整预测和服务器私有路径均未写入仓库。
- 下载审计：`NOT_RUN_no_new_data_or_dependency_download`；代理路径检查：`NOT_RUN_no_download_requested`。本轮没有新数据或大依赖下载。
- canonical/eval manifest SHA256：`97eb54211c9ce783baf96e8852ec84a8413f4fee80adc7c3d2b9608b12775fcd` / `f37ed78d861b6c04f37ade692db59b99a025c591fd6e4c2a21339cfb38c0254e`。

## 真实状态

- `J3/J3-diagnostic`: `COMPLETE`, 1200 steps, scope `pilot`.
- `S1/R-ER-S1`: `COMPLETE`, 600 steps, scope `pilot`.
- `S2/R-ER-S2`: `COMPLETE`, 600 steps, scope `pilot`.
- `S1/R-QPL-KD-S1`: `SKIPPED_EQUIVALENT`, 0 steps, scope `pilot`.
  - PL admission=OK, accepted_by_class={'206': 17}, disabled=[].
  - observed redundant run steps=600; it is not counted as an independent result.
- `S2/R-QPL-KD-S2`: `SKIPPED_EQUIVALENT`, 0 steps, scope `pilot`.
- `S1/R-QPL-S1`: `COMPLETE`, 600 steps, scope `pilot`.
  - PL admission=OK, accepted_by_class={'206': 17}, disabled=[].
- `S0/S0-v3`: `COMPLETE`, 1200 steps, scope `pilot`.
- `None/asset_check`: `COMPLETE`, 0 steps, scope `asset_diagnosis`.
- `S1/zero_S1_S0-v3_expanded_vocabulary`: `COMPLETE`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`expanded`, override IDs=[206, 792].
- `S1/zero_S1_S0-v3_previous_vocabulary`: `COMPLETE`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`previous`, override IDs=[206].
- `S2/zero_S2_R-ER-S1_expanded_vocabulary`: `COMPLETE`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`expanded`, override IDs=[206, 792, 1122].
- `S2/zero_S2_R-ER-S1_previous_vocabulary`: `COMPLETE`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`previous`, override IDs=[206, 792].
- `S2/zero_S2_R-QPL-KD-S1_expanded_vocabulary`: `NOT_RUN`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`expanded`, override IDs=NOT_RUN.
- `S2/zero_S2_R-QPL-KD-S1_previous_vocabulary`: `NOT_RUN`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`previous`, override IDs=NOT_RUN.
