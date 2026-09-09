# C-MOT V3.1 修复与实验摘要

本报告只引用脱敏哈希、basename和真实执行状态；未运行项保留为 `NOT_RUN`。

- 公共执行路径接入了 V3.1 配置解析、stable video/frame/track UID、分来源 GT/PL 监督、质量门控 PL、一对一 PL 匹配、合法 GT 回放、跨来源分层采样、推理阈值分离、重复查询抑制和 checkpoint→prediction→TrackEval 绑定。
- 本轮 motion 保持关闭；未声明 GRU、MoE、LoRA、Prompt或新检测器已运行。
- J3 仅为 `diagnostic_joint`，不向 CIL 提供 checkpoint、PL、memory或校准参数。
- 原始数据、标注、权重、完整预测和服务器私有路径均未写入仓库。
- 下载审计：`NOT_RUN_no_new_data_or_dependency_download`；代理路径检查：`NOT_RUN_no_download_requested`。本轮没有新数据或大依赖下载。
- canonical/eval manifest SHA256：`97eb54211c9ce783baf96e8852ec84a8413f4fee80adc7c3d2b9608b12775fcd` / `f37ed78d861b6c04f37ade692db59b99a025c591fd6e4c2a21339cfb38c0254e`。

## 真实状态

- `S1/R-QPLSEG-KD-S1`: `COMPLETE`, 600 steps, scope `pilot`.
  - PL admission=OK, selected_segments_by_class={206:20}, disabled=[]; selected_frame_predictions=139.
- `S2/R-QPLSEG-KD-S2`: `COMPLETE`, 600 steps, scope `pilot`.
  - PL admission=OK, selected_segments_by_class={206:20}, disabled=[792]; selected_frame_predictions=146.
- `S2/R-QPLSEG-PF-S2`: `NOT_RUN_CONDITION_NOT_MET`, 0 steps, scope `pilot`.
- `S1/R-QPLSEG-S1`: `COMPLETE`, 600 steps, scope `pilot`.
  - PL admission=OK, selected_segments_by_class={206:20}, disabled=[]; selected_frame_predictions=139.
- `S2/R-QPLSEG-S2`: `COMPLETE`, 600 steps, scope `pilot`.
  - PL admission=OK, selected_segments_by_class={206:20}, disabled=[792]; selected_frame_predictions=146.
- `None/asset_check`: `COMPLETE`, 0 steps, scope `asset_diagnosis`.
- `S1/zero_S1_S0-v3_expanded_vocabulary`: `COMPLETE`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`expanded`, override IDs=[206, 792].
- `S1/zero_S1_S0-v3_previous_vocabulary`: `COMPLETE`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`previous`, override IDs=[206].
- `S2/zero_S2_R-QPLSEG-KD-S1_expanded_vocabulary`: `COMPLETE`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`expanded`, override IDs=[206, 792, 1122].
- `S2/zero_S2_R-QPLSEG-KD-S1_previous_vocabulary`: `COMPLETE`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`previous`, override IDs=[206, 792].
- `S2/zero_S2_R-QPLSEG-S1_expanded_vocabulary`: `COMPLETE`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`expanded`, override IDs=[206, 792, 1122].
- `S2/zero_S2_R-QPLSEG-S1_previous_vocabulary`: `COMPLETE`, 0 steps, scope `pilot`.
  - zero-step vocabulary=`previous`, override IDs=[206, 792].

## 中间执行审计

- cap 修复前曾观察到选择器输出 32 个片段（契约上限为 20）；该中间运行在 9 个 optimizer steps 后停止，产物已隔离且不计入正式结果。
- 修复后 `R-QPLSEG-S1` 从 step 1 重新开始并完成 600 步；正式 S1/S2 结果均只引用修复后的 20 片段清单。
