# V3.1 controlled-ablation delivery

本分支只补齐 V3.1 要求的受控对照，不修改现有 QPL segment、KD、推理阈值、回放 memory 或评价切片。

- `cmot/config.py` 增加四个控制方法的显式 flag 映射。
- `tools/cmot/run_continual_v3.py` 增加 `controls` 路由、固定输入 SHA 校验、C1 frame-identity 派生视图、显式 S2 parent 路由、sampler/exposure 审计和脱敏报告生成。
- C1 只重键训练 PL 的 track identity；PL box、class、frame、segment 和采样保持不变。
- C2/C3/C4 分别固定 parent、S2 current view、replay、QPL manifest、KD 开关和 motion=none；不生成新的 S2 QPL。
- 公共结果只包含方法、指标、曝光统计、文件 basename、SHA256 和控制表；不包含数据、GT、权重、完整预测、服务器路径或凭据。

定向验证：受影响的两个 Python 文件已通过 `py_compile` 和 `git diff --check`；未运行全仓库 pytest 或新增大规模 smoke。
