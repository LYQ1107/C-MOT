# C-MOT V4 已知限制

- 本轮只执行 S2 三个主 run；AUX 是否执行由报告中的固定 gate 决定。
- 运动监督只使用合法 GT/GT replay，PL 不直接进入 motion regression。
- history-AGN 的 semantic projection 若无有效语义梯度，按实际梯度审计保留该事实。
- 600 steps、固定视频清单和固定阈值属于 pilot 结果；未运行项保留 `NOT_RUN`。
