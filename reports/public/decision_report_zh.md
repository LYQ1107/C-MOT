# C-MOT V2 修复与真实实验交付（脱敏）

本报告对应附件 `CMOT_Codex_修复优先与实验执行_V2.md`，审阅基准为
`cc488678df2ffdf5dd5103ce6380cfa9a2e9065f`。报告只包含实际运行后的
小型、脱敏摘要；原始视频、标注、权重、完整预测和训练日志留在服务器，
不进入公开仓库。

## 结论

公共执行路径已修复并实际运行：新 S0 完成 600 个 optimizer steps；同一
S0 分叉的 B1/O1/O2 S1 各完成 300 steps；随后 B1/O2 S2 各完成 300 steps。
所有最终评价均绑定具体 checkpoint、prediction SHA、resolved-config SHA 和
immutable view manifest。

S1 的固定评价切片上，O1 的全部已见类 TrackEval detection-average HOTA 为
`0.162504302`，略高于 B1 的 `0.159927010`；O2 为 `0.158304870`，没有
超过 O1，照实记录。S2 上 O2 的全部已见类 HOTA 为 `0.185716814`，B1 为
`0.172217674`；这只是本次固定 pilot 协议的实际结果，不外推为完整
BDD100K benchmark 或方法有效性结论。

这里的 `scope=full` 表示该行声明的训练步数和绑定评价已完整执行；评价集仍
是固定的 `pilot` 切片（16 个独立视频、3,238 帧），不是完整 benchmark。

## 指标

`HOTA` 为 TrackEval 的 `HOTA(0)`，另列 `HOTA_mean`；MOTA 为 CLEAR，
IDF1 为 Identity。global ID：car=`206`、pedestrian=`792`、truck=`1122`。
`old/new` 是该阶段的类别角色，`all_seen` 是官方 TrackEval
`cls_comb_det_av` 聚合。

| method | stage | scope | steps | role/class | HOTA | HOTA_mean | MOTA | IDF1 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| S0-repaired | S0 | full / pilot eval | 600 | seen / car | 0.353572292 | 0.252145991 | -1.358633333 | 0.241925306 |
| B1 | S1 | full / pilot eval | 300 | old / car | 0.160476638 | 0.126087856 | -6.414633333 | 0.063692245 |
| B1 | S1 | full / pilot eval | 300 | new / pedestrian | 0.124656573 | 0.097877171 | 0.008847255 | 0.100064419 |
| B1 | S1 | full / pilot eval | 300 | all_seen | 0.159927010 | 0.125669415 | -5.685222941 | 0.064336697 |
| O1 | S1 | full / pilot eval | 300 | old / car | 0.162738728 | 0.126961026 | -6.013366667 | 0.066130211 |
| O1 | S1 | full / pilot eval | 300 | new / pedestrian | 0.149253976 | 0.117587237 | 0.037991153 | 0.126133909 |
| O1 | S1 | full / pilot eval | 300 | all_seen | 0.162504302 | 0.126816155 | -5.326212215 | 0.067243390 |
| O2 | S1 | full / pilot eval | 300 | old / car | 0.158807060 | 0.122150655 | -6.088033333 | 0.063022404 |
| O2 | S1 | full / pilot eval | 300 | new / pedestrian | 0.127539891 | 0.101509955 | 0.028883685 | 0.108240535 |
| O2 | S1 | full / pilot eval | 300 | all_seen | 0.158304870 | 0.121831066 | -5.393434388 | 0.063829533 |
| B1 | S2 | full / pilot eval | 300 | old / car | 0.174063199 | 0.134832358 | -6.512966667 | 0.075531538 |
| B1 | S2 | full / pilot eval | 300 | old / pedestrian | 0.011702196 | 0.009262736 | 0.000260213 | 0.004663212 |
| B1 | S2 | full / pilot eval | 300 | new / truck | 0.000000000 | 0.000000000 | -0.000871080 | 0.000000000 |
| B1 | S2 | full / pilot eval | 300 | all_seen | 0.172217674 | 0.133452542 | -5.583978737 | 0.074170855 |
| O2 | S2 | full / pilot eval | 300 | old / car | 0.187803152 | 0.143135060 | -6.173966667 | 0.084680208 |
| O2 | S2 | full / pilot eval | 300 | old / pedestrian | 0.011817546 | 0.009166469 | -0.000520427 | 0.004144004 |
| O2 | S2 | full / pilot eval | 300 | new / truck | 0.000000000 | 0.000000000 | -0.000871080 | 0.000000000 |
| O2 | S2 | full / pilot eval | 300 | all_seen | 0.185716814 | 0.141602201 | -5.293418308 | 0.083071248 |

## 已实施接口与执行路径

- `cmot/schema.py`、BDD/TAO converter、real-video dataset：分离 source ID、global semantic ID、text row、select column 和 track ID；补齐 `exhaustive_global_ids`、ignore region、annotation validity、timestamp/dt metadata。
- view builder、clip memory、sampler：主层和辅助层的部分标注掩码、GT/PL 冲突优先级、未来类隔离、真实连续历史片段、合法标签快照回放和跨视频分层采样；回放 clip 不再跨独立窗口拼接。
- `ovtr/models/ovtr.py`、updater 和 runtime：分类/检测/运动损失只做一次公共归一化；GT 与 PL 分开匹配；运动只读取过去/当前特征，监督使用合法 GT/gt_replay；inverse-sigmoid ref 更新实际使用 residual 和真实 dt。
- 实际接通的运动实现只有 `one_step_residual_v2`：零初始化末层、软语义条件、detach、warmup、速度上限和 dt 检查。GRU、未接入的三模式模型均未宣称运行。
- inference/runtime：重复 query 抑制在 global class assignment 后执行，track aging 只发生一次；推理输出使用原子 partial 文件替换，并记录运行统计。
- config/train/infer/evaluate：repair_v2 配置映射到真实 dataset、optimizer、sampler、motion、inference 和 evaluator；checkpoint、prediction、metric 之间保存并校验绑定；评价保留空帧、ignore/crowd 和官方类别聚合。
- 本轮还修复了一个实际暴露的调用链错误（匹配结果局部变量缩进导致的 `NameError`），以及由合并回放窗口导致的无效 dt；修复后最终 runs 的 `invalid_dt_count` 均为 0。

## 数据、权重与安全审计

- 数据：使用服务器上已有、只读的 BDD100K MOT canonical；盘点规模为 190 个视频、37,725 帧、423,951 个标注。最终训练视图限制为每阶段 32 个视频，评价使用固定 16 个视频、3,238 帧；TAO-Amodal 仅完成本地稀疏资源盘点，未进入本轮训练/评价。
- 类别顺序：`car → pedestrian → truck`；当前流只使用新类合法 GT 与旧类 PL；replay 只来自此前保存的合法 memory snapshot。
- foundation detection pretrain：已有本地审计权重，SHA-256 `0862cac87ad50f58a01ce17d4e44af0468ad8639cfccd18d66d2e9b2570d839e`，用于新 S0 foundation partial init。
- official OVTR 5-frame 权重：已有本地资源，SHA-256 `7b184a0f149259047ef3f03263cf9178fc9c5e051d08882f065aa52118266d56`，只作来源/asset 检查，未作为合法 S0。
- 本轮未下载新大型数据或依赖；bulk download `NOT_RUN`，下载字节数 `0`，没有代理回退。净化下载策略仍是 direct-only、fail-closed。
- Git 中没有数据、标注、权重、原始大日志、账号、Cookie、token、代理信息或服务器私有路径。

## 训练曝光与绑定摘要

- S0：600 steps，2,400 frame exposures，600 current clips，32 个训练视频；来源 `gt=18,660`。
- B1/O1/O2 S1：各 300 steps，均为 1,200 frame exposures（current=225、replay=75），106 个视频键；来源均为 `gt=2,768`、`gt_replay=3,024`、`pl=15,817`。B1/O1/O2 使用同一个新 S0、同一 S1 current view、同一 replay view、同一 sampler plan 和同一评价阈值。
- B1 S2：300 steps，来源 `gt=1,170`、`gt_replay=2,528`、`pl=60,608`，`invalid_dt=0`。
- O2 S2：300 steps，来源 `gt=1,170`、`gt_replay=2,528`、`pl=57,790`，`motion_advance=54,756`，`invalid_dt=0`。
- O1 S1 的 motion audit：step 1 零初始化末层梯度/更新为 0；step 300 梯度范数 `0.000016230519`、更新范数 `0.020437998697`，说明运动参数确实进入训练更新。

checkpoint、prediction、metric、config、view、memory、sampler 和数据清单的完整哈希见 [`results.json`](results.json)；原始预测与原始 TrackEval 文件仅以 basename + SHA 引用，不上传仓库。

## 未完成或未运行

| 项目 | 状态 |
| --- | --- |
| O1 S2 | `NOT_RUN/null` |
| B0-FT | `NOT_RUN/null` |
| 完整 BDD100K benchmark | `NOT_RUN/null`；本报告是固定 pilot |
| TAO 完整训练/评价 | `NOT_RUN/null` |
| 新大型数据/依赖下载 | `NOT_RUN/null`；无代理回退 |
| 早期错误回放窗口 run | `discarded`；不计入最终结果 |

内容提交为 `2a95303827a1b90bed352fe8c5fb42043411a5c7`；最终远端 branch HEAD 在普通推送后的 handoff 中核对，未伪造远端 SHA。
