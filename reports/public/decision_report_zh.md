# C-MOT 执行结果（脱敏）

本报告只包含服务器上实际运行的结果。除 `asset_check` 外的结果均为固定 4 个独立夜间视频、每视频 80 帧的 `pilot` 前缀，不是完整 BDD100K benchmark。未运行项写为 `NOT_RUN/null`。

## 真实结果

| method | stage | steps | scope | old HOTA | new HOTA | seen HOTA | old IDF1 | new IDF1 | seen IDF1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OVTR-official-asset | S2_truck | 0 | asset_check | 0.1661646919 | 0.0240659639 | 0.1646718622 | 0.1211609436 | 0.0072727273 | 0.1199644817 |
| OVTR-realvideo-adapted | S0_ref | 20 | pilot | null | 0.1677079345 | 0.1677079345 | null | 0.1462974530 | 0.1462974530 |
| B0-FT | S1_pedestrian | 100 | pilot | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 0.0000000000 |
| B0-FT | S1_pedestrian | 300 | pilot | 0.0000000000 | 0.0211486373 | 0.0011437385 | 0.0000000000 | 0.0080370943 | 0.0004346537 |
| O1-DynAgnostic | S1_pedestrian | 100 | pilot | 0.1229134155 | 0.0449352037 | 0.1186962792 | 0.0847898847 | 0.0315789474 | 0.0819121864 |
| O1-DynAgnostic | S1_pedestrian | 300 | pilot | 0.0695475330 | 0.0569259084 | 0.0688649434 | 0.0377909402 | 0.0224016671 | 0.0369586736 |
| B1-PLR | S1_pedestrian | 100 | pilot | 0.1149905780 | 0.0530257794 | 0.1116394625 | 0.0725349856 | 0.0306122449 | 0.0702677637 |
| B1-PLR | S1_pedestrian | 300 | pilot | 0.0248632416 | 0.0682250325 | 0.0272082886 | 0.0082410692 | 0.0428134557 | 0.0101107769 |
| O2-CMOT | S1_pedestrian | 100 | pilot | 0.0900579083 | 0.0331565239 | 0.0869806271 | 0.0470630023 | 0.0094339623 | 0.0450279878 |
| O2-CMOT | S1_pedestrian | 300 | pilot | 0.0233749198 | 0.0208029088 | 0.0232358230 | 0.0046584612 | 0.0093312597 | 0.0049111706 |
| B1-PLR | S2_truck | 100 | pilot | 0.0921011543 | 0.0000000000 | 0.0911335782 | 0.0613424737 | 0.0000000000 | 0.0606980354 |
| B1-PLR | S2_truck | 300 | pilot | 0.0205260351 | 0.0000000000 | 0.0203103972 | 0.0046992287 | 0.0000000000 | 0.0046498605 |
| O2-CMOT | S2_truck | 100 | pilot | 0.0050135947 | 0.0000000000 | 0.0049609240 | 0.0005340694 | 0.0000000000 | 0.0005284587 |
| O2-CMOT | S2_truck | 300 | pilot | 0.0171596350 | 0.0000000000 | 0.0169793631 | 0.0029401369 | 0.0000000000 | 0.0029092491 |

## 来源与安全审计

- 本地使用了 clean OVTR 快照、已存在的 BDD100K MOT 图像/COCO box-track 标注和 TAO-Amodal/BDD 稀疏资源；原始目录只读。
- OVTR 官方本地权重只作 `asset_check`；S0 使用已审计的检测预训练权重，20 步后保存并严格重载。
- 批量下载真实字节数为 0。净化子环境关闭代理变量、curl 配置和代理回退；官方 BDD archive 的 DNS/直连路径无法确认，因此下载被阻断。
- public 报告只引用私有运行产物的 basename 和 SHA-256，不包含数据、权重、原始日志、服务器路径、账号或代理信息。

## 解释边界

- B1 与 O2 使用同一 S0、视频清单、PL/replay 视图、步数、阈值和评价器；O2 只增加类别条件运动分支。
- 本次 pilot 中训练步数增加通常没有带来指标提升；该事实不被改写为方法有效性结论。O1 class-agnostic 与 B0-FT 已作为独立控制。
- 当前实现和结果是可验证的工程 pilot，不宣称 TPAMI 最终方法或 SOTA。

## 未完成

- `S0_ref 300-step standalone training`: `NOT_RUN`（20-step S0 smoke was used as the shared legal local starting point.）
- `full BDD100K benchmark`: `NOT_RUN`（This execution reports a fixed 4-video pilot prefix.）
- `new bulk dataset download`: `NOT_RUN`（The purified direct route could not resolve the official BDD archive host; no proxy fallback was used.）

原始预测与 raw TrackEval 文件的私有 artifact 引用及 SHA-256 位于 `results.json`；它们因公开仓库禁止上传数据/大日志而不入 Git。
