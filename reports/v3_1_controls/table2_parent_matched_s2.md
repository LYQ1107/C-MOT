# 表2：S2 parent-matched QPL contribution（C2）

C2 与 R-QPLSEG-S2 的 parent 均绑定到 R-QPLSEG-S1；Delta 定义为 QPLSEG − parent-matched ER。

TP/FP/FN are reported at the first TrackEval threshold (HOTA@0.05); IDSW is cumulative.

| method | status | old HOTA | old IDF1 | old MOTA | truck HOTA | truck IDF1 | truck MOTA | truck TP@0.05 | truck FP@0.05 | truck FN@0.05 | truck IDSW | truck recall | all HOTA | all IDF1 | all MOTA | DetA | AssA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R-QPLSEG-PARENT-ER-S2 | COMPLETE | 0.201403422 | 0.188836533 | 0.099540008 | 0.144554671 | 0.166769611 | -0.175087108 | 143 | 328 | 1005 | 0 | 0.124564460 | 0.182453838 | 0.181480892 | 0.007997636 | 0.094338986 | 0.362737604 |
| R-QPLSEG-S2 | COMPLETE | 0.211701580 | 0.198730758 | 0.109539930 | 0.215893461 | 0.254760341 | 0.011324042 | 203 | 172 | 945 | 0 | 0.176829268 | 0.213098873 | 0.217407286 | 0.076801300 | 0.111582336 | 0.416097035 |
| Delta_QPLSEG_minus_PARENT_ER | — | 0.010298158 | 0.009894225 | 0.009999922 | 0.071338790 | 0.087990731 | 0.186411150 | 60.000000000 | -156.000000000 | -60.000000000 | 0.000000000 | 0.052264808 | 0.030645035 | 0.035926394 | 0.068803665 | 0.017243350 | 0.053359431 |

- parent SHA equal: `true`.
