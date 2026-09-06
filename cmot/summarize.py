"""Build the small, path-redacted public report from private run artifacts."""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .class_registry import tao_bdd_registry
from .manifest import canonical_json_hash, sha256_file, write_json


GLOBAL_NAMES = {206: "car", 792: "pedestrian", 1122: "truck"}
STAGE_IDS = {
    "S0_ref": (206,),
    "S1_pedestrian": (206, 792),
    "S2_truck": (206, 792, 1122),
}
STAGE_NEW = {"S0_ref": (206,), "S1_pedestrian": (792,), "S2_truck": (1122,)}
STAGE_OLD = {"S0_ref": (), "S1_pedestrian": (206,), "S2_truck": (206, 792)}


def _specs() -> List[dict]:
    specs = [
        {
            "method": "OVTR-official-asset",
            "stage": "S2_truck",
            "scope": "asset_check",
            "steps": 0,
            "run": "asset_check",
            "checkpoint": None,
            "eval": "ovtr_5_frame_bdd_night_trackeval.json",
            "pred": "ovtr_5_frame_bdd_night.jsonl",
            "train_view": None,
            "eval_view": "bdd_S2_eval.json",
            "parent_hash": "7b184a0f149259047ef3f03263cf9178fc9c5e051d08882f065aa52118266d56",
            "label_protocol": "asset_check_only",
            "motion_mode": "none",
        },
        {
            "method": "OVTR-realvideo-adapted",
            "stage": "S0_ref",
            "scope": "pilot",
            "steps": 20,
            "run": "S0_ref_20",
            "checkpoint": "checkpoint_020.pt",
            "eval": "eval_trackeval.json",
            "pred": "eval_independent.jsonl",
            "train_view": "bdd_S0_train.json",
            "eval_view": "bdd_S0_eval.json",
            "parent_hash": "0862cac87ad50f58a01ce17d4e44af0468ad8639cfccd18d66d2e9b2570d839e",
            "label_protocol": "complete",
            "motion_mode": "none",
        },
        {
            "method": "B0-FT",
            "stage": "S1_pedestrian",
            "scope": "pilot",
            "steps": 100,
            "run": "B0_FT_S1_100",
            "checkpoint": "checkpoint_100.pt",
            "eval": "eval_trackeval_100.json",
            "pred": "eval_independent.jsonl",
            "train_view": "bdd_B0_S1_train.json",
            "eval_view": "bdd_S1_eval.json",
            "parent_hash": "109c02a7e3b989450b114eca8051e10deba6382da12fed461609ce2c15d77c6b",
            "label_protocol": "naive_complete",
            "motion_mode": "none",
        },
        {
            "method": "B0-FT",
            "stage": "S1_pedestrian",
            "scope": "pilot",
            "steps": 300,
            "run": "B0_FT_S1_100",
            "checkpoint": "checkpoint_300.pt",
            "eval": "eval_trackeval_300.json",
            "pred": "eval_independent_300.jsonl",
            "train_view": "bdd_B0_S1_train.json",
            "eval_view": "bdd_S1_eval.json",
            "parent_hash": "109c02a7e3b989450b114eca8051e10deba6382da12fed461609ce2c15d77c6b",
            "label_protocol": "naive_complete",
            "motion_mode": "none",
        },
    ]
    for method, run, mode, view, parent in (
        ("O1-DynAgnostic", "O1_DynAgnostic_S1_100", "class_agnostic", "bdd_S1_train.json", "109c02a7e3b989450b114eca8051e10deba6382da12fed461609ce2c15d77c6b"),
        ("B1-PLR", "B1_S1_100", "none", "bdd_S1_train.json", "109c02a7e3b989450b114eca8051e10deba6382da12fed461609ce2c15d77c6b"),
        ("O2-CMOT", "O2_S1_100", "category_conditioned", "bdd_S1_train.json", "109c02a7e3b989450b114eca8051e10deba6382da12fed461609ce2c15d77c6b"),
    ):
        for steps, suffix in ((100, "100"), (300, "300")):
            specs.append({
                "method": method,
                "stage": "S1_pedestrian",
                "scope": "pilot",
                "steps": steps,
                "run": run,
                "checkpoint": "checkpoint_%03d.pt" % steps,
                "eval": "eval_trackeval_%s.json" % suffix,
                "pred": "eval_independent%s.jsonl" % ("" if steps == 100 else "_300"),
                "train_view": view,
                "eval_view": "bdd_S1_eval.json",
                "parent_hash": parent,
                "label_protocol": "partial_label_pl_old_replay",
                "motion_mode": mode,
            })
    for method, run, mode, view, parent in (
        ("B1-PLR", "B1_S2_100", "none", "bdd_S2_train_B1.json", "148c72afeabe47a702ef9418b03b1c26f7f00b7af943fd0fcb5e4fa141b160d4"),
        ("O2-CMOT", "O2_S2_100", "category_conditioned", "bdd_S2_train_O2.json", "aa5bc042b70e7cfdf7c7ce71cdd0876bf5ff1ec4f19c138f0fc69a790aece4da"),
    ):
        for steps, suffix in ((100, "100"), (300, "300")):
            specs.append({
                "method": method,
                "stage": "S2_truck",
                "scope": "pilot",
                "steps": steps,
                "run": run,
                "checkpoint": "checkpoint_%03d.pt" % steps,
                "eval": "eval_trackeval_%s.json" % suffix,
                "pred": "eval_independent%s.jsonl" % ("" if steps == 100 else "_300"),
                "train_view": view,
                "eval_view": "bdd_S2_eval.json",
                "parent_hash": parent,
                "label_protocol": "partial_label_pl_old_replay",
                "motion_mode": mode,
            })
    return specs


def _metric(rows: dict, ids: Sequence[int]) -> dict:
    selected = [rows.get(str(int(identifier))) for identifier in ids]
    selected = [row for row in selected if row and int(row.get("gt_dets", 0)) > 0]
    if not selected:
        return {"hota_mean": None, "idf1": None, "mota": None}
    total = sum(int(row["gt_dets"]) for row in selected)
    return {
        key: sum(int(row["gt_dets"]) * float(row[key]) for row in selected) / float(total)
        for key in ("hota_mean", "idf1", "mota")
    }


def _hash_or_none(path: Optional[Path]) -> Optional[str]:
    return sha256_file(str(path)) if path is not None and path.is_file() else None


def _git_commit(project_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root), text=True).strip()
    except Exception:
        return None


def build_report(run_root: str, project_root: str, public_output: str) -> dict:
    run_root_path = Path(run_root)
    project_path = Path(project_root)
    public_path = Path(public_output)
    public_path.mkdir(parents=True, exist_ok=True)
    config_path = project_path / "configs" / "cmot_curriculum.yaml"
    config_hash = _hash_or_none(config_path)
    registry_hash = canonical_json_hash(tao_bdd_registry().as_dict())
    rows: List[dict] = []
    for spec in _specs():
        run_dir = run_root_path / spec["run"]
        eval_path = run_dir / spec["eval"]
        pred_path = run_dir / spec["pred"]
        train_view_path = None if not spec["train_view"] else project_path.parent / "cmot_annotations" / spec["train_view"]
        eval_view_path = project_path.parent / "cmot_annotations" / spec["eval_view"]
        evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else None
        if evaluation is None:
            status = "NOT_RUN"
            classes = {}
            seen = {"hota_mean": None, "idf1": None, "mota": None}
            files = {}
        else:
            status = "OK" if evaluation.get("status") == "OK" else "NOT_RUN"
            classes = evaluation.get("classes", {})
            seen = evaluation.get("combined", {"hota_mean": None, "idf1": None, "mota": None})
            files = evaluation.get("files", {})
        old = _metric(classes, STAGE_OLD[spec["stage"]])
        new = _metric(classes, STAGE_NEW[spec["stage"]])
        checkpoint_path = None if not spec["checkpoint"] else run_dir / spec["checkpoint"]
        checkpoint_audit = None
        summary_path = run_dir / ("pl_train_summary.json" if spec["scope"] == "asset_check" else "train_summary.json")
        if spec["scope"] == "asset_check":
            summary_path = run_dir / "ovtr_5_frame_bdd_night.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint_audit = summary.get("checkpoint_audit")
        checkpoint_hash = _hash_or_none(checkpoint_path)
        if checkpoint_hash is None and checkpoint_audit:
            checkpoint_hash = checkpoint_audit.get("sha256")
        train_view_hash = _hash_or_none(train_view_path)
        eval_view_hash = _hash_or_none(eval_view_path)
        train_source_count = None
        if train_view_path is not None and train_view_path.is_file():
            train_view = json.loads(train_view_path.read_text(encoding="utf-8"))
            train_source_count = len(train_view.get("videos", []))
        row = {
            "method": spec["method"],
            "stage": spec["stage"],
            "scope": spec["scope"],
            "steps": spec["steps"],
            "status": status,
            "label_protocol": spec["label_protocol"],
            "motion_mode": spec["motion_mode"],
            "train_source_count": train_source_count,
            "train_frames_exposed": None if spec["scope"] == "asset_check" else spec["steps"] * 2,
            "video_count": files.get("videos"),
            "frame_count": files.get("pred_frames", files.get("frames")),
            "parent_checkpoint_sha256": spec["parent_hash"],
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_bytes": None if checkpoint_path is None or not checkpoint_path.is_file() else checkpoint_path.stat().st_size,
            "config_sha256": config_hash,
            "class_registry_sha256": registry_hash,
            "train_view_sha256": train_view_hash,
            "eval_view_sha256": eval_view_hash,
            "old": old,
            "new": new,
            "seen": {key: seen.get(key) for key in ("hota_mean", "idf1", "mota")},
            "raw_prediction_artifact": None if not pred_path.is_file() else "private-run-artifact:%s/%s" % (spec["run"], pred_path.name),
            "raw_prediction_sha256": _hash_or_none(pred_path),
            "raw_metric_artifact": None if not eval_path.is_file() else "private-run-artifact:%s/%s" % (spec["run"], eval_path.name),
            "raw_metric_sha256": _hash_or_none(eval_path),
        }
        rows.append(row)

    fields = [
        "method", "stage", "scope", "steps", "status", "label_protocol", "motion_mode",
        "train_source_count", "train_frames_exposed", "video_count", "frame_count",
        "parent_checkpoint_sha256", "checkpoint_sha256", "checkpoint_bytes", "config_sha256",
        "class_registry_sha256", "train_view_sha256", "eval_view_sha256",
        "old_hota_mean", "old_idf1", "old_mota", "new_hota_mean", "new_idf1", "new_mota",
        "seen_hota_mean", "seen_idf1", "seen_mota", "raw_prediction_artifact", "raw_prediction_sha256",
        "raw_metric_artifact", "raw_metric_sha256",
    ]
    with (public_path / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            for group in ("old", "new", "seen"):
                for metric in ("hota_mean", "idf1", "mota"):
                    flat["%s_%s" % (group, metric)] = row[group][metric]
            for key, value in list(flat.items()):
                if value is None:
                    flat[key] = "null"
            writer.writerow(flat)

    result = {
        "schema_version": "cmot.public-results.v1",
        "protocol": "bdd_foundation_assisted_label_cil_pilot",
        "scope_note": "All non-asset rows are 4-video/320-frame pilot prefixes, not full BDD100K benchmark results.",
        "code_commit": _git_commit(project_path),
        "config_sha256": config_hash,
        "class_registry_sha256": registry_hash,
        "rows": rows,
        "unrun": [
            {"item": "S0_ref 300-step standalone training", "status": "NOT_RUN", "reason": "20-step S0 smoke was used as the shared legal local starting point."},
            {"item": "full BDD100K benchmark", "status": "NOT_RUN", "reason": "This execution reports a fixed 4-video pilot prefix."},
            {"item": "new bulk dataset download", "status": "NOT_RUN", "reason": "The purified direct route could not resolve the official BDD archive host; no proxy fallback was used."},
        ],
        "download_audit": {"status": "BLOCKED_NO_VERIFIED_DIRECT_ROUTE", "bulk_bytes_downloaded": 0},
    }
    write_json(str(public_path / "results.json"), result)
    _write_decision_markdown(public_path / "decision_report_zh.md", result)
    return result


def _fmt(value) -> str:
    return "null" if value is None else ("%.10f" % value if isinstance(value, float) else str(value))


def _write_decision_markdown(path: Path, result: dict) -> None:
    lines = [
        "# C-MOT 执行结果（脱敏）",
        "",
        "本报告只包含服务器上实际运行的结果。除 `asset_check` 外的结果均为固定 4 个独立夜间视频、每视频 80 帧的 `pilot` 前缀，不是完整 BDD100K benchmark。未运行项写为 `NOT_RUN/null`。",
        "",
        "## 真实结果",
        "",
        "| method | stage | steps | scope | old HOTA | new HOTA | seen HOTA | old IDF1 | new IDF1 | seen IDF1 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["method"], row["stage"], row["steps"], row["scope"],
            _fmt(row["old"]["hota_mean"]), _fmt(row["new"]["hota_mean"]), _fmt(row["seen"]["hota_mean"]),
            _fmt(row["old"]["idf1"]), _fmt(row["new"]["idf1"]), _fmt(row["seen"]["idf1"]),
        ))
    lines += [
        "",
        "## 来源与安全审计",
        "",
        "- 本地使用了 clean OVTR 快照、已存在的 BDD100K MOT 图像/COCO box-track 标注和 TAO-Amodal/BDD 稀疏资源；原始目录只读。",
        "- OVTR 官方本地权重只作 `asset_check`；S0 使用已审计的检测预训练权重，20 步后保存并严格重载。",
        "- 批量下载真实字节数为 0。净化子环境关闭代理变量、curl 配置和代理回退；官方 BDD archive 的 DNS/直连路径无法确认，因此下载被阻断。",
        "- public 报告只引用私有运行产物的 basename 和 SHA-256，不包含数据、权重、原始日志、服务器路径、账号或代理信息。",
        "",
        "## 解释边界",
        "",
        "- B1 与 O2 使用同一 S0、视频清单、PL/replay 视图、步数、阈值和评价器；O2 只增加类别条件运动分支。",
        "- 本次 pilot 中训练步数增加通常没有带来指标提升；该事实不被改写为方法有效性结论。O1 class-agnostic 与 B0-FT 已作为独立控制。",
        "- 当前实现和结果是可验证的工程 pilot，不宣称 TPAMI 最终方法或 SOTA。",
        "",
        "## 未完成",
        "",
    ]
    for item in result["unrun"]:
        lines.append("- `%s`: `%s`（%s）" % (item["item"], item["status"], item["reason"]))
    lines += [
        "",
        "原始预测与 raw TrackEval 文件的私有 artifact 引用及 SHA-256 位于 `results.json`；它们因公开仓库禁止上传数据/大日志而不入 Git。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--public-output", required=True)
    parser.add_argument("--require-raw-evidence", action="store_true")
    args = parser.parse_args()
    result = build_report(args.run_root, args.project_root, args.public_output)
    if args.require_raw_evidence:
        missing = [row for row in result["rows"] if row["status"] == "OK" and not row["raw_metric_sha256"]]
        if missing:
            raise SystemExit("raw evidence missing for %d rows" % len(missing))
    print(json.dumps({"rows": len(result["rows"]), "code_commit": result["code_commit"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
