"""COOLer-compatible BDD100K TrackEval adapter.

The local TrackEval adapter already implements the BDD ignore/crowd file
layout and the 19-threshold HOTA calculation.  This wrapper adds the
protocol-required per-class macro values and keeps raw fractions separate
from the public 0–100 presentation.
"""

from pathlib import Path
from typing import Any, Mapping, Optional

from ..evaluate import evaluate_trackeval
from ..manifest import sha256_file, write_json


def _average(rows, key: str):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else sum(values) / float(len(values))


def _percent(value):
    return None if value is None else float(value) * 100.0


def evaluate_cooler_bdd(
    view_path: str,
    prediction_path: str,
    output_path: str,
    trackeval_root: str,
    expected_binding: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Evaluate a full validation view with no threshold/model selection."""
    raw = evaluate_trackeval(
        view_path,
        prediction_path,
        output_path + ".trackeval.json",
        trackeval_root,
        expected_binding=expected_binding,
    )
    with Path(view_path).open("r", encoding="utf-8") as handle:
        view = __import__("json").load(handle)
    active_ids = [int(value) for value in view.get("active_global_ids", [])]
    class_rows = {str(gid): raw["classes"][str(gid)] for gid in active_ids if str(gid) in raw.get("classes", {})}
    rows = list(class_rows.values())
    metric_names = ("MOTA", "HOTA_mean", "IDF1", "DetA_mean", "AssA_mean")
    macro = {name: _average(rows, name) for name in metric_names}
    combined = {name: raw.get("combined", {}).get(name) for name in metric_names}
    public = {
        name: {
            "raw_fraction": {metric: row.get(metric) for metric in metric_names},
            "percent": {metric: _percent(row.get(metric)) for metric in metric_names},
        }
        for name, row in class_rows.items()
    }
    result = {
        "schema_version": "cmot.cooler_compat.metrics.v1",
        "status": raw.get("status", "OK"),
        "evaluator": "TrackEval BDD100K + HOTA/CLEAR/Identity",
        "hota_definition": "mean over TrackEval's 19 alpha thresholds",
        "active_global_ids": active_ids,
        "per_class": public,
        "macro_seen": {
            "raw_fraction": macro,
            "percent": {metric: _percent(value) for metric, value in macro.items()},
        },
        "overall": {
            "aggregation": raw.get("combined", {}).get("aggregation"),
            "raw_fraction": combined,
            "percent": {metric: _percent(value) for metric, value in combined.items()},
        },
        "counts": raw.get("files", {}),
        "checkpoint_binding": raw.get("checkpoint_binding", {}),
        "prediction_sha256": sha256_file(prediction_path),
        "raw_trackeval_basename": Path(output_path + ".trackeval.json").name,
    }
    write_json(output_path, result)
    return result


__all__ = ["evaluate_cooler_bdd"]
