"""Evaluate C-MOT JSONL predictions with the local official TrackEval code."""

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .manifest import sha256_file, write_json


GLOBAL_TO_BDD = {206: "car", 792: "pedestrian", 1122: "truck"}
GLOBAL_TO_NAME = {206: "car", 792: "pedestrian", 1122: "truck"}


def _read_predictions(path: str) -> Tuple[Dict[str, List[dict]], dict]:
    result: Dict[str, List[dict]] = {}
    metadata = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "metadata":
                metadata.update(record.get("metadata", record))
                continue
            key = record.get("frame_key")
            if not key:
                raise ValueError("prediction line %d has no frame_key" % line_number)
            if key in result:
                raise ValueError("duplicate prediction frame_key: %s" % key)
            seen_ids = set()
            for prediction in record.get("predictions", []):
                track_id = int(prediction["track_id"])
                if track_id in seen_ids:
                    raise ValueError("duplicate track_id %s on frame %s" % (track_id, key))
                seen_ids.add(track_id)
                bbox = prediction.get("bbox_xyxy")
                if bbox is None or len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
                    raise ValueError("invalid prediction box on frame %s" % key)
                if float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]):
                    raise ValueError("non-positive prediction box on frame %s" % key)
                if not math.isfinite(float(prediction.get("score", 0.0))):
                    raise ValueError("invalid prediction score on frame %s" % key)
            result[key] = record.get("predictions", [])
    return result, metadata


def _label(global_id: int, track_id: int, bbox: Sequence[float], score: Optional[float] = None, crowd: bool = False) -> dict:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    result = {
        "id": int(track_id),
        "category": GLOBAL_TO_BDD[int(global_id)],
        "box2d": {"x1": x0, "y1": y0, "x2": x1, "y2": y1},
    }
    if score is not None:
        result["score"] = float(score)
    if crowd:
        result["attributes"] = {"Crowd": True}
    return result


def _write_trackeval_files(
    view_path: str,
    prediction_path: str,
    root: Path,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
) -> dict:
    with open(view_path, "r", encoding="utf-8") as handle:
        view = json.load(handle)
    predictions, prediction_metadata = _read_predictions(prediction_path)
    videos = sorted([v for v in view.get("videos", []) if v.get("split") == "val"], key=lambda value: value["video_id"])
    if max_videos is not None:
        videos = videos[: int(max_videos)]
    gt_root = root / "gt"
    tracker_root = root / "trackers" / "cmot" / "data"
    gt_root.mkdir(parents=True, exist_ok=True)
    tracker_root.mkdir(parents=True, exist_ok=True)
    gt_frames = 0
    pred_frames = 0
    gt_counts = {str(gid): 0 for gid in GLOBAL_TO_BDD}
    pred_counts = {str(gid): 0 for gid in GLOBAL_TO_BDD}
    expected_keys = set()
    for video in videos:
        seq = video["video_id"]
        frames = sorted(video.get("frames", []), key=lambda frame: (int(frame["frame_index"]), frame["frame_key"]))
        if max_frames_per_video is not None:
            frames = frames[:int(max_frames_per_video)]
        frame_keys = [frame["frame_key"] for frame in frames]
        expected_keys.update(frame_keys)
        gt_data = []
        pred_data = []
        for output_index, frame in enumerate(frames):
            gt_labels = []
            for ann_index, ann in enumerate(frame.get("annotations", [])):
                gid = int(ann["global_semantic_id"])
                if gid not in GLOBAL_TO_BDD:
                    continue
                crowd = bool(int(ann.get("iscrowd", 0))) or ann.get("label_status") == "ignore"
                gt_labels.append(_label(gid, int(ann["track_id"]), ann["bbox_xyxy"], crowd=crowd))
                if not crowd:
                    gt_counts[str(gid)] += 1
            for ignore_index, region in enumerate(frame.get("ignore_regions", [])):
                values = region.get("bbox_xyxy", [])
                if len(values) == 4:
                    # BDD100K's adapter treats these distractor labels as
                    # crowd-ignore regions and removes overlapping predictions.
                    gt_labels.append(_label(206, -1000000 - ignore_index, values, crowd=True))
            frame_key = frame["frame_key"]
            pred_labels = [
                _label(int(pred["global_id"]), int(pred["track_id"]), pred["bbox_xyxy"], pred.get("score"))
                for pred in predictions.get(frame_key, [])
                if int(pred.get("global_id", -1)) in GLOBAL_TO_BDD
            ]
            for pred in predictions.get(frame_key, []):
                gid = int(pred.get("global_id", -1))
                if gid in GLOBAL_TO_BDD:
                    pred_counts[str(gid)] += 1
            gt_data.append({"index": output_index, "labels": gt_labels})
            pred_data.append({"index": output_index, "labels": pred_labels})
            gt_frames += 1
            pred_frames += 1
        (gt_root / (seq + ".json")).write_text(json.dumps(gt_data), encoding="utf-8")
        (tracker_root / (seq + ".json")).write_text(json.dumps(pred_data), encoding="utf-8")
    extra_keys = sorted(set(predictions) - expected_keys)
    missing_keys = sorted(expected_keys - set(predictions))
    if extra_keys or missing_keys:
        raise ValueError(
            "prediction frame set mismatch: missing=%s extra=%s"
            % (missing_keys[:3], extra_keys[:3])
        )
    return {
        "videos": len(videos),
        "gt_frames": gt_frames,
        "pred_frames": pred_frames,
        "video_ids": [v["video_id"] for v in videos],
        "expected_frame_keys": len(expected_keys),
        "gt_detections": gt_counts,
        "pred_detections": pred_counts,
        "prediction_metadata": prediction_metadata,
    }


def evaluate_trackeval(
    view_path: str,
    prediction_path: str,
    output_path: str,
    trackeval_root: str,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    expected_binding: Optional[Mapping[str, object]] = None,
) -> dict:
    """Run HOTA, CLEAR and Identity on exactly the selected video list."""
    trackeval_root = str(Path(trackeval_root).resolve())
    if trackeval_root not in sys.path:
        sys.path.insert(0, trackeval_root)
    from trackeval import Evaluator
    from trackeval.datasets import BDD100K
    from trackeval.metrics import CLEAR, HOTA, Identity

    temp_root = Path(tempfile.mkdtemp(prefix="cmot_trackeval_"))
    try:
        files = _write_trackeval_files(view_path, prediction_path, temp_root, max_videos=max_videos, max_frames_per_video=max_frames_per_video)
        with open(view_path, "r", encoding="utf-8") as handle:
            view_payload = json.load(handle)
        active_ids = [int(value) for value in view_payload.get("active_global_ids", GLOBAL_TO_BDD)]
        active_names = [GLOBAL_TO_NAME[value] for value in active_ids if value in GLOBAL_TO_NAME]
        if not active_names:
            raise ValueError("evaluation view has no active classes")
        config = {
            "GT_FOLDER": str(temp_root / "gt"),
            "TRACKERS_FOLDER": str(temp_root / "trackers"),
            "TRACKERS_TO_EVAL": ["cmot"],
            "CLASSES_TO_EVAL": active_names,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_FOLDER": str(temp_root / "trackeval_output"),
            "PRINT_CONFIG": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
        dataset = BDD100K(config)
        # Keep the official BDD class combination enabled.  The three-class
        # slice is the protocol's declared active set, not a hand-weighted
        # substitute for TrackEval's class combiner.
        dataset.should_classes_combine = True
        dataset.use_super_categories = False
        evaluator = Evaluator({
            "PRINT_CONFIG": False,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": True,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
            "TIME_PROGRESS": False,
            "BREAK_ON_ERROR": True,
        })
        result, messages = evaluator.evaluate([dataset], [HOTA(), CLEAR(), Identity()])
        combined = result["BDD100K"]["cmot"]["COMBINED_SEQ"]
        def _metric_row(hota_data, clear_data, identity_data, count_data):
            def _array(name, cast=float):
                value = hota_data.get(name)
                if value is None:
                    return None
                return [cast(item) for item in value.tolist()]
            return {
                # HOTA(0) is retained under an explicit name; headline HOTA
                # is the mandated mean over all 19 thresholds.
                "HOTA_mean": float(hota_data["HOTA"].mean()),
                "hota_at_005": float(hota_data["HOTA(0)"]),
                "DetA_mean": float(hota_data["DetA"].mean()),
                "AssA_mean": float(hota_data["AssA"].mean()),
                "HOTA_thresholds": _array("HOTA"),
                "DetA_thresholds": _array("DetA"),
                "AssA_thresholds": _array("AssA"),
                "TP": _array("HOTA_TP", int),
                "FP": _array("HOTA_FP", int),
                "FN": _array("HOTA_FN", int),
                "IDSW": int(clear_data["IDSW"]),
                "IDTP": int(identity_data["IDTP"]),
                "IDFP": int(identity_data["IDFP"]),
                "IDFN": int(identity_data["IDFN"]),
                "MOTA": float(clear_data["MOTA"]),
                "IDF1": float(identity_data["IDF1"]),
                "gt_dets": int(count_data["GT_Dets"]),
                "pred_dets": int(count_data["Dets"]),
            }
        rows = {}
        for global_id, class_name in GLOBAL_TO_NAME.items():
            if class_name not in active_names:
                continue
            per_class = combined[class_name]
            hota_data = per_class["HOTA"]
            clear_data = per_class["CLEAR"]
            identity_data = per_class["Identity"]
            count_data = per_class["Count"]
            rows[str(global_id)] = {"class_name": class_name, **_metric_row(hota_data, clear_data, identity_data, count_data)}
        official_det = combined["cls_comb_det_av"]
        official_class = combined["cls_comb_cls_av"]
        def _combined_row(source):
            hota_data = source["HOTA"]
            clear_data = source["CLEAR"]
            identity_data = source["Identity"]
            count_data = source["Count"]
            return _metric_row(hota_data, clear_data, identity_data, count_data)
        binding = dict(expected_binding or {})
        for key in (
            "checkpoint_sha256",
            "teacher_checkpoint_sha256",
            "resolved_config_sha256",
            "view_manifest_hash",
        ):
            if key in files["prediction_metadata"]:
                binding[key] = files["prediction_metadata"][key]
        binding.setdefault("prediction_sha256", sha256_file(prediction_path))
        binding_mismatches = {}
        for key, expected in (expected_binding or {}).items():
            actual = binding.get(key)
            if actual != expected:
                binding_mismatches[str(key)] = {"expected": expected, "actual": actual}
        if binding_mismatches:
            raise ValueError("prediction/checkpoint/view binding mismatch: %s" % binding_mismatches)

        output = {
            "status": "OK",
            "evaluator": "TrackEval BDD100K + HOTA/CLEAR/Identity",
            "view": Path(view_path).name,
            "predictions": Path(prediction_path).name,
            "files": files,
            "messages": messages,
            "classes": rows,
            "combined": {
                "aggregation": "TrackEval_cls_comb_det_av",
                **_combined_row(official_det),
            },
            "combined_class_average": {
                "aggregation": "TrackEval_cls_comb_cls_av",
                **_combined_row(official_class),
            },
            "checkpoint_binding": binding,
            "binding_verified": True,
            "prediction_sha256": sha256_file(prediction_path),
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    write_json(output_path, output)
    return output


def _weighted(rows: Mapping[str, dict], key: str) -> float:
    total = sum(float(row.get("gt_dets", 0)) for row in rows.values())
    if total <= 0:
        return 0.0
    return float(sum(float(row.get("gt_dets", 0)) * float(row[key]) for row in rows.values()) / total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trackeval-root", required=True)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-frames-per-video", type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate_trackeval(args.view, args.predictions, args.output, args.trackeval_root, args.max_videos, args.max_frames_per_video), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
