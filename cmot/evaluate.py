"""Evaluate C-MOT JSONL predictions with the local official TrackEval code."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .manifest import write_json


GLOBAL_TO_BDD = {206: "car", 792: "pedestrian", 1122: "truck"}
GLOBAL_TO_NAME = {206: "car", 792: "pedestrian", 1122: "truck"}


def _read_predictions(path: str) -> Dict[str, List[dict]]:
    result: Dict[str, List[dict]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record["frame_key"]
            result[key] = record.get("predictions", [])
    return result


def _label(global_id: int, track_id: int, bbox: Sequence[float], score: Optional[float] = None) -> dict:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    result = {
        "id": int(track_id),
        "category": GLOBAL_TO_BDD[int(global_id)],
        "box2d": {"x1": x0, "y1": y0, "x2": x1, "y2": y1},
    }
    if score is not None:
        result["score"] = float(score)
    return result


def _write_trackeval_files(view_path: str, prediction_path: str, root: Path, max_videos: Optional[int] = None, max_frames_per_video: Optional[int] = None) -> dict:
    with open(view_path, "r", encoding="utf-8") as handle:
        view = json.load(handle)
    predictions = _read_predictions(prediction_path)
    videos = sorted([v for v in view.get("videos", []) if v.get("split") == "val"], key=lambda value: value["video_id"])
    if max_videos is not None:
        videos = videos[: int(max_videos)]
    gt_root = root / "gt"
    tracker_root = root / "trackers" / "cmot" / "data"
    gt_root.mkdir(parents=True, exist_ok=True)
    tracker_root.mkdir(parents=True, exist_ok=True)
    gt_frames = 0
    pred_frames = 0
    for video in videos:
        seq = video["video_id"]
        frame_by_index = {int(frame["frame_index"]): frame for frame in video.get("frames", [])}
        max_index = max(frame_by_index) if frame_by_index else -1
        if max_frames_per_video is not None:
            ordered_indices = sorted(frame_by_index)
            if ordered_indices:
                max_index = ordered_indices[min(int(max_frames_per_video), len(ordered_indices)) - 1]
        gt_data = []
        pred_data = []
        for frame_index in range(max_index + 1):
            frame = frame_by_index.get(frame_index)
            gt_labels = [] if frame is None else [
                _label(int(ann["global_semantic_id"]), int(ann["track_id"]), ann["bbox_xyxy"])
                for ann in frame.get("annotations", [])
                if int(ann["global_semantic_id"]) in GLOBAL_TO_BDD and ann.get("label_status") != "ignore"
            ]
            frame_key = None if frame is None else frame["frame_key"]
            pred_labels = [
                _label(int(pred["global_id"]), int(pred["track_id"]), pred["bbox_xyxy"], pred.get("score"))
                for pred in (predictions.get(frame_key, []) if frame_key is not None else [])
                if int(pred.get("global_id", -1)) in GLOBAL_TO_BDD
            ]
            gt_data.append({"index": frame_index, "labels": gt_labels})
            pred_data.append({"index": frame_index, "labels": pred_labels})
            gt_frames += 1
            pred_frames += 1
        (gt_root / (seq + ".json")).write_text(json.dumps(gt_data), encoding="utf-8")
        (tracker_root / (seq + ".json")).write_text(json.dumps(pred_data), encoding="utf-8")
    return {"videos": len(videos), "gt_frames": gt_frames, "pred_frames": pred_frames, "video_ids": [v["video_id"] for v in videos]}


def evaluate_trackeval(
    view_path: str,
    prediction_path: str,
    output_path: str,
    trackeval_root: str,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
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
        config = {
            "GT_FOLDER": str(temp_root / "gt"),
            "TRACKERS_FOLDER": str(temp_root / "trackers"),
            "TRACKERS_TO_EVAL": ["cmot"],
            "CLASSES_TO_EVAL": ["pedestrian", "car", "truck"],
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_FOLDER": str(temp_root / "trackeval_output"),
            "PRINT_CONFIG": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
        dataset = BDD100K(config)
        # This pinned TrackEval revision has a NumPy scalar bug when its BDD
        # super-category combiner is enabled for a custom three-class list.
        # Per-class HOTA/CLEAR/Identity computation is unchanged; we combine
        # the resulting class rows below with explicit detection weights.
        dataset.should_classes_combine = False
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
        rows = {}
        for global_id, class_name in GLOBAL_TO_NAME.items():
            per_class = combined[class_name]
            hota_data = per_class["HOTA"]
            clear_data = per_class["CLEAR"]
            identity_data = per_class["Identity"]
            count_data = per_class["Count"]
            rows[str(global_id)] = {
                "class_name": class_name,
                "hota": float(hota_data["HOTA(0)"]),
                "hota_mean": float(hota_data["HOTA"].mean()),
                "deta": float(hota_data["DetA"][0]),
                "assa": float(hota_data["AssA"][0]),
                "mota": float(clear_data["MOTA"]),
                "idf1": float(identity_data["IDF1"]),
                "gt_dets": int(count_data["GT_Dets"]),
                "pred_dets": int(count_data["Dets"]),
            }
        output = {
            "status": "OK",
            "evaluator": "TrackEval BDD100K + HOTA/CLEAR/Identity",
            "view": Path(view_path).name,
            "predictions": Path(prediction_path).name,
            "files": files,
            "messages": messages,
            "classes": rows,
            "combined": {
                "aggregation": "gt_detection_weighted_over_three_classes",
                "hota_mean": _weighted(rows, "hota_mean"),
                "mota": _weighted(rows, "mota"),
                "idf1": _weighted(rows, "idf1"),
            },
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
