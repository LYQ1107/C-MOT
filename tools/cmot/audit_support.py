"""Audit per-class support without confusing frames with independent videos."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from cmot.manifest import canonical_json_hash, read_json, write_json


CLASS_IDS = {"car": 206, "pedestrian": 792, "truck": 1122}


def _duration_and_span(observations):
    by_track = defaultdict(list)
    for timestamp, track_id in observations:
        by_track[str(track_id)].append(float(timestamp))
    visible_seconds = 0.0
    span_seconds = 0.0
    pairs = 0
    for values in by_track.values():
        values = sorted(set(values))
        if len(values) > 1:
            visible_seconds += sum(max(0.0, values[index + 1] - values[index]) for index in range(len(values) - 1))
            span_seconds += max(values) - min(values)
            pairs += sum(1 for index in range(len(values) - 1) if values[index + 1] > values[index])
    return visible_seconds, span_seconds, pairs


def audit_support(manifest_path: str, delta_t_max: float = 5.0) -> dict:
    manifest = read_json(manifest_path)
    result = {
        "schema_version": "cmot.support-audit.v1",
        "manifest_basename": Path(manifest_path).name,
        "manifest_hash": manifest.get("manifest_hash") or canonical_json_hash(manifest),
        "delta_t_max_s": float(delta_t_max),
        "classes": {},
    }
    for class_name, global_id in CLASS_IDS.items():
        per_split = {}
        for split in ("train", "dev", "val", "eval"):
            videos = [video for video in manifest.get("videos", []) if video.get("split") == split]
            if not videos:
                continue
            source_videos = set()
            tracks = set()
            observations = []
            source_seconds = Counter()
            generated = 0
            missing = 0
            total_frames = 0
            competing_windows = set()
            for video in videos:
                source_id = str(video.get("source_video_id", video.get("video_id")))
                source_videos.add(source_id)
                track_timestamps = defaultdict(list)
                for frame in video.get("frames", []):
                    total_frames += 1
                    timestamp = frame.get("timestamp_s")
                    if timestamp is None:
                        timestamp = float(frame.get("frame_index", 0))
                    anns = [ann for ann in frame.get("annotations", []) if int(ann.get("global_semantic_id", -1)) == global_id]
                    if not anns:
                        missing += 1
                    unique_tracks = set()
                    for ann in anns:
                        track_id = int(ann["track_id"])
                        tracks.add((source_id, track_id))
                        track_timestamps[track_id].append(float(timestamp))
                        observations.append((float(timestamp), "%s:%s" % (source_id, track_id)))
                        source_seconds[source_id] += 1.0
                        generated += int(bool(ann.get("generated", False)))
                        unique_tracks.add(track_id)
                    if len(unique_tracks) > 1:
                        competing_windows.add((source_id, frame.get("frame_key")))
                # The pair count is calculated below from timestamps and does
                # not use frame count as a proxy for independent identities.
            visible_seconds, span_seconds, pair_count = _duration_and_span(observations)
            valid_pairs = 0
            by_track = defaultdict(list)
            for timestamp, track_id in observations:
                by_track[track_id].append(timestamp)
            for values in by_track.values():
                values = sorted(set(values))
                valid_pairs += sum(1 for a, b in zip(values, values[1:]) if 0.0 < b - a <= float(delta_t_max))
            total_exposure = float(sum(source_seconds.values()))
            shares = sorted((value / total_exposure for value in source_seconds.values()), reverse=True) if total_exposure else []
            effective = None if not shares else 1.0 / sum(share * share for share in shares)
            per_split[split] = {
                "independent_source_video_count": len(source_videos),
                "track_count": len(tracks),
                "observation_box_count": len(observations),
                "effective_visible_seconds": visible_seconds,
                "track_span_seconds": span_seconds,
                "supervised_same_identity_pairs": valid_pairs,
                "same_class_competing_windows": len(competing_windows),
                "generated_ratio": None if not observations else float(generated) / len(observations),
                "annotation_missing_frame_ratio": None if not total_frames else float(missing) / total_frames,
                "top1_source_exposure": None if not shares else shares[0],
                "top5_source_exposure": None if not shares else sum(shares[:5]),
                "effective_video_exposure": effective,
                "annotation_scope_note": "missing frames are not assumed background; inspect label_scope/unannotated metadata",
            }
        result["classes"][class_name] = {"global_semantic_id": global_id, "by_split": per_split}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta-t-max", type=float, default=5.0)
    args = parser.parse_args()
    value = audit_support(args.manifest, args.delta_t_max)
    write_json(args.output, value)
    print(json.dumps({"status": "OK", "classes": len(value["classes"])}, sort_keys=True))


if __name__ == "__main__":
    main()
