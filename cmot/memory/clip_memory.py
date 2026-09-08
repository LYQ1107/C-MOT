"""Short-clip replay memory with legal GT snapshots and reachable-byte audit."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ..manifest import canonical_json_hash, read_json, write_json
from .selection import stratified_reservoir


def _actual_media(media: Mapping[str, Any], frame: Mapping[str, Any]) -> dict:
    value = dict(media or {})
    path = value.get("path")
    if path and Path(path).is_file():
        value["bytes"] = int(Path(path).stat().st_size)
    if not path or not Path(str(path)).is_file():
        return {}
    value["path"] = str(Path(path).resolve())
    return value


class ClipReplayMemory:
    """A fixed-content replay buffer shared by all method variants."""

    SCHEMA_VERSION = "cmot.clip-memory.v2"

    def __init__(self, protocol_hash: str, registry_hash: str, clips: Optional[Iterable[Mapping]] = None):
        self.protocol_hash = str(protocol_hash)
        self.registry_hash = str(registry_hash)
        self.clips = [dict(value) for value in (clips or [])]
        self.version = canonical_json_hash(self._payload())

    def _payload(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "protocol_hash": self.protocol_hash,
            "registry_hash": self.registry_hash,
            "clips": self.clips,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]):
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("repair_v2 requires cmot.clip-memory.v2")
        memory = cls(payload["protocol_hash"], payload.get("registry_hash", ""), payload.get("clips", []))
        if payload.get("version") and payload["version"] != memory.version:
            raise ValueError("replay memory content hash mismatch")
        return memory

    def update_from_stage(
        self,
        allowed_gt_view,
        media_index: Mapping[str, Any],
        stage_id: str,
        budget_bytes: int,
        clip_len: int = 4,
        seed: int = 20260907,
    ) -> dict:
        view = read_json(str(allowed_gt_view)) if isinstance(allowed_gt_view, (str, Path)) else allowed_gt_view
        candidate_clips = []
        clip_len = int(clip_len)
        for video in view.get("videos", []):
            frames = sorted(video.get("frames", []), key=lambda f: (int(f["frame_index"]), f["frame_key"]))
            if len(frames) < clip_len:
                continue
            for start in range(0, len(frames) - clip_len + 1):
                window = frames[start:start + clip_len]
                if any(int(window[i + 1]["frame_index"]) != int(window[i]["frame_index"]) + 1 for i in range(len(window) - 1)):
                    continue
                frame_payload = []
                focus_ids = set()
                media_paths = {}
                usable = True
                annotation_bytes = 0
                for frame in window:
                    annotations = [
                        dict(ann) for ann in frame.get("annotations", [])
                        if ann.get("label_source") in ("gt", "gt_replay")
                    ]
                    media = _actual_media(media_index.get(frame["frame_key"], {}), frame)
                    if not media:
                        usable = False
                        break
                    for ann in annotations:
                        focus_ids.add(int(ann["global_semantic_id"]))
                    annotation_bytes += len(json.dumps(annotations, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                    media_paths[media["path"]] = int(media["bytes"])
                    frame_payload.append({
                        "frame_key": frame["frame_key"],
                        "frame_index": int(frame["frame_index"]),
                        "file_name": frame["file_name"],
                        "timestamp_s": frame.get("timestamp_s"),
                        "width": int(frame["width"]),
                        "height": int(frame["height"]),
                        "annotations": annotations,
                        "exhaustive_global_ids": list(frame.get("exhaustive_global_ids", frame.get("supervised_global_ids", []))),
                        "ignore_regions": list(frame.get("ignore_regions", [])),
                        "media_path": media["path"],
                        "media_bytes": int(media["bytes"]),
                    })
                if not usable or not focus_ids:
                    continue
                clip_id = "%s:%s:%s" % (stage_id, video["video_id"], window[0]["frame_key"])
                candidate_clips.append({
                    "clip_id": clip_id,
                    "source_stage": str(stage_id),
                    "source_video_id": str(video.get("source_video_id", video["video_id"])),
                    "video_id": str(video["video_id"]),
                    "image_root": video.get("image_root"),
                    "split": video.get("split", "train"),
                    "focus_global_id": min(focus_ids),
                    "global_semantic_ids": sorted(focus_ids),
                    "frames": frame_payload,
                    "media_paths": media_paths,
                    "annotation_bytes": annotation_bytes,
                    "logical_bytes": sum(media_paths.values()) + annotation_bytes,
                })
        # Preserve older legal clips and add current legal GT clips.  Existing
        # clip IDs are immutable; a new stage may replace the same logical
        # window only when its source-stage identity differs.
        candidates = {str(clip["clip_id"]): dict(clip) for clip in self.clips}
        for clip in candidate_clips:
            candidates[str(clip["clip_id"])] = clip
        selected = stratified_reservoir(candidates.values(), int(budget_bytes), seed=seed)
        self.clips = selected
        self.version = canonical_json_hash(self._payload())
        return {
            "status": "OK",
            "stage_id": str(stage_id),
            "clips": len(self.clips),
            "candidate_clips": len(candidate_clips),
            "logical_bytes": self.audit_reachable_bytes()["logical_bytes"],
            "budget_bytes": int(budget_bytes),
            "version": self.version,
        }

    def sample(self, batch_spec: Optional[Mapping] = None, generator=None) -> list:
        values = list(self.clips)
        if not values:
            return []
        spec = dict(batch_spec or {})
        count = min(len(values), max(0, int(spec.get("count", len(values)))))
        rng = generator
        if rng is None:
            import random
            rng = random.Random(0)
        return rng.sample(values, count)

    def save(self, path: str) -> dict:
        payload = self._payload()
        payload["version"] = self.version
        write_json(path, payload)
        return {"path": Path(path).name, "version": self.version, "clips": len(self.clips)}

    @classmethod
    def load(cls, path: str, expected_protocol_hash: str, expected_registry_hash: Optional[str] = None):
        payload = read_json(path)
        if payload.get("protocol_hash") != expected_protocol_hash:
            raise ValueError("replay protocol hash mismatch")
        if expected_registry_hash is not None and payload.get("registry_hash") != expected_registry_hash:
            raise ValueError("replay registry hash mismatch")
        return cls.from_payload(payload)

    def as_view(self, stage_id: str, active_global_ids: Iterable[int], split: str = "train") -> dict:
        active = {int(value) for value in active_global_ids}
        # Keep every saved clip as an independent replay video.  Merging clips
        # by source video would make the dataset build new windows across the
        # gaps between independently selected historical snippets, fabricating
        # dt and violating true consecutive-clip replay.
        output_videos = []
        for clip in sorted(self.clips, key=lambda value: str(value["clip_id"])):
            clip_id = str(clip["clip_id"])
            replay_video_id = "replay_%s" % hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:24]
            frames = []
            for frame in clip.get("frames", []):
                current = dict(frame)
                current["annotations"] = []
                for ann in frame.get("annotations", []):
                    if int(ann["global_semantic_id"]) not in active:
                        continue
                    value = dict(ann)
                    value["label_source"] = "gt_replay"
                    value["label_status"] = "reliable"
                    current["annotations"].append(value)
                current["exhaustive_global_ids"] = sorted(
                    set(int(v) for v in frame.get("exhaustive_global_ids", []) if int(v) in active)
                )
                current["supervised_global_ids"] = list(current["exhaustive_global_ids"])
                current["label_scope"] = "replay"
                current["annotation_valid"] = True
                current["ignore_regions"] = list(frame.get("ignore_regions", []))
                frames.append(current)
            if not frames:
                continue
            frames = sorted(frames, key=lambda f: (int(f.get("frame_index", 0)), f["frame_key"]))
            output_videos.append({
                "video_id": replay_video_id,
                "replay_clip_id": clip_id,
                "source_video_id": clip.get("source_video_id", clip.get("video_id")),
                "split": clip.get("split", split),
                "dataset": "BDD100K MOT",
                "image_root": clip.get("image_root"),
                "frames": frames,
                "width": int(frames[0]["width"]),
                "height": int(frames[0]["height"]),
            })
        payload = {
            "schema_version": "cmot.v2",
            "manifest_kind": "replay_view",
            "stage_id": str(stage_id),
            "stream": "replay",
            "split": split,
            "active_global_ids": sorted(active),
            "memory_version": self.version,
            "videos": output_videos,
        }
        payload["manifest_hash"] = canonical_json_hash(payload)
        return payload

    def audit_reachable_bytes(self) -> dict:
        unique_paths = {}
        annotation_bytes = 0
        for clip in self.clips:
            annotation_bytes += int(clip.get("annotation_bytes", 0))
            for path, size in clip.get("media_paths", {}).items():
                unique_paths[str(path)] = max(unique_paths.get(str(path), 0), int(size))
            for frame in clip.get("frames", []):
                path = frame.get("media_path")
                if path and Path(path).is_file():
                    unique_paths[str(Path(path).resolve())] = int(Path(path).stat().st_size)
        media_bytes = sum(unique_paths.values())
        return {
            "media_bytes": media_bytes,
            "annotation_bytes": annotation_bytes,
            "logical_bytes": media_bytes + annotation_bytes,
            "unique_media_paths": len(unique_paths),
        }
