"""Small, hashable replay memory containing only an allowed stage view."""

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ..manifest import canonical_json_hash, read_json, write_json
from .selection import stratified_reservoir


class ClipReplayMemory:
    """A deterministic metadata replay buffer with reachable-byte accounting.

    Media are referenced, not silently made free by hard links or symlinks.
    ``media_index`` may provide ``bytes`` and an optional physical ``path`` for
    every frame.  The logical budget always charges the supplied media bytes;
    annotation JSON bytes are charged as well.
    """

    def __init__(self, protocol_hash: str, registry_hash: str, clips: Optional[Iterable[Mapping]] = None):
        self.protocol_hash = str(protocol_hash)
        self.registry_hash = str(registry_hash)
        self.clips = [dict(value) for value in (clips or [])]
        self.version = canonical_json_hash(self._payload())

    def _payload(self) -> dict:
        return {
            "schema_version": "cmot.clip-memory.v1",
            "protocol_hash": self.protocol_hash,
            "registry_hash": self.registry_hash,
            "clips": self.clips,
        }

    def update_from_stage(self, allowed_gt_view, media_index: Mapping[str, Any], stage_id: str, budget_bytes: int) -> dict:
        view = read_json(str(allowed_gt_view)) if isinstance(allowed_gt_view, (str, Path)) else allowed_gt_view
        candidates = []
        for video in view.get("videos", []):
            video_id = str(video["video_id"])
            source_video_id = video.get("source_video_id", video_id)
            for frame in video.get("frames", []):
                allowed = [ann for ann in frame.get("annotations", []) if ann.get("label_source") in ("gt", "gt_replay")]
                if not allowed:
                    continue
                frame_key = str(frame["frame_key"])
                media = media_index.get(frame_key, {})
                if isinstance(media, (int, float)):
                    media = {"bytes": int(media)}
                media_bytes = int(media.get("bytes", 0))
                annotation_bytes = len(json.dumps(allowed, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                for ann in allowed:
                    candidates.append({
                        "clip_id": "%s:%s:%s" % (stage_id, frame_key, ann.get("track_id")),
                        "stage_id": str(stage_id),
                        "source_video_id": str(source_video_id),
                        "video_id": video_id,
                        "frame_key": frame_key,
                        "track_id": int(ann["track_id"]),
                        "global_semantic_id": int(ann["global_semantic_id"]),
                        "label_source": ann.get("label_source", "gt"),
                        "media_path": media.get("path"),
                        "media_bytes": media_bytes,
                        "annotation_bytes": annotation_bytes,
                        "logical_bytes": media_bytes + annotation_bytes,
                    })
        chosen = stratified_reservoir(candidates, int(budget_bytes))
        self.clips = chosen
        self.version = canonical_json_hash(self._payload())
        return {
            "status": "OK",
            "stage_id": stage_id,
            "clips": len(self.clips),
            "logical_bytes": self.audit_reachable_bytes()["logical_bytes"],
            "version": self.version,
        }

    def sample(self, batch_spec: Optional[Mapping] = None, generator: Optional[random.Random] = None) -> list:
        values = list(self.clips)
        if not values:
            return []
        spec = dict(batch_spec or {})
        count = min(len(values), max(0, int(spec.get("count", len(values)))))
        rng = generator or random.Random(0)
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
        memory = cls(payload["protocol_hash"], payload.get("registry_hash", ""), payload.get("clips", []))
        if payload.get("version") and payload["version"] != memory.version:
            raise ValueError("replay memory content hash mismatch")
        return memory

    def audit_reachable_bytes(self) -> dict:
        # A physical file can be shared by clips, but it is still charged once
        # because that is the real reachable media byte cost.
        unique_paths = {}
        annotation_bytes = 0
        anonymous_media_bytes = 0
        for clip in self.clips:
            annotation_bytes += int(clip.get("annotation_bytes", 0))
            path = clip.get("media_path")
            if path:
                unique_paths[str(path)] = max(unique_paths.get(str(path), 0), int(clip.get("media_bytes", 0)))
            else:
                anonymous_media_bytes += int(clip.get("media_bytes", 0))
        media_bytes = sum(unique_paths.values()) + anonymous_media_bytes
        return {
            "media_bytes": media_bytes,
            "annotation_bytes": annotation_bytes,
            "logical_bytes": media_bytes + annotation_bytes,
            "unique_media_paths": len(unique_paths),
        }

