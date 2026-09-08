"""Deterministic bounded selection for replay clips."""

import hashlib
from collections import defaultdict
from typing import Iterable, List, Mapping


def stratified_reservoir(clips: Iterable[Mapping], budget_bytes: int, seed: int = 20260907) -> List[dict]:
    """Select whole clips by class/source strata under one byte budget."""
    budget = max(0, int(budget_bytes))
    groups = defaultdict(list)
    for clip in clips:
        value = dict(clip)
        focus_ids = value.get("global_semantic_ids") or [value.get("focus_global_id", value.get("global_semantic_id", "unknown"))]
        source_uid = str(value.get("source_video_uid", value.get("source_video_id", "unknown")))
        # A clip containing multiple classes participates in each class/source
        # stratum, but is de-duplicated when the final buffer is assembled.
        for focus_id in focus_ids:
            groups[(str(focus_id), source_uid)].append(value)
    for key, values in groups.items():
        values.sort(key=lambda value: hashlib.sha256(("%s|%s|%s" % (seed, key, value.get("clip_id"))).encode("utf-8")).hexdigest())
    selected = []
    selected_ids = set()
    used = 0
    heads = [values[0] for _, values in sorted(groups.items()) if values]
    for clip in heads:
        clip_id = str(clip.get("clip_id"))
        if clip_id in selected_ids:
            continue
        size = int(clip.get("logical_bytes", 0))
        if used + size <= budget:
            selected.append(clip)
            selected_ids.add(clip_id)
            used += size
    rest = []
    seen_rest = set()
    for values in groups.values():
        for clip in values[1:]:
            clip_id = str(clip.get("clip_id"))
            if clip_id not in selected_ids and clip_id not in seen_rest:
                rest.append(clip)
                seen_rest.add(clip_id)
    rest.sort(key=lambda value: hashlib.sha256(("%s|fill|%s" % (seed, value.get("clip_id"))).encode("utf-8")).hexdigest())
    for clip in rest:
        clip_id = str(clip.get("clip_id"))
        if clip_id in selected_ids:
            continue
        size = int(clip.get("logical_bytes", 0))
        if used + size <= budget:
            selected.append(clip)
            selected_ids.add(clip_id)
            used += size
    return selected
