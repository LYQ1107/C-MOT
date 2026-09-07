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
        key = (str(value.get("focus_global_id", value.get("global_semantic_id", "unknown"))), str(value.get("source_video_id", "unknown")))
        groups[key].append(value)
    for key, values in groups.items():
        values.sort(key=lambda value: hashlib.sha256(("%s|%s|%s" % (seed, key, value.get("clip_id"))).encode("utf-8")).hexdigest())
    selected = []
    used = 0
    heads = [values[0] for _, values in sorted(groups.items()) if values]
    for clip in heads:
        size = int(clip.get("logical_bytes", 0))
        if used + size <= budget:
            selected.append(clip)
            used += size
    rest = [clip for values in groups.values() for clip in values[1:]]
    rest.sort(key=lambda value: hashlib.sha256(("%s|fill|%s" % (seed, value.get("clip_id"))).encode("utf-8")).hexdigest())
    for clip in rest:
        size = int(clip.get("logical_bytes", 0))
        if used + size <= budget:
            selected.append(clip)
            used += size
    return selected
