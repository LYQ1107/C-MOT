"""Deterministic bounded selection for replay clips."""

from collections import defaultdict
from typing import Iterable, List, Mapping


def stratified_reservoir(clips: Iterable[Mapping], budget_bytes: int) -> List[dict]:
    """Select clips by class/source strata without exceeding logical bytes.

    The first pass gives each sorted stratum a chance, then a deterministic
    fill pass uses the remaining budget.  This is intentionally simple and
    auditable; it is not presented as an optimal coreset.
    """

    budget = max(0, int(budget_bytes))
    groups = defaultdict(list)
    for clip in clips:
        value = dict(clip)
        key = (str(value.get("global_semantic_id", "unknown")), str(value.get("source_video_id", "unknown")))
        groups[key].append(value)
    for values in groups.values():
        values.sort(key=lambda value: str(value.get("clip_id", "")))
    selected = []
    used = 0
    heads = [values[0] for _, values in sorted(groups.items()) if values]
    for clip in heads:
        size = int(clip.get("logical_bytes", 0))
        if used + size <= budget:
            selected.append(clip)
            used += size
    rest = [clip for values in groups.values() for clip in values[1:]]
    rest.sort(key=lambda value: str(value.get("clip_id", "")))
    for clip in rest:
        size = int(clip.get("logical_bytes", 0))
        if used + size <= budget:
            selected.append(clip)
            used += size
    return selected
