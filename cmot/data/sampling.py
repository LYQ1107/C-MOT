"""Deterministic current/replay clip sampler with optional QPL scheduling."""

import hashlib
from collections import defaultdict, deque
from typing import Sequence

import torch


def _interleaved_order(dataset, candidates, seed: int, stage_id: str, stream: str):
    """Interleave class/source strata without introducing random state."""
    groups = defaultdict(list)
    for index in candidates:
        item = dataset.clip_index[index]
        focus = tuple(int(value) for value in item.get("focus_global_ids", []))
        source = str(item.get("source_video_uid", item.get("source_video_id", "unknown")))
        keys = [(int(value), source) for value in focus] or [("unknown", source)]
        for key in keys:
            groups[key].append(index)
    for key, values in groups.items():
        values.sort(
            key=lambda index: hashlib.sha256(
                ("%s|%s|%s|%s|%s" % (seed, stage_id, stream, key, index)).encode("utf-8")
            ).hexdigest()
        )
    strata = [deque(values) for _, values in sorted(groups.items(), key=lambda value: repr(value[0])) if values]
    interleaved = []
    seen = set()
    while strata:
        next_strata = []
        for values in strata:
            if values:
                candidate = values.popleft()
                if candidate not in seen:
                    interleaved.append(candidate)
                    seen.add(candidate)
            if values:
                next_strata.append(values)
        strata = next_strata
    return interleaved


def _pl_order(dataset, candidates, seed: int, stage_id: str):
    """Round-robin PL-containing clips by source video and segment."""
    groups = defaultdict(list)
    for index in candidates:
        item = dataset.clip_index[index]
        source = str(item.get("source_video_uid", item.get("source_video_id", "unknown")))
        segments = tuple(str(value) for value in item.get("pl_segment_ids", []))
        for segment_id in segments:
            groups[(source, segment_id)].append(index)
    for key, values in groups.items():
        values.sort(
            key=lambda index: hashlib.sha256(
                ("%s|%s|pl|%s|%s" % (seed, stage_id, key, index)).encode("utf-8")
            ).hexdigest()
        )
    strata = [deque(groups[key]) for key in sorted(groups, key=repr)]
    result = []
    seen = set()
    while strata:
        next_strata = []
        for values in strata:
            if values:
                index = values.popleft()
                if index not in seen:
                    result.append(index)
                    seen.add(index)
            if values:
                next_strata.append(values)
        strata = next_strata
    return result


class BalancedClipSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        total_steps: int,
        seed: int,
        stage_id: str,
        stream_schedule: Sequence[str] = ("current", "current", "current", "replay"),
        start_step: int = 0,
        negative_fraction: float = 0.2,
        pl_clip_fraction: float = 0.0,
        enable_pl: bool = False,
    ):
        self.dataset = dataset
        self.total_steps = int(total_steps)
        self.seed = int(seed)
        self.stage_id = str(stage_id)
        self.stream_schedule = tuple(str(v) for v in stream_schedule)
        self.start_step = int(start_step)
        self.negative_fraction = float(negative_fraction)
        self.pl_clip_fraction = float(pl_clip_fraction)
        self.enable_pl = bool(enable_pl)
        self.pl_schedule_fallback_count = 0
        self._cursors = defaultdict(int)
        self._orders = {}
        self._all_orders = {}
        for stream in ("current", "replay"):
            for positive in (True, False):
                all_candidates = [
                    index
                    for index, item in enumerate(dataset.clip_index)
                    if item.get("stream", "current") == stream
                    and (bool(item.get("focus_global_ids")) if positive else not bool(item.get("focus_global_ids")))
                ]
                candidates = list(all_candidates)
                if self.enable_pl and stream == "current":
                    non_pl = [index for index in candidates if not bool(dataset.clip_index[index].get("has_pl", False))]
                    if non_pl:
                        candidates = non_pl
                self._all_orders[(stream, positive)] = _interleaved_order(
                    dataset, all_candidates, self.seed, self.stage_id, stream
                )
                self._orders[(stream, positive)] = _interleaved_order(
                    dataset, candidates, self.seed, self.stage_id, stream
                )
        self._pl_order = _pl_order(
            dataset,
            [
                index
                for index, item in enumerate(dataset.clip_index)
                if item.get("stream", "current") == "current" and bool(item.get("has_pl", False))
            ],
            self.seed,
            self.stage_id,
        )
        self._pl_cursor = 0

    def __len__(self):
        return max(0, self.total_steps - self.start_step)

    def _next_from(self, orders, stream: str, positive: bool):
        key = (stream, positive)
        values = orders.get(key, [])
        if not values and positive:
            values = orders.get((stream, False), [])
            key = (stream, False)
        if not values:
            return None
        cursor = self._cursors[key]
        value = values[cursor % len(values)]
        self._cursors[key] = cursor + 1
        return value

    def _next_pl(self):
        if not self._pl_order:
            return None
        value = self._pl_order[self._pl_cursor % len(self._pl_order)]
        self._pl_cursor += 1
        return value

    def _current_ordinal(self, absolute_step: int) -> int:
        return sum(
            1
            for index in range(int(absolute_step))
            if self.stream_schedule[index % len(self.stream_schedule)] == "current"
        )

    def _is_negative_step(self, absolute_step: int) -> bool:
        fraction = float(self.negative_fraction)
        if fraction <= 0.0:
            return False
        if fraction >= 1.0:
            return True
        before = int(float(absolute_step) * fraction)
        after = int(float(absolute_step + 1) * fraction)
        return after > before

    def __iter__(self):
        for absolute_step in range(self.start_step, self.total_steps):
            stream = self.stream_schedule[absolute_step % len(self.stream_schedule)]
            if stream == "replay" and not self._orders.get(("replay", True)) and not self._orders.get(("replay", False)):
                stream = "current"
            positive = not self._is_negative_step(absolute_step)
            value = None
            if (
                stream == "current"
                and self.enable_pl
                and self.pl_clip_fraction > 0.0
                and self._pl_order
            ):
                period = max(1, int(round(1.0 / self.pl_clip_fraction)))
                if self._current_ordinal(absolute_step) % period == 0:
                    value = self._next_pl()
            elif stream == "current" and self.enable_pl and self.pl_clip_fraction > 0.0:
                # The requested fraction was scheduled, but there is no
                # eligible PL-containing clip.  The fallback is observable.
                period = max(1, int(round(1.0 / self.pl_clip_fraction)))
                if self._current_ordinal(absolute_step) % period == 0:
                    self.pl_schedule_fallback_count += 1
            if value is None:
                value = self._next_from(self._orders, stream, positive)
            if value is None and stream == "current" and self.enable_pl:
                value = self._next_from(self._all_orders, stream, positive)
            if value is None:
                value = self._next_from(self._orders, "current", False)
                if value is None:
                    value = self._next_from(self._orders, "current", True)
            if value is None:
                raise RuntimeError("balanced sampler has no usable clip")
            yield int(value)

    def state_dict(self):
        return {
            "total_steps": self.total_steps,
            "start_step": self.start_step,
            "seed": self.seed,
            "stage_id": self.stage_id,
            "stream_schedule": list(self.stream_schedule),
            "pl_clip_fraction": self.pl_clip_fraction,
            "enable_pl": self.enable_pl,
            "cursors": {"%s|%s" % key: int(value) for key, value in self._cursors.items()},
            "pl_cursor": int(self._pl_cursor),
            "pl_schedule_fallback_count": int(self.pl_schedule_fallback_count),
        }

    def load_state_dict(self, state):
        if int(state.get("seed", self.seed)) != self.seed or state.get("stage_id", self.stage_id) != self.stage_id:
            raise ValueError("sampler state does not match continual_v3 plan")
        if abs(float(state.get("pl_clip_fraction", self.pl_clip_fraction)) - self.pl_clip_fraction) > 1e-9:
            raise ValueError("sampler PL fraction does not match continual_v3 plan")
        if bool(state.get("enable_pl", self.enable_pl)) != self.enable_pl:
            raise ValueError("sampler PL enable flag does not match continual_v3 plan")
        for key, value in state.get("cursors", {}).items():
            stream, positive = key.rsplit("|", 1)
            self._cursors[(stream, positive == "True")] = int(value)
        self._pl_cursor = int(state.get("pl_cursor", 0))
        self.pl_schedule_fallback_count = int(state.get("pl_schedule_fallback_count", 0))

    def plan_hash(self) -> str:
        cursors = dict(self._cursors)
        pl_cursor = self._pl_cursor
        fallback_count = self.pl_schedule_fallback_count
        start_step = self.start_step
        self.start_step = 0
        values = list(iter(self))
        self.start_step = start_step
        self._cursors.clear()
        self._cursors.update(cursors)
        self._pl_cursor = pl_cursor
        self.pl_schedule_fallback_count = fallback_count
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()
