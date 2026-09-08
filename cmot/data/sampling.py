"""Small deterministic current/replay clip sampler."""

import hashlib
import random
from collections import defaultdict, deque
from typing import Sequence

import torch


class BalancedClipSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, total_steps: int, seed: int, stage_id: str,
                 stream_schedule: Sequence[str] = ("current", "current", "current", "replay"),
                 start_step: int = 0, negative_fraction: float = 0.2):
        self.dataset = dataset
        self.total_steps = int(total_steps)
        self.seed = int(seed)
        self.stage_id = str(stage_id)
        self.stream_schedule = tuple(str(v) for v in stream_schedule)
        self.start_step = int(start_step)
        self.negative_fraction = float(negative_fraction)
        self._cursors = defaultdict(int)
        self._orders = {}
        for stream in ("current", "replay"):
            for positive in (True, False):
                candidates = [
                    i for i, item in enumerate(dataset.clip_index)
                    if item.get("stream", "current") == stream and (bool(item.get("focus_global_ids")) if positive else not bool(item.get("focus_global_ids")))
                ]
                # Interleave class -> source-video -> clip strata.  The
                # resulting plan uses only legal GT focus fields, so adding or
                # removing PL cannot change the schedule.
                groups = defaultdict(list)
                for index in candidates:
                    item = dataset.clip_index[index]
                    focus = tuple(int(v) for v in item.get("focus_global_ids", []))
                    source = str(item.get("source_video_uid", item.get("source_video_id", "unknown")))
                    keys = [(int(value), source) for value in focus] or [("unknown", source)]
                    for key in keys:
                        groups[key].append(index)
                for key, values in groups.items():
                    values.sort(key=lambda i: hashlib.sha256(("%s|%s|%s|%s|%s" % (self.seed, self.stage_id, stream, key, i)).encode("utf-8")).hexdigest())
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
                self._orders[(stream, positive)] = interleaved

    def __len__(self):
        return max(0, self.total_steps - self.start_step)

    def _next(self, stream: str, positive: bool):
        key = (stream, positive)
        values = self._orders.get(key, [])
        if not values and positive:
            values = self._orders.get((stream, False), [])
            key = (stream, False)
        if not values:
            return None
        cursor = self._cursors[key]
        value = values[cursor % len(values)]
        self._cursors[key] = cursor + 1
        return value

    def __iter__(self):
        for absolute_step in range(self.start_step, self.total_steps):
            stream = self.stream_schedule[absolute_step % len(self.stream_schedule)]
            if stream == "replay" and not self._orders.get(("replay", True)) and not self._orders.get(("replay", False)):
                stream = "current"
            positive = (absolute_step % 5) != 4
            value = self._next(stream, positive)
            if value is None:
                value = self._next("current", False)
                if value is None:
                    value = self._next("current", True)
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
            "cursors": {"%s|%s" % key: int(value) for key, value in self._cursors.items()},
        }

    def load_state_dict(self, state):
        if int(state.get("seed", self.seed)) != self.seed or state.get("stage_id", self.stage_id) != self.stage_id:
            raise ValueError("sampler state does not match repair_v2 plan")
        for key, value in state.get("cursors", {}).items():
            stream, positive = key.rsplit("|", 1)
            self._cursors[(stream, positive == "True")] = int(value)

    def plan_hash(self) -> str:
        cursors = dict(self._cursors)
        start_step = self.start_step
        self.start_step = 0
        values = list(iter(self))
        self.start_step = start_step
        self._cursors.clear()
        self._cursors.update(cursors)
        return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()
