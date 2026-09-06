"""Explicit, hash-bound stage runner state machine.

The runner deliberately does not infer completion from a filename.  A stage
is complete only after its input hashes, real optimizer step count, checkpoint
and evaluation artifact have all been checked by the caller.
"""

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from ..manifest import canonical_json_hash, sha256_file, write_json


STATES = (
    "PENDING",
    "PREFLIGHT_OK",
    "RUNNING",
    "TRAINED",
    "EVALUATED",
    "COMPLETE",
    "FAILED",
    "BLOCKED_RESOURCE",
    "BLOCKED_NETWORK",
    "AUTH_REQUIRED",
    "NOT_RUN",
)


def _file_or_value(value: Any) -> Any:
    if isinstance(value, (str, os.PathLike)) and Path(value).is_file():
        return {"basename": Path(value).name, "sha256": sha256_file(str(value))}
    return value


def stable_experiment_id(
    stage_id: str,
    method: str,
    config: Any,
    manifests: Iterable[Any],
    parent_checkpoint: Any,
    pl_manifest: Any = None,
    memory_version: Any = None,
    seed: int = 20260907,
    budget: Any = None,
) -> str:
    """Return a deterministic ID from all state that can affect a stage."""

    payload = {
        "stage_id": stage_id,
        "method": method,
        "config": _file_or_value(config),
        "manifests": [_file_or_value(value) for value in manifests],
        "parent_checkpoint": _file_or_value(parent_checkpoint),
        "pl_manifest": _file_or_value(pl_manifest),
        "memory_version": _file_or_value(memory_version),
        "seed": int(seed),
        "budget": budget,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:20]


@dataclass
class ExperimentTask:
    stage_id: str
    method: str
    experiment_id: str
    output_dir: str
    expected_steps: int
    parent_checkpoint: Optional[str] = None
    train_view: Optional[str] = None
    eval_view: Optional[str] = None
    pl_manifest: Optional[str] = None
    memory_version: Optional[str] = None
    seed: int = 20260907
    budget: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "method": self.method,
            "experiment_id": self.experiment_id,
            "output_dir": self.output_dir,
            "expected_steps": int(self.expected_steps),
            "parent_checkpoint": None if self.parent_checkpoint is None else Path(self.parent_checkpoint).name,
            "train_view": None if self.train_view is None else Path(self.train_view).name,
            "eval_view": None if self.eval_view is None else Path(self.eval_view).name,
            "pl_manifest": None if self.pl_manifest is None else Path(self.pl_manifest).name,
            "memory_version": self.memory_version,
            "seed": int(self.seed),
            "budget": self.budget,
        }


class CurriculumRunner:
    """Run one already-resolved stage with explicit status transitions."""

    def __init__(self, task: ExperimentTask):
        self.task = task
        self.output_dir = Path(task.output_dir)
        self.status_path = self.output_dir / "status.json"

    def _write_status(self, state: str, **extra: Any) -> dict:
        if state not in STATES:
            raise ValueError("unknown runner state %s" % state)
        value = {
            "schema_version": "cmot.runner-status.v1",
            "state": state,
            "experiment_id": self.task.experiment_id,
            "task": self.task.as_dict(),
            "updated_unix": round(time.time(), 3),
        }
        value.update(extra)
        write_json(str(self.status_path), value)
        return value

    def preflight(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for label, value in (
            ("parent_checkpoint", self.task.parent_checkpoint),
            ("train_view", self.task.train_view),
            ("eval_view", self.task.eval_view),
            ("pl_manifest", self.task.pl_manifest),
        ):
            if value and not Path(value).is_file():
                missing.append(label)
        if int(self.task.expected_steps) < 1:
            missing.append("expected_steps")
        if missing:
            self._write_status("PENDING", failure="PREFLIGHT_MISSING", missing=missing)
            raise FileNotFoundError("preflight failed: %s" % ",".join(missing))
        return self._write_status(
            "PREFLIGHT_OK",
            input_hashes={
                key: None if value is None else _file_or_value(value)
                for key, value in (
                    ("parent_checkpoint", self.task.parent_checkpoint),
                    ("train_view", self.task.train_view),
                    ("eval_view", self.task.eval_view),
                    ("pl_manifest", self.task.pl_manifest),
                )
            },
        )

    def run(
        self,
        train_fn: Callable[[ExperimentTask], Dict[str, Any]],
        eval_fn: Callable[[ExperimentTask, Dict[str, Any]], Dict[str, Any]],
        memory_fn: Optional[Callable[[ExperimentTask, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> dict:
        """Execute callbacks and verify their declared artifacts before advancing."""

        self.preflight()
        self._write_status("RUNNING", pid=os.getpid())
        try:
            train_result = train_fn(self.task)
            steps = int(train_result.get("steps_completed", -1))
            checkpoint = train_result.get("checkpoint")
            checkpoint_path = None if not checkpoint else Path(str(checkpoint))
            if checkpoint_path is not None and not checkpoint_path.is_absolute():
                checkpoint_path = self.output_dir / checkpoint_path
            if steps < int(self.task.expected_steps) or checkpoint_path is None or not checkpoint_path.is_file():
                raise RuntimeError("training callback did not prove expected optimizer steps/checkpoint")
            self._write_status("TRAINED", train=train_result, checkpoint_sha256=sha256_file(str(checkpoint_path)))
            eval_result = eval_fn(self.task, train_result)
            if eval_result.get("status") not in ("OK", "COMPLETE"):
                raise RuntimeError("evaluation callback did not return OK")
            self._write_status("EVALUATED", train=train_result, evaluation=eval_result)
            memory_result = None if memory_fn is None else memory_fn(self.task, train_result)
            return self._write_status(
                "COMPLETE",
                train=train_result,
                evaluation=eval_result,
                memory=memory_result,
            )
        except Exception as exc:
            self._write_status(
                "FAILED",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise


__all__ = ["STATES", "ExperimentTask", "CurriculumRunner", "stable_experiment_id"]
