import json

from cmot.memory.clip_memory import ClipReplayMemory
from cmot.protocols.runner import CurriculumRunner, ExperimentTask, stable_experiment_id
from tools.cmot.audit_support import audit_support


def _manifest():
    return {
        "schema_version": "cmot.v1",
        "manifest_hash": "toy",
        "videos": [{
            "video_id": "v0",
            "source_video_id": 7,
            "split": "train",
            "frames": [
                {"frame_key": "v0/0", "frame_index": 0, "timestamp_s": 0.0, "annotations": [
                    {"track_id": 1, "global_semantic_id": 206, "label_source": "gt"},
                    {"track_id": 2, "global_semantic_id": 206, "label_source": "gt"},
                ]},
                {"frame_key": "v0/1", "frame_index": 1, "timestamp_s": 1.0, "annotations": [
                    {"track_id": 1, "global_semantic_id": 206, "label_source": "gt"},
                ]},
            ],
        }],
    }


def test_support_audit_counts_source_videos_and_pairs(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    result = audit_support(str(path))
    car = result["classes"]["car"]["by_split"]["train"]
    assert car["independent_source_video_count"] == 1
    assert car["track_count"] == 2
    assert car["supervised_same_identity_pairs"] == 1
    assert car["same_class_competing_windows"] == 1


def test_replay_charges_reachable_media_and_reloads(tmp_path):
    memory = ClipReplayMemory("protocol", "registry")
    result = memory.update_from_stage(
        _manifest(),
        {"v0/0": {"path": "/private/frame.jpg", "bytes": 100}, "v0/1": {"path": "/private/frame.jpg", "bytes": 100}},
        "S0_ref",
        10000,
    )
    assert result["clips"] == 3
    assert memory.audit_reachable_bytes()["media_bytes"] == 100
    saved = tmp_path / "memory.json"
    memory.save(str(saved))
    restored = ClipReplayMemory.load(str(saved), "protocol", "registry")
    assert restored.version == memory.version


def test_runner_requires_real_train_and_eval_artifacts(tmp_path):
    parent = tmp_path / "parent.pt"
    train_view = tmp_path / "train.json"
    eval_view = tmp_path / "eval.json"
    for path in (parent, train_view, eval_view):
        path.write_text("x", encoding="utf-8")
    task = ExperimentTask(
        "S1_pedestrian", "B1", stable_experiment_id("S1", "B1", "cfg", [], str(parent)),
        str(tmp_path / "run"), 2, str(parent), str(train_view), str(eval_view), seed=1,
    )
    runner = CurriculumRunner(task)
    checkpoint = tmp_path / "checkpoint_002.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    result = runner.run(
        lambda _: {"steps_completed": 2, "checkpoint": str(checkpoint)},
        lambda *_: {"status": "OK"},
    )
    assert result["state"] == "COMPLETE"
    assert json.loads((tmp_path / "run" / "status.json").read_text(encoding="utf-8"))["state"] == "COMPLETE"
