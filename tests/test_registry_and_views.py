import json

from cmot.class_registry import tao_bdd_registry
from cmot.data.view_builder import build_stage_view


def test_registry_keeps_source_global_and_select_namespaces_separate():
    registry = tao_bdd_registry()
    assert registry.global_id_for_dataset("bdd100k_mot", 1) == 792
    assert registry.global_id_for_dataset("bdd100k_mot", 2) == 206
    assert registry.global_id_for_dataset("bdd100k_mot", 3) == 1122
    registry.set_active(["car", "pedestrian"])
    assert registry.active_global_ids() == (206, 792)
    assert registry.active_select_ids() == (206, 792)
    assert registry.global_to_column(206) == 0
    assert registry.column_to_global(1) == 792


def _toy_manifest():
    annotations = [
        {
            "track_id": 7,
            "dataset_category_id": 2,
            "global_semantic_id": 206,
            "bbox_xyxy": [0, 0, 10, 10],
            "label_source": "gt",
            "label_status": "reliable",
            "score": None,
            "iscrowd": 0,
        },
        {
            "track_id": 8,
            "dataset_category_id": 1,
            "global_semantic_id": 792,
            "bbox_xyxy": [10, 10, 20, 20],
            "label_source": "gt",
            "label_status": "reliable",
            "score": None,
            "iscrowd": 0,
        },
        {
            "track_id": 9,
            "dataset_category_id": 3,
            "global_semantic_id": 1122,
            "bbox_xyxy": [20, 20, 30, 30],
            "label_source": "gt",
            "label_status": "reliable",
            "score": None,
            "iscrowd": 0,
        },
    ]
    frames = []
    for index in range(2):
        frames.append(
            {
                "frame_key": "toy/%06d" % index,
                "frame_index": index,
                "file_name": "toy-%06d.jpg" % index,
                "width": 100,
                "height": 100,
                "timestamp_s": float(index),
                "label_scope": "complete",
                "supervised_global_ids": [206, 792, 1122],
                "annotations": annotations,
            }
        )
    return {
        "schema_version": "cmot.v1",
        "manifest_kind": "canonical_real_video",
        "videos": [
            {
                "video_id": "toy-video",
                "split": "train",
                "width": 100,
                "height": 100,
                "image_root": "toy",
                "frames": frames,
            }
        ],
    }


def test_partial_stage_view_hides_old_and_future_labels(tmp_path):
    canonical = tmp_path / "canonical.json"
    output = tmp_path / "s1.json"
    canonical.write_text(json.dumps(_toy_manifest()), encoding="utf-8")
    stats = build_stage_view(str(canonical), "S1_pedestrian", str(output), replay_max_frames=0)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert stats["global_ids"] == {792: 2}
    assert payload["label_scope"] == "partial"
    assert payload["videos"][0]["frames"][0]["supervised_global_ids"] == [792]
    assert all(
        int(annotation["global_semantic_id"]) == 792
        for frame in payload["videos"][0]["frames"]
        for annotation in frame["annotations"]
    )


def test_naive_complete_view_is_explicit(tmp_path):
    canonical = tmp_path / "canonical.json"
    output = tmp_path / "b0.json"
    canonical.write_text(json.dumps(_toy_manifest()), encoding="utf-8")
    build_stage_view(
        str(canonical), "S1_pedestrian", str(output), replay_max_frames=0,
        label_scope="complete",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["label_protocol"] == "naive_complete"
    assert payload["videos"][0]["frames"][0]["supervised_global_ids"] == [206, 792]
