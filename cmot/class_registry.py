"""Class and identifier registry used by every C-MOT stage."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import SemanticClass, StageSpec


DEFAULT_CLASSES = {
    # IDs are the zero-based rows in ovtr/util/list_LVIS.py and the local
    # 1203-row semantic banks.  They are deliberately not TAO category IDs.
    "car": SemanticClass("car", 206, 206, ("car_(automobile)", "automobile")),
    "pedestrian": SemanticClass("pedestrian", 792, 792, ("person", "person.n.01", "baby")),
    "truck": SemanticClass("truck", 1122, 1122, ("truck",)),
}


class ClassRegistry:
    """Separate semantic IDs from model select columns and source IDs."""

    def __init__(
        self,
        classes: Mapping[str, SemanticClass] = DEFAULT_CLASSES,
        dataset_category_maps: Optional[Mapping[str, Mapping[int, str]]] = None,
    ):
        self.classes: Dict[str, SemanticClass] = dict(classes)
        self.dataset_category_maps: Dict[str, Dict[int, str]] = {
            source: {int(k): str(v) for k, v in mapping.items()}
            for source, mapping in (dataset_category_maps or {}).items()
        }
        self._active: Tuple[str, ...] = tuple()
        self._validate()

    def _validate(self) -> None:
        semantic_ids = [v.global_semantic_id for v in self.classes.values()]
        text_rows = [v.text_row for v in self.classes.values()]
        if len(set(semantic_ids)) != len(semantic_ids):
            raise ValueError("global semantic IDs collide")
        if len(set(text_rows)) != len(text_rows):
            raise ValueError("text rows collide")
        for source, mapping in self.dataset_category_maps.items():
            for category_id, name in mapping.items():
                if name not in self.classes:
                    raise ValueError("%s maps category %s to unknown class %s" % (source, category_id, name))

    def set_active(self, names_or_ids: Sequence[object]) -> None:
        names: List[str] = []
        for value in names_or_ids:
            if isinstance(value, str):
                name = value
            else:
                matching = [n for n, c in self.classes.items() if c.global_semantic_id == int(value)]
                if not matching:
                    raise KeyError("unknown global semantic ID %s" % value)
                name = matching[0]
            if name not in self.classes:
                raise KeyError("unknown class %s" % name)
            if name not in names:
                names.append(name)
        self._active = tuple(names)

    def active_names(self) -> Tuple[str, ...]:
        return self._active

    def active_global_ids(self) -> Tuple[int, ...]:
        return tuple(self.classes[n].global_semantic_id for n in self._active)

    def active_select_ids(self) -> Tuple[int, ...]:
        # The semantic bank is indexed by text/global rows, while this tuple
        # is also the exact model column order for the current stage.
        return self.active_global_ids()

    def global_to_column(self, global_id: int) -> int:
        ids = self.active_global_ids()
        if int(global_id) not in ids:
            raise KeyError("global ID %s is not active" % global_id)
        return ids.index(int(global_id))

    def column_to_global(self, column_id: int) -> int:
        ids = self.active_global_ids()
        return int(ids[int(column_id)])

    def global_id_for_dataset(self, source: str, dataset_category_id: int) -> int:
        name = self.dataset_category_maps[source][int(dataset_category_id)]
        return self.classes[name].global_semantic_id

    def class_name_for_global(self, global_id: int) -> str:
        for name, value in self.classes.items():
            if value.global_semantic_id == int(global_id):
                return name
        raise KeyError(global_id)

    def stage(self, stage_id: str) -> StageSpec:
        stages = {
            "S0_ref": ("S0_ref", "car foundation", ("car",), ("car",), (), "complete", "none"),
            "S1_pedestrian": ("S1_pedestrian", "pedestrian increment", ("car", "pedestrian"), ("pedestrian",), ("car",), "partial", "none"),
            "S2_truck": ("S2_truck", "truck increment", ("car", "pedestrian", "truck"), ("truck",), ("car", "pedestrian"), "partial", "none"),
        }
        if stage_id not in stages:
            raise KeyError(stage_id)
        sid, name, active, new, old, protocol, motion = stages[stage_id]
        return StageSpec(
            sid,
            name,
            tuple(self.classes[n].global_semantic_id for n in active),
            tuple(self.classes[n].global_semantic_id for n in new),
            tuple(self.classes[n].global_semantic_id for n in old),
            label_protocol=protocol,
            motion_mode=motion,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "classes": {name: cls.as_dict() for name, cls in self.classes.items()},
            "dataset_category_maps": self.dataset_category_maps,
            "active_names": list(self._active),
            "active_global_ids": list(self.active_global_ids()),
            "active_select_ids": list(self.active_select_ids()),
        }

    def write(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tao_bdd_registry() -> ClassRegistry:
    registry = ClassRegistry(
        dataset_category_maps={
            # TAO/LVIS category IDs are source IDs.  805 is the LVIS
            # person.n.01 category whose display name is "baby"; it is not
            # being claimed as official BDD100K MOT person labeling.
            "tao_bdd_sparse": {211: "car", 805: "pedestrian", 1144: "truck"},
            # BDD100K MOT box_track_20 uses 1=pedestrian, 2=car,
            # 3=truck.  These source IDs are intentionally not the
            # global/text/select identifiers above.
            "bdd100k_mot": {1: "pedestrian", 2: "car", 3: "truck"},
        }
    )
    return registry
