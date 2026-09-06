# Data contract

## Namespaces

The following identifiers must never be used interchangeably:

| Name | Meaning |
| --- | --- |
| `dataset_category_id` | source dataset category, e.g. BDD `1/2/3` |
| `global_semantic_id` | zero-based row in the 1203-row LVIS semantic bank: car `206`, pedestrian `792`, truck `1122` |
| `text_row` | semantic-bank row; equal to the global semantic ID for this registry |
| `select_column_id` | position inside the current stage's ordered `active_global_ids` list |
| `track_id` | source object identity; PL IDs are explicitly offset into a separate namespace |

The BDD mapping used by the primary local run is:

| BDD category | class | global/text row |
| ---: | --- | ---: |
| 1 | pedestrian | 792 |
| 2 | car | 206 |
| 3 | truck | 1122 |

## Frame and label rules

- Canonical manifests list every selected frame, including frames with zero labels.
- Bounding boxes are stored as absolute `xyxy` in the canonical manifest and become normalized `cxcywh` only at the model boundary.
- A training stage view includes only current/new labels, bounded old-class GT replay, and old-class PL records.  It never includes future-class GT.
- An evaluation view includes complete GT for the stage's seen classes and no PL labels.
- `label_scope=partial` makes unobserved old classes unknown; absence is not a background target.
- `label_source` is one of `gt`, `gt_replay`, or `pl`; the source is retained through conversion and training metadata.

