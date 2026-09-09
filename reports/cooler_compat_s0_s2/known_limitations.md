- The explicitly inspected candidate `BDD100K` root contains standard 100k
  per-image detection data (70,000 train and 10,000 val images/labels), but no
  `box_track_20` source.  The existing local `box_track_20` labels have 1,400
  train and 200 val annotation videos, while the corresponding read-only track
  image root has only 200 train and 200 val video directories; 1,200 train
  video directories and 238,661 referenced train frames are missing.
- The candidate's `images20-track-val-1.zip` contains only 200 val track video
  directories and no labels or annotations.  It cannot complete the train
  side of this benchmark, so the full-data gate remains closed.
- COOLer reference commit was not downloaded or found locally; no new download was attempted.
- Consequently no checkpoint, prediction, TrackEval metric, or method gap was generated; those remain NOT_RUN/null.  The current diagnosis is the explicit data-gate blocker above.
