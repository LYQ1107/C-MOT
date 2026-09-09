Implemented the independent cmot.cooler_compat.v1 path: strict stage views, replay-free pair sampling, tracker PL exclusion, accumulation trainer, TrackEval adapter, and fail-closed runner.

Formal training status: BLOCKED by the local full-BDD gate.

The explicitly requested `/BDD100K` candidate was inspected and recorded in
`full_bdd_discovery.json`.  It lacks `box_track_20`; the separate local label
bundle is missing 1,200 train track-image directories and 238,661 referenced
train frames.  No data download or source modification was performed.
