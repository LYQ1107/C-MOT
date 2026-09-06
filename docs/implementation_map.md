# Implementation map

The C-MOT execution path is deliberately separate from the upstream LVIS pseudo-video entry point.

| Concern | Implementation | Evidence |
| --- | --- | --- |
| Clean upstream import | `UPSTREAM_PROVENANCE.json` and archived source tree | source commit is recorded without importing its nested Git history |
| Identifier separation | `cmot/schema.py`, `cmot/class_registry.py:19-139` | registry maps BDD category IDs to LVIS semantic rows and stage columns explicitly |
| Real BDD conversion | `cmot/data/bdd_converter.py:23-155` | canonical manifest records every selected frame and checks image existence |
| TAO fallback conversion | `cmot/data/tao_converter.py:23-168` | keeps the TAO `person.n.01`/`baby` provenance explicit |
| Stage views | `cmot/data/view_builder.py:47-174` | future classes are excluded; training/evaluation views are distinct |
| Partial labels | `ovtr/models/ovtr.py:230-278,418-605` and stage-view metadata | class-wise valid masks avoid treating hidden old classes as background in the primary and auxiliary paths |
| Raw-state inference | `ovtr/models/ovtr.py:1030-1073`, `cmot/ovtr_runtime.py:125-186` | post-processing operates on a copy, preserving next-frame raw fields |
| Causal motion | `ovtr/models/ovtr.py:128-161,917-977`, `ovtr/models/updater.py:175-179,247-251` | current decoder feature + current box + semantic embedding affect next inverse-sigmoid reference |
| Motion supervision | `ovtr/models/ovtr.py:319-347,605-612` | next-frame GT is read only inside the loss target path and `loss_motion` is registered |
| Direct-download audit | `cmot/download.py:42-180` | cleans child environment, disables proxy/config, audits links/routes, fails closed |
| Resource/support audit | `cmot/inventory.py:30-76`, `tools/cmot/audit_support.py:30-120` | runtime-supplied asset inventory and source-video/track/exposure statistics; no frame count is treated as video count |
| Converter interface | `cmot/data/converters.py:10-61`, `tools/cmot/convert_dataset.py` | `DatasetConverter.inspect/convert/validate`; only BDD and the locally parsed sparse TAO-BDD format are supported |
| Bounded replay | `cmot/memory/clip_memory.py:12-134`, `cmot/memory/selection.py:7-40` | legal GT-only clip metadata, deterministic stratified selection and reachable media-byte accounting |
| Stage runner | `cmot/protocols/runner.py:41-211`, `tools/cmot/run_curriculum.py` | hash-bound preflight and explicit PENDING→PREFLIGHT_OK→RUNNING→TRAINED→EVALUATED→COMPLETE transitions |
| Training | `cmot/train.py:59-195` | optimizer step, checkpoint, reload metadata, loss/gradient/update audit |
| Evaluation | `cmot/evaluate.py:42-196` | local TrackEval BDD100K HOTA/CLEAR/Identity adapter with empty frames retained |

The standalone `cmot/models/motion_prior.py` additionally exposes the
protocol's K-mode history interface and `cmot/losses/motion.py` exposes the
one-normalization NLL contract.  The reported pilot checkpoints use the
legacy-compatible `ovtr/models/ovtr.py:CausalMotionHead` path, whose delta is
already connected to the next inverse-sigmoid reference; the standalone prior
and loss contracts are tested separately and are not silently substituted for
the reported model.

The upstream `ovtr/tools/ovtr_multi_frame_lite_train.sh` remains a reference only: it hard-codes protected GPUs and the LVIS generated pseudo-video dataset, so it is not the C-MOT runner.
