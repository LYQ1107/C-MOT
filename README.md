# C-MOT

C-MOT is a reproducible implementation layer for real-video category-incremental adaptation of OVTR.  It keeps the upstream OVTR implementation as a read-only-derived baseline and adds explicit data contracts, partial-label views, bounded replay, causal category-conditioned motion, raw prediction evidence, and auditable runners.

The public repository contains source and small, sanitized reports only.  Dataset frames, annotations, semantic assets, checkpoints, and full logs stay outside the repository.

## Scope

The primary local protocol uses BDD100K box-track data that was already available on the execution server.  Training-domain weather sequences and independent night sequences are used as a fixed video list.  The three semantic stages are:

`S0_ref: car` → `S1_pedestrian: car + pedestrian` → `S2_truck: car + pedestrian + truck`.

The source category IDs, global semantic IDs, text rows, model select columns, and track IDs are distinct namespaces.  See [docs/data_contract.md](docs/data_contract.md) and [docs/protocols.md](docs/protocols.md) for the exact rules.

## Runtime

The validated server environment is Python 3.8 with the upstream OVTR dependencies.  The C-MOT commands need both the project root and `ovtr/` on `PYTHONPATH`:

```shell
export PYTHONPATH="$PWD:$PWD/ovtr"
```

All paths below are command-line inputs.  They are intentionally not hard-coded into public configuration files.

```shell
python -m cmot.data.bdd_converter --help
python -m cmot.data.view_builder --help
python -m cmot.infer --help
python -m cmot.train --help
python -m cmot.evaluate --help
python tools/cmot/audit_support.py --help
python tools/cmot/run_curriculum.py --help
```

The official TrackEval BDD100K adapter is used when it is available locally.  A missing evaluator is reported as `NOT_RUN`; no replacement score is silently substituted.

## Safety and provenance

- The original OVTR checkout and source datasets are read-only inputs.
- Bulk downloads use the fail-closed policy in [docs/no_proxy_download_policy.md](docs/no_proxy_download_policy.md); a failed direct route never falls back to a proxy.
- No credentials, cookies, tokens, private server paths, source data, weights, or large logs belong in Git.
- The imported OVTR source is identified in [UPSTREAM_PROVENANCE.json](UPSTREAM_PROVENANCE.json).

## Reports

Public result files live under `reports/public/`.  Values are written only after the corresponding command has run.  Unrun fields are `null` or `NOT_RUN`; pilot results are not presented as full benchmarks.
