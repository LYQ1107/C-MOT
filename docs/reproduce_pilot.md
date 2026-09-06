# Reproduce the pilot

The commands below are templates; they require local paths supplied by the operator and do not assume public repository access to data or weights.

```shell
export PYTHONPATH="$PWD:$PWD/ovtr"
python -m cmot.inventory --output /path/to/private/inventory.json
python -m cmot.download head https://example.invalid/asset.zip
python -m cmot.data.bdd_converter --source ANNOTATION,IMAGE_ROOT,train,domain --output /path/to/private/canonical.json
python -m cmot.data.view_builder --canonical /path/to/private/canonical.json --stage S0_ref --output /path/to/private/S0_train.json --split train --mode train
```

Use `cmot.infer` for an `asset_check` and `cmot.train` for the 20/100/300-step checkpoints.  Then call `cmot.evaluate` with the raw JSONL and the local TrackEval checkout.  The public reports contain the exact sanitized command metadata and hashes; raw paths are kept in the private audit only.

