"""Concrete converter interface for datasets actually found locally."""

from pathlib import Path
from typing import Any, Mapping

from .bdd_converter import convert_bdd
from .tao_converter import convert_tao_bdd


class DatasetConverter:
    """Common inspect/convert/validate boundary used by the CLI."""

    dataset_name = "unknown"

    def inspect(self, source_root, annotation_root):
        raise NotImplementedError

    def convert(self, source_root, annotation_root, output_root, options):
        raise NotImplementedError

    def validate(self, bundle):
        if not isinstance(bundle, Mapping) or bundle.get("schema_version") != "cmot.v1":
            raise ValueError("bundle is not a cmot.v1 manifest")
        if not bundle.get("videos"):
            raise ValueError("bundle contains no videos")
        return {"status": "OK", "videos": len(bundle["videos"]), "manifest_hash": bundle.get("manifest_hash")}


class BDD100KConverter(DatasetConverter):
    dataset_name = "bdd100k_mot"

    def inspect(self, source_root, annotation_root):
        annotation = Path(annotation_root)
        return {
            "dataset": self.dataset_name,
            "annotation_exists": annotation.is_file(),
            "image_root_exists": Path(source_root).is_dir(),
            "annotation_basename": annotation.name,
        }

    def convert(self, source_root, annotation_root, output_root, options):
        split = options.get("split", "train")
        domain = options.get("domain", "unknown")
        return convert_bdd([(str(annotation_root), str(source_root), split, domain)], str(output_root), bool(options.get("require_images", True)))


class TAOConverter(DatasetConverter):
    dataset_name = "tao_amodal_bdd_sparse"

    def inspect(self, source_root, annotation_root=None):
        root = Path(source_root)
        return {
            "dataset": self.dataset_name,
            "root_exists": root.is_dir(),
            "train_exists": (root / "annotations" / "train.json").is_file(),
            "validation_exists": (root / "annotations" / "validation.json").is_file(),
        }

    def convert(self, source_root, annotation_root, output_root, options):
        return convert_tao_bdd(str(source_root), str(output_root), bool(options.get("require_images", True)))

