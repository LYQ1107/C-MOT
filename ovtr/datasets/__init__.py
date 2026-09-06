# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data

from .torchvision_datasets import CocoDetection
from .lvis_seqs import build as build_lvis_generated_mot

from mmdet.datasets.builder import DATASETS, PIPELINES
from .builder import build_dataloader, build_dataset
from .coco_video_dataset import CocoVideoDataset
from .parsers import CocoVID
from .pipelines import (LoadMultiImagesFromFile, SeqCollect,
                        SeqDefaultFormatBundle, SeqLoadAnnotations,
                        SeqNormalize, SeqPad, SeqRandomFlip, SeqResize)
from .tao_dataset import TaoDataset
from .seq_multi_image_mix_dataset import SeqMultiImageMixDataset

__all__ = [
    "DATASETS",
    "PIPELINES",
    "build_dataloader",
    "build_dataset",
    "CocoVID",
    "BDDVideoDataset",
    "CocoVideoDataset",
    "LoadMultiImagesFromFile",
    "SeqLoadAnnotations",
    "SeqResize",
    "SeqNormalize",
    "SeqRandomFlip",
    "SeqPad",
    "SeqDefaultFormatBundle",
    "SeqCollect",
    "TaoDataset",
    "SeqMultiImageMixDataset"
]

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args, cfg):
    if args.dataset_file == 'lvis_generated_img_seqs':
        return build_lvis_generated_mot(image_set, args, cfg)
    raise ValueError(f'dataset {args.dataset_file} not supported')
