# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) Yutliu
# ------------------------------------------------------------------------
# Modified from OVTrack (https://github.com/SysCV/ovtrack)
# ------------------------------------------------------------------------
"""
MOT dataset which returns image_id for evaluation.
"""
from collections import defaultdict
import collections
import copy
import math
from operator import itemgetter
from pathlib import Path
import random
import cv2
import os 
import numpy as np
import torch
import torch.utils.data
from .tao_dataset import TaoDataset
from detectron2.structures import Instances
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import DATASETS, PIPELINES
from util.list_LVIS import CLASSES

@DATASETS.register_module(force=True)
class LVIS_seqs_Dataset(TaoDataset):
    def __init__(self,
        args_mot = None,
        train_pipeline = None, 
        load_as_video=True,
        match_gts=True,
        skip_nomatch_pairs=True,
        key_img_sampler=dict(interval=1),
        ref_img_sampler=dict(scope=3, num_ref_imgs=1, method="uniform"),
        *args,
        **kwargs,):

        self.ids = None

        super(LVIS_seqs_Dataset, self).__init__(load_as_video=True,
            match_gts=True,
            skip_nomatch_pairs=True,
            key_img_sampler=key_img_sampler,
            ref_img_sampler=ref_img_sampler,
            *args,
            **kwargs,)
        
        self.CLASSES = CLASSES
        self.get_repeat_factors()

        self.num_frames_per_batch = max(args_mot.sampler_lengths)
        self.sample_mode = args_mot.sample_mode
        self.sample_interval = args_mot.sample_interval
        self.video_dict = {}
        self.item_num = None

        if args_mot.filter_ignore:
            print('Training with ignore.', flush=True)

        # video sampler.
        self.sampler_steps: list = args_mot.sampler_steps
        self.lengths: list = args_mot.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.lengths) > 0
            assert len(self.lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            # self.item_num = len(self.img_files) - (self.lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]
            self.current_epoch = 0

        # pipeline
        pipeline = train_pipeline
        self.pipeline_seq = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = build_from_cfg(transform, PIPELINES)
                self.pipeline_seq.append(transform)
            else:
                raise TypeError('pipeline must be a dict')
        self.num_samples = len(self)
        self._skip_type_keys = None

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    def pre_continuous_frames(self, start, end, interval=1):
        targets = []
        images = []
        for i in range(start, end, interval):
            img_i, targets_i = self._pre_single_frame(i)
            images.append(img_i)
            targets.append(targets_i)
        return images, targets

    def __getitem__(self, idx_sampled):
        idx = self.repeat_indices[idx_sampled]
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
    
            results = copy.deepcopy(data)

            for (transform, transform_type) in zip(self.pipeline_seq,
                                                    self.pipeline_types):
                if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                    continue

                if hasattr(transform, 'get_indexes'):
                    random_value = random.random()
                    if random_value < 0.5:  
                        indexes = transform.get_indexes(self)
                        if not isinstance(indexes, collections.abc.Sequence):
                            indexes = [indexes]

                        # Find the pre-mixed image group.
                        mix_results = []
                        instance_num = len(results[0]['gt_labels'])
                        for index in indexes:
                            r_id = self.repeat_indices[index]
                            results_i = self.prepare_train_img(r_id)
                            while results_i is None:
                                r_id = self._rand_another(r_id)
                                results_i = self.prepare_train_img(r_id)
                            mix_results.append(copy.deepcopy(results_i)) 
                            instance_num += len(results_i[0]['gt_labels'])
                        if instance_num < 100: # Use Mosaic for scenarios with fewer objects.
                            for i, _result in enumerate(results):
                                _mix_result = []
                                for item in mix_results:
                                    _mix_result.append(item[i])
                                _result['mix_results'] = _mix_result
                    # seq_randint = random.randint(1, self.num_frames_per_batch-1)
                    for _ in range(self.num_frames_per_batch - 1):  
                        results.append(copy.deepcopy(results[0]))

                if transform_type == 'SeqRandomAffine':
                    results = transform(results, area_ratio)
                elif transform_type == 'RandomOcclusion':
                    results = transform(results, min_wh)
                    # Obtain the area ratio of the original
                    area_ratio = self.get_area_ratio(results)
                elif transform_type == 'SeqRandomFlip':
                    results = transform(results, unique)
                elif transform_type == 'DynamicMosaic':
                    if (random_value < 0.5) and (instance_num < 100):
                        results = transform(results)
                    # self.visualize(results)
                    min_wh = self.get_min_wh(results)
                    unique = self.category_unique(results)
                else:    
                    results = transform(results)

                if 'mix_results' in results:
                    results.pop('mix_results')

                # if transform_type == 'SeqResize':
                    # self.visualize(results)

            if isinstance(results, dict): # Check if the object is empty.
                data_flag = True
                for i in range(self.num_frames_per_batch):
                    if len(results[f'gt_bboxes_{i}'].data) == 0:
                        data_flag = False
                if not data_flag:
                    idx = self._rand_another(idx)
                    data_flag = True
                    continue
            
            filename = self.data_infos[idx]['filename']
            gt_instances = self.results_to_instances(results)
            gt_instances.update({'filename': filename})
            # pdb.set_trace()
            return gt_instances
        
    def category_unique(self, results):
        unique = True
        for _result in results:
            labels = _result['gt_labels']
            _, counts = np.unique(labels, return_counts=True)
            if (counts > 5).any():  
                unique = False
            if not unique:
                break
        return unique
        
    def get_min_wh(self, results):
        # not pair
        gt_bboxes = results[0]['gt_bboxes']
        width = (gt_bboxes[:, 2] - gt_bboxes[:, 0])
        height = (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        return [width.min(), height.min()]

    def get_area_ratio(self, results):
        area_ratio = [ ]
        for _result in results:
            _area_ratio = (_result['gt_bboxes'][:, 2] - _result['gt_bboxes'][:, 0]) * (_result['gt_bboxes'][:, 3] - _result['gt_bboxes'][:, 1]) # / area 
            area_ratio.append(_area_ratio)
        return area_ratio

    def visualize(self, results):
        Path('./data_vis').mkdir(parents=True, exist_ok=True)
        name = None
        root = str(Path.cwd())
        for i, _result in enumerate(results):
            name = _result['img_info']['filename'].split('/')[-1].split('.')[-2] + f'_v{i}.jpg'
            img = _result['img']
            bboxs = _result['gt_bboxes']
            for x, label in zip(bboxs, _result['gt_labels']):
                c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                cv2.rectangle(img, c1, c2, (144, 238, 144), thickness=1)
                cv2.putText(img, CLASSES[label], (c1[0], c1[1] + 3), 0, 0.5, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(root, './data_vis', name), img)
        
    def results_to_instances(self, results):
        gt_instances = []
        images = []
        data = {}
        results_seqs = []
        for i in range(self.num_frames_per_batch):
            results_seqs.append({'img_metas':results[f'img_metas_{i}'], 
                                  'img':results[f'img_{i}'], 
                                  'gt_bboxes':results[f'gt_bboxes_{i}'], 
                                  'gt_labels':results[f'gt_labels_{i}'], 
                                  'gt_match_indices':results[f'gt_match_indices_{i}'], 
                                })

        for result_i in results_seqs:
            gt_instances_i = self._target_to_instances(result_i)
            gt_instances.append(gt_instances_i)
            images.append(result_i['img'].data)
        data.update({
            'imgs': images,
            'gt_instances': gt_instances,
        })
        return data

    def _target_to_instances(self, data):
        img_shape_without_pad = data['img_metas'].data['pad_shape']
        gt_instance = Instances(img_shape_without_pad[:2])
        gt_instance.boxes = self.box_xyxy_to_cxcywh(data['gt_bboxes'].data, img_shape_without_pad)
        gt_instance.labels = data['gt_labels'].data
        gt_instance.obj_ids = data['gt_match_indices'].data 
        return gt_instance
    
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        results = self.prepare_results(img_info)

        if self.match_gts:
            results["ann_info"]["match_indices"] = results['ann_info']["instance_ids"]
   
        self.pre_pipeline([results])
        return self.pipeline([results])
    
    def match_results(self, results, ref_results):
        match_indices, ref_match_indices = self._match_gts(
            results["ann_info"], ref_results["ann_info"]
        )
        results["ann_info"]["match_indices"] = match_indices
        ref_results["ann_info"]["match_indices"] = ref_match_indices
        return results, ref_results

    def _match_gts(self, ann, ref_ann):
        if "instance_ids" in ann:
            ins_ids = list(ann["instance_ids"])
            ref_ins_ids = list(ref_ann["instance_ids"])
            match_indices = np.array(ins_ids)
            ref_match_indices = np.array(ref_ins_ids)
        else:
            print('error')
        return match_indices, ref_match_indices
    
    def box_xyxy_to_cxcywh(self, boxes, size):
        bbox = torch.zeros_like(boxes)
        bbox[:,0] = (boxes[:,0] + boxes[:,2])/2
        bbox[:,1] = (boxes[:,1] + boxes[:,3])/2
        bbox[:,2] = (boxes[:,2] - boxes[:,0])
        bbox[:,3] = (boxes[:,3] - boxes[:,1])
        w_img = size[1]
        h_img = size[0]
        bbox = torch.div(bbox, torch.tensor([w_img, h_img, w_img, h_img], dtype=torch.float32, device=boxes.device))
        return bbox
    
    def ref_img_sampling(
        self, img_info, scope, num_ref_imgs=1, method="uniform", pesudo=False
    ):
        if num_ref_imgs != 1 or method != "uniform":
            raise NotImplementedError
        if img_info.get("frame_id", -1) < 0 or scope <= 0:
            ref_img_info = img_info.copy()
        else:
            vid_id = img_info["video_id"]
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            frame_id = img_info["frame_id"]
            if method == "uniform":
                left = max(0, frame_id - scope)
                right = min(frame_id + scope, len(img_ids) - 1)
                if pesudo:
                    valid_inds = img_ids[left : right + 1]
                else:
                    if len(img_ids) == 1:
                        valid_inds = img_ids[left : right + 1]
                    else:
                        valid_inds = (
                            img_ids[left:frame_id] + img_ids[frame_id + 1 : right + 1]
                        )
                ref_img_id = random.choice(valid_inds)
            ref_img_info = self.coco.loadImgs([ref_img_id])[0]
            ref_img_info["filename"] = ref_img_info["file_name"]
        return ref_img_info
    
    def prepare_results(self, img_info):
        ann_info = self.get_lvis_ann_info(img_info)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            idx = self.img_ids.index(img_info["id"])
            results["proposals"] = self.proposals[idx]
        return results

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = self.lvis.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def get_repeat_factors(self):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(self.img_ids)
        filter_empty_gt = True
        repeat_thr = 0.006
        for idx in range(num_images):
            cat_ids = set(self.get_cat_ids(idx))
            if len(cat_ids) == 0 and not filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            category_freq[k] = v / num_images
        # print(category_freq)
        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.get_cat_ids(idx))
            if len(cat_ids) == 0 and not filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.floor(repeat_factor))
        self.repeat_indices = repeat_indices
        self.ids = list(itemgetter(*repeat_indices)(self.img_ids))

    def __len__(self):
        if self.ids is None:
            return len(self.data_infos)
        else:
            return len(self.ids)


class LVIS_seqs_Dataset_val(TaoDataset):
    def __init__(self,
        args_mot = None,
        load_as_video=True,
        match_gts=True,
        skip_nomatch_pairs=True,
        key_img_sampler=dict(interval=1),
        ref_img_sampler=dict(scope=3, num_ref_imgs=1, method="uniform"),
        *args,
        **kwargs,):

        super(LVIS_seqs_Dataset_val, self).__init__(load_as_video=True,
            match_gts=True,
            skip_nomatch_pairs=True,
            key_img_sampler=key_img_sampler,
            ref_img_sampler=ref_img_sampler,
            *args,
            **kwargs,)
        
        self.CLASSES = CLASSES
        self.num_frames_per_batch = max(args_mot.sampler_lengths)
        self.sample_mode = args_mot.sample_mode
        self.sample_interval = args_mot.sample_interval
        self.video_dict = {}
        self.item_num = None

        if args_mot.filter_ignore:
            print('Training with ignore.', flush=True)

        self.item_num = len(self) - (self.num_frames_per_batch - 1) * self.sample_interval

        # video sampler.
        self.sampler_steps: list = args_mot.sampler_steps
        self.lengths: list = args_mot.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))

        self.num_samples = len(self)
        self._skip_type_keys = None

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    def pre_continuous_frames(self, start, end, interval=1):
        targets = []
        images = []
        for i in range(start, end, interval):
            img_i, targets_i = self._pre_single_frame(i)
            images.append(img_i)
            targets.append(targets_i)
        return images, targets

    def __getitem__(self, idx):
        results, ann_info = self.prepare_test_img(idx)
        frame_id = self.data_infos[idx]['frame_id']
        gt_instances = self.results_to_instances(results, frame_id, ann_info)
        return gt_instances
        
    def results_to_instances(self, results, frame_id, ann_info):
        img_shape_without_pad = results['img_metas'][0].data['pad_shape']
        ori_img_shape = results['img_metas'][0].data['ori_shape']
        gt_instance = Instances(img_shape_without_pad[:2])
        image = results['img'][0]
        gt_instance.labels = torch.as_tensor(ann_info['labels'], device=image.device)
        data={
            'imgs': [image],
            'gt_instances': [gt_instance],
            'info':[[frame_id, ori_img_shape]],
            'file_path':[ann_info['file_path']]
        }
        return data
    
    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(img_info)
        ann_info.update({'file_path':img_info['file_name']})
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results), ann_info
    
    def ref_img_sampling(
        self, img_info, scope, num_ref_imgs=1, method="uniform", pesudo=False
    ):
        if num_ref_imgs != 1 or method != "uniform":
            raise NotImplementedError
        if img_info.get("frame_id", -1) < 0 or scope <= 0:
            ref_img_info = img_info.copy()
        else:
            vid_id = img_info["video_id"]
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            frame_id = img_info["frame_id"]
            if method == "uniform":
                left = max(0, frame_id - scope)
                right = min(frame_id + scope, len(img_ids) - 1)
                if pesudo:
                    valid_inds = img_ids[left : right + 1]
                else:
                    if len(img_ids) == 1:
                        valid_inds = img_ids[left : right + 1]
                    else:
                        valid_inds = (
                            img_ids[left:frame_id] + img_ids[frame_id + 1 : right + 1]
                        )
                ref_img_id = random.choice(valid_inds)
            ref_img_info = self.coco.loadImgs([ref_img_id])[0]
            ref_img_info["filename"] = ref_img_info["file_name"]
        return ref_img_info
    
    def prepare_results(self, img_info):
        ann_info = self.get_ann_info(img_info)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            idx = self.img_ids.index(img_info["id"])
            results["proposals"] = self.proposals[idx]
        return results


def build(image_set, args, cfg):
    cfg.pop('type')
    if image_set == 'train':
        dataset = LVIS_seqs_Dataset(args, **cfg)
    elif image_set == 'val':
        dataset = LVIS_seqs_Dataset_val(args, **cfg)
    return dataset
