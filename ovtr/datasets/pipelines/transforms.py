# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from OVTrack (https://github.com/SysCV/ovtrack)
# ------------------------------------------------------------------------
import copy
from tkinter import Y
import mmcv
import cv2
import numpy as np
from mmdet.core import find_inside_bboxes
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import (MixUp, Mosaic, Normalize, Pad,
                                      RandomAffine, RandomFlip, Resize,
                                      YOLOXHSVRandomAug, CutOut)
from numpy import random
import torch
import math


@PIPELINES.register_module()
class SeqResize(Resize):
    def __init__(self, share_params=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        outs, scale = [], None
        for i, _results in enumerate(results):
            if self.share_params and i > 0:
                _results["scale"] = scale
            _results = super().__call__(_results)
            if self.share_params and i == 0:
                scale = _results["scale"]
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SizeRatioFilter():
    def __call__(self, results):
        outs = []
        for i, _results in enumerate(results):
            outs.append(_results)
        return outs

@PIPELINES.register_module()
class RandomOcclusion(CutOut):
    def __init__(self, share_params=True, holes_area=0, erase_threshold=20, occlusion_ratio=None, *args, **kwargs):
        self.holes_area = holes_area
        super().__init__(*args, **kwargs)
        self.share_params = share_params
        self.erase_threshold = erase_threshold
        self.occlusion_ratio = occlusion_ratio

    def __call__(self, results, min_wh):
        """Call function to drop some regions of image."""    
        outs = []
        # not pair
        self.h, self.w, c = results[0]['img'].shape
        if (min_wh[0] < self.erase_threshold) or (min_wh[1] < self.erase_threshold):
            candidates = self.candidates[0][0]
        else:
            candidates = self.candidates[0][1]

        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 5)
        n_holes = min(int(self.holes_area / (min_wh[0] * min_wh[1])), n_holes)

        for _result in results:
            fill_indx = np.random.randint(0, len(self.fill_in))
            fill_in = self.fill_in[fill_indx]
            out = self.Random_erase(_result, n_holes, min_wh, candidates, fill_in)
            out = self.random_occlusion(out, fill_in)
            outs.append(out)
        return outs
    
    def random_occlusion(self, result, fill_in):
        if 'mix_results' in result:
            clear_ids = np.concatenate((result['ann_info']['gt_clear_ids'], result['mix_results'][0]['ann_info']['gt_clear_ids'],\
                    result['mix_results'][1]['ann_info']['gt_clear_ids'], result['mix_results'][2]['ann_info']['gt_clear_ids']))
        else:
            clear_ids = result['ann_info']['gt_clear_ids']
        cur_ids = result['gt_match_indices']
        indices = [np.where(cur_ids == id)[0] for id in clear_ids]
        valid_indices = [idx[0] for idx in indices if len(idx) > 0]
        
        bboxes = result['gt_bboxes'][valid_indices]
        valid_num = bboxes.shape[0]
        choice_num = int(min(self.occlusion_ratio[0] * result['gt_bboxes'].shape[0], self.occlusion_ratio[1] * valid_num))
        if choice_num != 0:
            random_indices = np.random.choice(np.arange(valid_num), size=choice_num, replace=False)
            selected_bboxes = bboxes[random_indices]
            for _occlusion in selected_bboxes:
                result['img'][math.ceil(_occlusion[1]): math.floor(_occlusion[3]), math.ceil(_occlusion[0]): math.floor(_occlusion[2]), :] = fill_in

            # Remove corresponding annotations
            valid_indices = np.array(valid_indices)
            occlusion_indices = valid_indices[random_indices]
            occlusion_mask = np.zeros_like(cur_ids, dtype=bool)
            occlusion_mask[occlusion_indices] = True
            result['gt_bboxes'] = result['gt_bboxes'][~occlusion_mask]
            result['gt_labels'] = result['gt_labels'][~occlusion_mask]
            result['gt_match_indices'] = result['gt_match_indices'][~occlusion_mask]
        return  result

    def Random_erase(self, result, n_holes, min_wh, candidates, fill_in):
        """Call function to drop some regions of image."""
        for _ in range(n_holes):
            x1 = np.random.randint(0, self.w)
            y1 = np.random.randint(0, self.h)

            candidates = np.random.uniform(candidates, 0.8)
            cutout_w = max(int(candidates * min_wh[0]),1)
            cutout_h = max(int(candidates * min_wh[1]),1)

            x2 = np.clip(x1 + cutout_w, 0, self.w)
            y2 = np.clip(y1 + cutout_h, 0, self.h)
            
            result['img'][y1: y2, x1: x2, :] = fill_in
        return result

@PIPELINES.register_module()
class SeqNormalize(Normalize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomFlip(RandomFlip):
    def __init__(self, share_params, deny_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params
        self.deny_ratio = deny_ratio

    def __call__(self, results, unique=False):
        if unique:
            random_value = random.random()
            if random_value < self.deny_ratio:  
                unique = False
        if not unique:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1) + [
                    non_flip_ratio
                ]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            flip = cur_dir is not None
            flip_direction = cur_dir

            for _results in results:
                _results["flip"] = flip
                _results["flip_direction"] = flip_direction

        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqPad(Pad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqRandomCrop(object):
    def __init__(
        self,
        crop_size,
        allow_negative_crop=False,
        share_params=False,
        bbox_clip_border=True,
    ):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.allow_negative_crop = allow_negative_crop
        self.share_params = share_params
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            "gt_bboxes": ["gt_labels", "gt_instance_ids"],
            "gt_bboxes_ignore": ["gt_labels_ignore", "gt_instance_ids_ignore"],
        }
        self.bbox2mask = {
            "gt_bboxes": "gt_masks",
            "gt_bboxes_ignore": "gt_masks_ignore",
        }
        self.bbox_clip_border = bbox_clip_border

    def get_offsets(self, img):
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return offset_h, offset_w

    def random_crop(self, results, offsets=None):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if offsets is not None:
                offset_h, offset_w = offsets
            else:
                offset_h, offset_w = self.get_offsets(img)
            results["img_info"]["crop_offsets"] = (offset_h, offset_w)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results["img_shape"] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get("bbox_fields", []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array(
                [offset_w, offset_h, offset_w, offset_h], dtype=np.float32
            )
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # self.allow_negative_crop is False, skip this image.
            if (
                key == "gt_bboxes"
                and not valid_inds.any()
                and not self.allow_negative_crop
            ):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_keys = self.bbox2label.get(key)
            for label_key in label_keys:
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2])
                )

        # crop semantic seg
        for key in results.get("seg_fields", []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]
        return results

    def __call__(self, results):
        if self.share_params:
            offsets = self.get_offsets(results[0]["img"])
        else:
            offsets = None

        outs = []
        for _results in results:
            _results = self.random_crop(_results, offsets)
            if _results is None:
                return None
            outs.append(_results)

        return outs


@PIPELINES.register_module()
class SeqPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        share_params=True,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.share_params = share_params
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def get_params(self):
        params = dict()
        # delta
        if np.random.randint(2):
            params["delta"] = np.random.uniform(
                -self.brightness_delta, self.brightness_delta
            )
        else:
            params["delta"] = None
        # mode
        mode = np.random.randint(2)
        params["contrast_first"] = True if mode == 1 else 0
        # alpha
        if np.random.randint(2):
            params["alpha"] = np.random.uniform(
                self.contrast_lower, self.contrast_upper
            )
        else:
            params["alpha"] = None
        # saturation
        if np.random.randint(2):
            params["saturation"] = np.random.uniform(
                self.saturation_lower, self.saturation_upper
            )
        else:
            params["saturation"] = None
        # hue
        if np.random.randint(2):
            params["hue"] = np.random.uniform(-self.hue_delta, self.hue_delta)
        else:
            params["hue"] = None
        # swap
        if np.random.randint(2):
            params["permutation"] = np.random.permutation(3)
        else:
            params["permutation"] = None
        return params

    def photo_metric_distortion(self, results, params=None):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if params is None:
            params = self.get_params()
        results["img_info"]["color_jitter"] = params

        if "img_fields" in results:
            assert results["img_fields"] == ["img"], "Only single img_fields is allowed"
        img = results["img"]
        assert img.dtype == np.float32, (
            "PhotoMetricDistortion needs the input image of dtype np.float32,"
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        )
        # random brightness
        if params["delta"] is not None:
            img += params["delta"]

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if params["contrast_first"]:
            if params["alpha"] is not None:
                img *= params["alpha"]

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if params["saturation"] is not None:
            img[..., 1] *= params["saturation"]

        # random hue
        if params["hue"] is not None:
            img[..., 0] += params["hue"]
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if not params["contrast_first"]:
            if params["alpha"] is not None:
                img *= params["alpha"]

        # randomly swap channels
        if params["permutation"] is not None:
            img = img[..., params["permutation"]]

        results["img"] = img
        return results

    def __call__(self, results):
        if self.share_params:
            params = self.get_params()
        else:
            params = None

        outs = []
        for _results in results:
            _results = self.photo_metric_distortion(_results, params)
            outs.append(_results)

        return outs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@PIPELINES.register_module()
class DynamicMosaic(Mosaic):
    def __init__(self, shuffle_ratio=0.5, dislocation_ratio=0.5, single_ratio_range=None, *args, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.shuffle_ratio = shuffle_ratio
        self.dislocation_ratio = dislocation_ratio
        self.single_ratio_range = single_ratio_range
        horizontal_dislocation = [self.img_scale[1], 0, self.img_scale[1], 0, -self.img_scale[1], 0, -self.img_scale[1], 0]
        vertical_dislocation = [0, self.img_scale[0], 0, -self.img_scale[0], 0, self.img_scale[0], 0, -self.img_scale[0]]
        self.dislocation = (horizontal_dislocation, vertical_dislocation)

    def __call__(self, results):
        outs = []
        self.loc_strs = ["top_left", "top_right", "bottom_left", "bottom_right"]
        for _results in results:
            random_value = random.random()
            if random_value < self.shuffle_ratio:  
                idx1, idx2 = random.choice(len(self.loc_strs), 2, replace=False)
                self.loc_strs[idx1], self.loc_strs[idx2] = self.loc_strs[idx2], self.loc_strs[idx1] 

            dislocation_ratio = random.uniform(-self.dislocation_ratio, self.dislocation_ratio) 
            direction = random.choice([0,1]) 
            _dislocation = self.dislocation[direction]
            self._dislocation = [int(item * dislocation_ratio) for item in _dislocation] 

            _results = self._mosaic_transform(_results)
            outs.append(_results)
        return outs

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert "mix_results" in results
        mosaic_labels = []
        mosaic_bboxes = []
        mosaic_inds = []
        if len(results["img"].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results["img"].dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results["img"].dtype,
            )

        # mosaic center x, y
        center_x = int(self.img_scale[1])
        center_y = int(self.img_scale[0])
        center_position = (center_x, center_y)

        for i, loc in enumerate(self.loc_strs):
            if i == 0:
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results["mix_results"][i - 1])

            img_i = results_patch["img"]
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            single_ratio = random.uniform(self.single_ratio_range[0], self.single_ratio_range[1])
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i * single_ratio), int(h_i * scale_ratio_i * single_ratio))
            )

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1]
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c: y2_c, x1_c: x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch["gt_bboxes"]
            gt_labels_i = results_patch["gt_labels"]
            gt_match_indices_i = results_patch["gt_match_indices"]

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = scale_ratio_i  * single_ratio * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = scale_ratio_i  * single_ratio * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)
            mosaic_inds.append(gt_match_indices_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            mosaic_inds = np.concatenate(mosaic_inds, 0)
            # mosaic_inds = np.arange(0, len(mosaic_labels))

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(
                    mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[1]
                )
                mosaic_bboxes[:, 1::2] = np.clip(
                    mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[0]
                )

            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels, mosaic_inds = self._filter_box_candidates(
                    mosaic_bboxes, mosaic_labels, mosaic_inds
                )

        # remove outside bboxes
        inside_inds = find_inside_bboxes(
            mosaic_bboxes, 2 * self.img_scale[0], 2 * self.img_scale[1]
        )
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]
        mosaic_inds = mosaic_inds[inside_inds]

        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape
        results["gt_bboxes"] = mosaic_bboxes
        results["gt_labels"] = mosaic_labels
        results["gt_match_indices"] = mosaic_inds

        return results
    
    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.
        ...
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0] + self._dislocation[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1] + self._dislocation[1], 0), \
                             center_position_xy[0] + self._dislocation[0], \
                             center_position_xy[1] + self._dislocation[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0] + self._dislocation[2], \
                             max(center_position_xy[1] - img_shape_wh[1] + self._dislocation[3], 0), \
                             min(center_position_xy[0] + img_shape_wh[0] + self._dislocation[2],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1] + self._dislocation[3]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0] + self._dislocation[4], 0), \
                             center_position_xy[1] + self._dislocation[5], \
                             center_position_xy[0] + self._dislocation[4], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1] + self._dislocation[5])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0] + self._dislocation[6], \
                             center_position_xy[1] + self._dislocation[7], \
                             min(center_position_xy[0] + img_shape_wh[0] + self._dislocation[6],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1] + self._dislocation[7])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def _filter_box_candidates(self, bboxes, labels, inds):
        """Filter out bboxes too small after Mosaic."""
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds], inds[valid_inds]


@PIPELINES.register_module()
class SeqRandomAffine(RandomAffine):
    def __init__(self, scaling_ratio_range_Mosaic=None, max_translate=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_direction_list = [[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]]
        self.ratio_threshold = 0.3
        self.scaling_ratio_range_Mosaic = scaling_ratio_range_Mosaic
        self.max_translate = max_translate

    def __call__(self, results, area_ratio = None):
        if 'mix_results' in results[0]:
            self.scaling_ratio = self.scaling_ratio_range_Mosaic
        else:
            self.scaling_ratio = self.scaling_ratio_range

        # Prepare for translation
        # max_translate_ratio = self.max_translate_ratio * 4
        max_translate_ratio = self.max_translate[len(results) - 2]
        trans_x_max = random.uniform(max_translate_ratio / 3, max_translate_ratio) # /2
        trans_y_max = random.uniform(max_translate_ratio / 3, max_translate_ratio) # /2
        if len(results) > 4:
            turning_point = random.randint(1, len(results)-1)
            translate_type_ids = random.choice(range(0, 8), size=2)
        else:
            turning_point = None
            translate_type_ids = random.choice(range(0, 8), size=1)
        self.trans_list = self.get_trans_coordinates(trans_x_max, trans_y_max, translate_type_ids, turning_point, len(results))
        outs = []
        for i, _results in enumerate(results):
            _results = self.random_affine_v2(_results, i, area_ratio[i])
        # for _results in results:
            # _results = self.random_affine(_results)
            outs.append(_results)
        return outs

    def get_trans_coordinates(self, x_range, y_range, type_ids, turning_point, frame_num):
        if turning_point is not None:
            trans_list_pre = self.direction(turning_point, self.trajectory_direction_list[type_ids[0]])
            trans_list_1 = copy.deepcopy(trans_list_pre)
            trans_list_2 = self.direction(frame_num - turning_point, self.trajectory_direction_list[type_ids[1]], coords=trans_list_pre[-1])
            trans_list = trans_list_1 + trans_list_2
        else:
            trans_list = self.direction(frame_num, self.trajectory_direction_list[type_ids[0]])

        # get trans coordinates
        trans_tensor = torch.tensor(trans_list, dtype=torch.float32) 

        x_coords = trans_tensor[:, 0]  
        y_coords = trans_tensor[:, 1]  

        x_min, x_max = (x_coords.min()).item(), (x_coords.max()).item()
        y_min, y_max = (y_coords.min()).item(), (y_coords.max()).item()
        
        x_scale = (x_max - x_min)/x_range/2
        y_scale = (y_max - y_min)/y_range/2

        x_center = (x_max + x_min)/2
        y_center = (y_max + y_min)/2

        new_trans_list=[]
        for point in trans_list:  
            new_point = [(point[0] - x_center)/x_scale if x_scale!=0 else 0, (point[1] - y_center)/y_scale if y_scale!=0 else 0]
            new_trans_list.append(new_point)  
        
        return new_trans_list

    def direction(self, frame_num, direction, coords = [0, 0]):
        trans_list = []
        for i in range(frame_num):
            coords[0] += direction[0]
            coords[1] += direction[1]
            trans_list.append(coords.copy())
        return trans_list
    
    def random_affine_v2(self, results, frame_id, area_ratio):
        img = results["img"]
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Rotation
        rotation_degree = random.uniform(
            -self.max_rotate_degree, self.max_rotate_degree
        )
        rotation_matrix = self._get_rotation_matrix(rotation_degree, width, height)

        # Scaling
        scaling_ratio = random.uniform(
            self.scaling_ratio[0], self.scaling_ratio[1]
        )
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = self.trans_list[frame_id][0] * width
        trans_y = self.trans_list[frame_id][1] * height
        
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix

        img = cv2.warpPerspective(
            img, warp_matrix, dsize=(width, height), borderValue=self.border_val
        )
        results["img"] = img
        results["img_shape"] = img.shape

        for key in results.get("bbox_fields", []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                # homogeneous coordinates
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_bboxes = np.vstack((xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                if self.bbox_clip_border:
                    warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)

                # remove outside bbox
                valid_index = self.find_inside_bboxes_v2(warp_bboxes, height, width, area_ratio, scaling_ratio, self.ratio_threshold)
                if not self.skip_filter:
                    # filter bboxes
                    filter_index = self.filter_gt_bboxes(
                        bboxes * scaling_ratio, warp_bboxes
                    )
                    valid_index = valid_index & filter_index

                results[key] = warp_bboxes[valid_index]
                if key in ["gt_bboxes"]:
                    if "gt_labels" in results:
                        results["gt_labels"] = results["gt_labels"][valid_index]
                        results["gt_match_indices"] = results["gt_match_indices"][
                            valid_index
                        ]

                if "gt_masks" in results:
                    raise NotImplementedError("RandomAffine only supports bbox.")
        return results
    
    @staticmethod
    def _get_rotation_matrix(rotate_degrees, width, height):
        radian = math.radians(rotate_degrees)

        center_x, center_y = width / 2, height / 2

        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)

        translate_to_origin = np.array(
            [[1, 0, -center_x],
            [0, 1, -center_y],
            [0, 0, 1]],
            dtype=np.float32
        )
        
        translate_back = np.array(
            [[1, 0, center_x],
            [0, 1, center_y],
            [0, 0, 1]],
            dtype=np.float32
        )
        
        final_matrix = translate_back @ rotation_matrix @ translate_to_origin
        return final_matrix

    @staticmethod
    def find_inside_bboxes_v2(bboxes, img_h, img_w, ori_area_ratio, scaling_ratio, ratio_threshold):
        """Find bboxes that express sufficient semantic information
        """
        cur_area_ratio = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) / (scaling_ratio ** 2)
        ratio = cur_area_ratio / ori_area_ratio
        inside_inds = ratio > ratio_threshold
        return inside_inds

    def random_affine(self, results):
        img = results["img"]
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Rotation
        rotation_degree = random.uniform(
            -self.max_rotate_degree, self.max_rotate_degree
        )
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(
            self.scaling_ratio_range[0], self.scaling_ratio_range[1]
        )
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        )
        trans_y = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        )
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix

        img = cv2.warpPerspective(
            img, warp_matrix, dsize=(width, height), borderValue=self.border_val
        )
        results["img"] = img
        results["img_shape"] = img.shape

        for key in results.get("bbox_fields", []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                # homogeneous coordinates
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_bboxes = np.vstack((xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                if self.bbox_clip_border:
                    warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)

                # remove outside bbox
                valid_index = find_inside_bboxes(warp_bboxes, height, width)
                if not self.skip_filter:
                    # filter bboxes
                    filter_index = self.filter_gt_bboxes(
                        bboxes * scaling_ratio, warp_bboxes
                    )
                    valid_index = valid_index & filter_index

                results[key] = warp_bboxes[valid_index]
                if key in ["gt_bboxes"]:
                    if "gt_labels" in results:
                        results["gt_labels"] = results["gt_labels"][valid_index]
                        results["gt_match_indices"] = results["gt_match_indices"][
                            valid_index
                        ]

                if "gt_masks" in results:
                    raise NotImplementedError("RandomAffine only supports bbox.")
        return results

@PIPELINES.register_module()
class SeqMixUp(MixUp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = self._mixup_transform(_results)
            outs.append(_results)
        return outs

    def _mixup_transform(self, results):
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert "mix_results" in results
        assert len(results["mix_results"]) == 1, "MixUp only support 2 images now !"

        if results["mix_results"][0]["gt_bboxes"].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results["mix_results"][0]
        retrieve_img = retrieve_results["img"]

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = (
                np.ones(
                    (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                    dtype=retrieve_img.dtype,
                )
                * self.pad_val
            )
        else:
            out_img = (
                np.ones(self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val
            )

        # 1. keep_ratio resize
        scale_ratio = min(
            self.dynamic_scale[0] / retrieve_img.shape[0],
            self.dynamic_scale[1] / retrieve_img.shape[1],
        )
        retrieve_img = mmcv.imresize(
            retrieve_img,
            (
                int(retrieve_img.shape[1] * scale_ratio),
                int(retrieve_img.shape[0] * scale_ratio),
            ),
        )

        # 2. paste
        out_img[: retrieve_img.shape[0], : retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(
            out_img,
            (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)),
        )

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results["img"]
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[
            y_offset : y_offset + target_h, x_offset : x_offset + target_w
        ]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results["gt_bboxes"]
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(
                retrieve_gt_bboxes[:, 0::2], 0, origin_w
            )
            retrieve_gt_bboxes[:, 1::2] = np.clip(
                retrieve_gt_bboxes[:, 1::2], 0, origin_h
            )

        if is_filp:
            retrieve_gt_bboxes[:, 0::2] = (
                origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1]
            )

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = cp_retrieve_gt_bboxes[:, 0::2] - x_offset
        cp_retrieve_gt_bboxes[:, 1::2] = cp_retrieve_gt_bboxes[:, 1::2] - y_offset
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes[:, 0::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 0::2], 0, target_w
            )
            cp_retrieve_gt_bboxes[:, 1::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 1::2], 0, target_h
            )

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_labels = retrieve_results["gt_labels"]
        retrieve_gt_match_indices = retrieve_results["gt_match_indices"] + 999
        if not self.skip_filter:
            keep_list = self._filter_box_candidates(
                retrieve_gt_bboxes.T, cp_retrieve_gt_bboxes.T
            )

            retrieve_gt_labels = retrieve_gt_labels[keep_list]
            retrieve_gt_match_indices = retrieve_gt_match_indices[keep_list]
            cp_retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]

        mixup_gt_bboxes = np.concatenate(
            (results["gt_bboxes"], cp_retrieve_gt_bboxes), axis=0
        )
        mixup_gt_labels = np.concatenate(
            (results["gt_labels"], retrieve_gt_labels), axis=0
        )
        mixup_gt_match_indices = np.concatenate(
            (results["gt_match_indices"], retrieve_gt_match_indices), axis=0
        )

        # remove outside bbox
        inside_inds = find_inside_bboxes(mixup_gt_bboxes, target_h, target_w)
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_labels = mixup_gt_labels[inside_inds]
        mixup_gt_match_indices = mixup_gt_match_indices[inside_inds]

        results["img"] = mixup_img.astype(np.uint8)
        results["img_shape"] = mixup_img.shape
        results["gt_bboxes"] = mixup_gt_bboxes
        results["gt_labels"] = mixup_gt_labels
        results["gt_match_indices"] = mixup_gt_match_indices

        return results


@PIPELINES.register_module()
class SeqYOLOXHSVRandomAug(YOLOXHSVRandomAug):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs
