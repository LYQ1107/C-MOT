# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from collections import defaultdict
from torch.utils.data import DataLoader
import os
import numpy as np
import random
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.slconfig import SLConfig
from util.tool import load_model
from main import get_args_parser
from detectron2.structures import Instances
from datasets import build_dataset
import datasets.samplers as samplers
import util.misc as utils
from datasets.data_prefetcher import data_dict_to_cuda
from util.list_LVIS import CLASSES, novel_list_ori, COLORS
from mmcv.runner import get_dist_info
np.random.seed(2024)

def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None, mask=None):
    # Plots one bounding box on image img
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3.5, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3.5, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return img

def draw_bboxes(ori_img, bbox, identities=None, mask=None, offset=(0, 0), cvt_color=False, img_path=None):
    img = ori_img
    for i, box in enumerate(bbox):
        if mask is not None and mask.shape[0] > 0:
            m = mask[i]
        else:
            m = None
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
            label = int(box[5])
        else:
            score = None
            label = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS[id % len(COLORS)]
        label_str = '{:d} {:s}'.format(id, CLASSES[label])
        if label in novel_list_ori:
            print(f"found novel category! {label_str} location is in {img_path}")
        img = plot_one_box([x1, y1, x2, y2], img, color, label_str, score=score, mask=m)
    return img

def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Track(object):
    track_cnt = 0

    def __init__(self, box):
        self.box = box
        self.time_since_update = 0
        self.id = Track.track_cnt
        Track.track_cnt += 1
        self.miss = 0

    def miss_one_frame(self):
        self.miss += 1

    def clear_miss(self):
        self.miss = 0

    def update(self, box):
        self.box = box
        self.clear_miss()


class TRTR(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.active_trackers = {}
        self.inactive_trackers = {}
        self.disappeared_tracks = []

    def _remove_track(self, slot_id):
        self.inactive_trackers.pop(slot_id)
        self.disappeared_tracks.append(slot_id)

    def clear_disappeared_track(self):
        self.disappeared_tracks = []

    def update(self, dt_instances: Instances, target_size=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        dt_idxes = set(dt_instances.obj_idxes.tolist())
        track_idxes = set(self.active_trackers.keys()).union(set(self.inactive_trackers.keys()))
        matched_idxes = dt_idxes.intersection(track_idxes)

        unmatched_tracker = track_idxes - matched_idxes
        for track_id in unmatched_tracker:
            # miss in this frame, move to inactive_trackers.
            if track_id in self.active_trackers:
                self.inactive_trackers[track_id] = self.active_trackers.pop(track_id)
            self.inactive_trackers[track_id].miss_one_frame()
            if self.inactive_trackers[track_id].miss > 10:
                self._remove_track(track_id)

        for i in range(len(dt_instances)):
            idx = dt_instances.obj_idxes[i]
            bbox = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i + 1]], axis=-1)
            label = dt_instances.cls_idxes[i]
            if label != -1:

                # get a positive track.
                if idx in self.inactive_trackers:
                    # set state of track active.
                    self.active_trackers[idx] = self.inactive_trackers.pop(idx)
                if idx not in self.active_trackers:
                    # create a new track.
                    self.active_trackers[idx] = Track(idx)
                self.active_trackers[idx].update(bbox)

        ret = []
        if dt_instances.has('masks'):
            mask = []
        for i in range(len(dt_instances)):
            label = dt_instances.cls_idxes[i]
            if label != -1:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate(
                    [dt_instances.boxes[i], dt_instances.scores[i:i + 1], dt_instances.cls_idxes[i:i + 1]], axis=-1)
                ret.append(
                    np.concatenate((box_with_score, [id])).reshape(1, -1)) # TETA does not require +1
                if dt_instances.has('masks'):
                    mask.append(dt_instances.masks[i])

        if len(ret) > 0:
            if dt_instances.has('masks'):
                return np.concatenate(ret), np.concatenate(mask)
            return np.concatenate(ret)
        if dt_instances.has('masks'):      
            img_h, img_w = target_size
            return np.empty((0, 7)), np.empty((0, 1, img_h, img_w))
        return np.empty((0, 7))


class OVTR_inference(object):
    def __init__(self, args, cfg, model=None):
        self.args = args
        self.detr = model

        self.tr_tracker = TRTR()

        self.img_height = 800
        self.img_width = 1333

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.results = defaultdict(list)
        self.result_path_track = args.result_path_track
        self.cur_vis_img_path = args.vis_output
        self.root = cfg.data.val.img_prefix
        self.num_classes = len(CLASSES)
        self.vis_points = args.vis_points
        self.dataset_list = ["YFCC100M", "HACS", "BDD", "ArgoVerse", "AVA", "LaSOT", "Charades"]

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float, score_threshold: float) -> Instances:
        keep = (dt_instances.scores > score_threshold) & (dt_instances.disappear_time == 0)
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]
    
    def tracking_state_hyperparams(self, file_path, prob_threshold, score_threshold, filter_score_thresh, miss_tolerance, maximum_quantity, ious_thresh):
        indd = self.dataset_list.index(file_path.split("/")[1])
        self.detr.track_base.filter_score_thresh = filter_score_thresh[indd]
        self.detr.track_base.score_thresh = score_threshold[indd]
        self.detr.track_base.miss_tolerance = miss_tolerance[indd]
        self.detr.track_base.maximum_quantity = maximum_quantity
        self.detr.transformer.decoder.isol_ratio = 5
        self.detr.ious_thresh = ious_thresh[indd]
        return prob_threshold[indd], score_threshold[indd]
        
    def update_results_teta(self, bbox_xyxy, identities, labels, scores=None, masks=None, dt_instances=None):     
        if dt_instances.boxes.shape[0] == 0:
            bbox_result = [np.zeros((0, 5), dtype=np.float32) for i in range(self.num_classes)]
        else:
            if isinstance(dt_instances.boxes, torch.Tensor):
                bboxes1 = dt_instances.boxes.detach().cpu().numpy()
                labels1 = dt_instances.cls_idxes.detach().cpu().numpy()
                labels1 = np.array(labels1, dtype=int)
            bbox_result = [bboxes1[labels1 == i, :]
                for i in range(self.num_classes)]    

        if bbox_xyxy.shape[0] == 0:
            track_result = [np.zeros((0, 6), dtype=np.float32) for i in range(self.num_classes)]
        else:
            bboxes2 = np.array(bbox_xyxy, dtype=np.float32)
            labels2 = np.array(labels, dtype=int)
            ids = np.array(identities, dtype=int)
            track_result = [
                np.concatenate((ids[labels2 == i, None], bboxes2[labels2 == i, :]), axis=1)
                for i in range(self.num_classes)
                ]
                
        result = dict(bbox_results=bbox_result, track_results=track_result)
    
        for k, v in result.items():
            self.results[k].append(v)

    def update(self, dt_instances: Instances):
        ret = []
        if dt_instances.has('masks'):
            mask = []
        for i in range(len(dt_instances)):
            label = dt_instances.cls_idxes[i]
            if label != -1:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate(
                    [dt_instances.boxes[i], dt_instances.scores[i:i + 1], dt_instances.cls_idxes[i:i + 1]], axis=-1)
                ret.append(                     
                    np.concatenate((box_with_score, [id])).reshape(1, -1))
                if dt_instances.has('masks'):
                    mask.append(dt_instances.masks[i])

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    @staticmethod
    def visualize_img_with_bbox(save_path, img_path, dt_instances: Instances, ref_pts=None, vis_points=None):
        img = cv2.imread(img_path)
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, np.concatenate(
                [dt_instances.boxes, dt_instances.scores.reshape(-1, 1), dt_instances.cls_idxes.reshape(-1, 1)],
                axis=-1), dt_instances.obj_idxes, img_path = img_path)
        os.makedirs(os.path.join(save_path, img_path.split('/')[-3], img_path.split('/')[-2]), exist_ok = True)
        if vis_points:
            img_show = draw_points(img_show, ref_pts)
        save_path = os.path.join(save_path, img_path.split('/')[-3], img_path.split('/')[-2], img_path.split('/')[-1])
        cv2.imwrite(save_path, img_show)

    def detect(self, prob_threshold=0.6, score_threshold=0.5, filter_score_thresh=0.5, miss_tolerance=5, maximum_quantity=60, area_threshold=100, ious_thresh=0.3,
               vis=False, data=None, track_instances=None, info=None, file_path=None):
        frame_id = info[0]
        prob_threshold, score_threshold = self.tracking_state_hyperparams(file_path, prob_threshold, score_threshold, filter_score_thresh, miss_tolerance, maximum_quantity, ious_thresh)

        res = self.detr.inference_single_image(data, track_instances, frame_id=frame_id, ori_img_size=info[1])
        track_instances = res['track_instances']
        dt_instances = track_instances.to(torch.device('cpu'))

        # filter det instances by score.
        dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold, score_threshold)
        dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

        if vis:
            all_ref_pts = tensor_to_numpy(res['ref_pts'][0, :, :2])
            self.visualize_img_with_bbox(self.cur_vis_img_path, os.path.join(self.root, file_path), dt_instances, ref_pts=all_ref_pts, vis_points=self.vis_points)

        tracker_outputs = self.update(dt_instances)

        self.update_results_teta(
                            bbox_xyxy=tracker_outputs[:, :4],
                            identities=tracker_outputs[:, 6],
                            labels=tracker_outputs[:, 5],
                            scores=tracker_outputs[:, 4],
                            masks=None,
                            dt_instances=dt_instances)

        if track_instances is not None:
            track_instances.remove('boxes')
            track_instances.remove('labels')
        return track_instances


def eval(args, cfg):
    utils.init_distributed_mode(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cfg.data.test.test_mode = True
    cfg.device = "cuda" #if not cpu_only else "cpu"
    torch.manual_seed(args.seed)

    # load model and weights
    model, _, = build_model(args, cfg)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    model = load_model(model, args.pretrained)
    model.eval()
    model = model.cuda()

    dataset_val = build_dataset(image_set='val', args=args, cfg=cfg.data.test)
    if args.distributed:
        if args.cache_mode:
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    collate_fn = utils.mot_collate_fn
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    
    tracker = OVTR_inference(args, cfg, model=model)

    torch.manual_seed(args.seed)

    tracker.result_path_track = os.path.abspath(tracker.result_path_track)
    os.makedirs((tracker.result_path_track), exist_ok = True)
    track_instances = None

    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(data_loader_val)):   
            info = data_dict.pop('info')[0]
            file_path = data_dict.pop('file_path')[0]
            data_dict = data_dict_to_cuda(data_dict, device=model.text_embeddings.device)
            track_instances = tracker.detect(vis=args.vis, data=data_dict, track_instances=track_instances, info=info, 
                                             prob_threshold=args.score_thresh, score_threshold=args.score_thresh, filter_score_thresh=args.filter_score_thresh, 
                                             miss_tolerance=args.miss_tolerance, maximum_quantity=args.maximum_quantity, area_threshold=1, ious_thresh=args.ious_thresh,
                                             file_path=file_path)

    resfile_path = tracker.result_path_track
    print('Inference completed')

    print('Start TETA')
    content_track = tracker.results['track_results']
    outputs={"track_results":content_track, "bbox_results":None}
    
    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.eval_options is None else args.eval_options
        
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        eval_kwargs.resfile_path = resfile_path
        print(dataset_val.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OVTR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    cfg = SLConfig.fromfile(args.config_file)

    eval(args, cfg)
