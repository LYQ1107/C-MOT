# -*- coding: UTF-8 -*-
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

def draw_bboxes(ori_img, bbox, identities=None, mask=None, offset=(0, 0)):
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
        indd = 0
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
    
    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    def visualize_img_with_bbox(self, save_path, ori_frame, dt_instances: Instances, ref_pts=None, vis_points=None):
        img = ori_frame
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, np.concatenate(
                [dt_instances.boxes, dt_instances.scores.reshape(-1, 1), dt_instances.cls_idxes.reshape(-1, 1)],
                axis=-1), dt_instances.obj_idxes)
        self.video_writer.write(img_show)

    def init_video_writer(self, output_path, frame_width, frame_height, fps):
        """Initialize the VideoWriter object."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        os.makedirs(output_path, exist_ok = True)
        self.video_writer = cv2.VideoWriter(output_path + '/demo_output.mp4', fourcc, fps, (frame_width, frame_height))
        if not self.video_writer.isOpened():
            print("Error: Could not initialize VideoWriter.")
            return False
        return True
    
    def detect(self, prob_threshold=0.6, score_threshold=0.5, filter_score_thresh=0.5, miss_tolerance=5, maximum_quantity=60, area_threshold=100, ious_thresh=0.3,
               vis=False, data=None, track_instances=None, info=None, ori_frame=None, frame_id=None, frame_width=640, frame_height=480):
        prob_threshold, score_threshold = self.tracking_state_hyperparams(None, prob_threshold, score_threshold, filter_score_thresh, miss_tolerance, maximum_quantity, ious_thresh)

        res = self.detr.inference_single_image({"imgs":[data[0]]}, track_instances, frame_id=frame_id, ori_img_size=[frame_height, frame_width, 3])
        track_instances = res['track_instances']
        dt_instances = track_instances.to(torch.device('cpu'))

        # filter det instances by score.
        dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold, score_threshold)
        dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

        all_ref_pts = tensor_to_numpy(res['ref_pts'][0, :, :2])
        self.visualize_img_with_bbox(self.cur_vis_img_path, ori_frame, dt_instances, ref_pts=all_ref_pts, vis_points=self.vis_points)

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


def eval(args, cfg, video_path):
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
    
    tracker = OVTR_inference(args, cfg, model=model)

    torch.manual_seed(args.seed)

    track_instances = None
    frame_id = 0
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not tracker.init_video_writer(args.vis_output, frame_width, frame_height, fps):
        return
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    with torch.no_grad():
        with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cur_frame, ori_frame = tracker.init_img(frame)
                track_instances = tracker.detect(vis=args.vis, data=cur_frame.cuda().float(), track_instances=track_instances,
                                                prob_threshold=args.score_thresh, score_threshold=args.score_thresh, filter_score_thresh=args.filter_score_thresh, 
                                                miss_tolerance=args.miss_tolerance, maximum_quantity=args.maximum_quantity, area_threshold=1, ious_thresh=args.ious_thresh,
                                                frame_id=frame_id, ori_frame=ori_frame, frame_width=frame_width, frame_height=frame_height
                                                )
                pbar.update(1)
                frame_id += 1             
    print('demo inference complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OVTR demo', parents=[get_args_parser()])
    args = parser.parse_args()
    cfg = SLConfig.fromfile(args.config_file)
    video_path = '../video/track_demo.mp4'

    eval(args, cfg, video_path)
