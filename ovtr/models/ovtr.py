# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
"""
DETR model and criterion classes.
"""

import torch
import torch.nn.functional as F
from torch import nn
import math
from typing import List
import copy
from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid,)

from detectron2.structures import Instances, Boxes, matched_boxlist_iou
from .backbone import build_backbone
from .matcher import build_matcher
from .matcher import QualityAwarePLMatcher
from .transformer import build_transformer
from .updater import build as build_updater
from .deformable_detr import SetCriterion
from .segmentation import sigmoid_focal_loss

from util.clip_utils import load_embeddings
from .utils import MLP, protect_det_preds, protect_track_preds, preprocess_for_masks
from cmot.losses.supervision import ClassificationSupervision, LossAccumulator, focal_binary_loss
from util.list_LVIS import Frequency_list_total_1, Frequency_list_70, novel_class

class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, processor_dct=None):
        super().__init__()
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)
        if track_instances.has("assigned_column"):
            labels = track_instances.assigned_column.to(device=prob.device, dtype=torch.long)
            scores = prob.gather(1, labels[:, None]).squeeze(1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes.clamp(0, 1)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels

        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        return track_instances

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.6, miss_tolerance=5, maximum_quantity=50,
                 birth_threshold=None, keep_threshold=None, export_threshold=None):
        # Preserve upstream attribute names while exposing V3's separate
        # birth, internal-keep and public-export thresholds.
        self.birth_threshold = float(0.6 if birth_threshold is None and score_thresh is None else
                                     (score_thresh if birth_threshold is None else birth_threshold))
        self.keep_threshold = float(0.6 if keep_threshold is None and filter_score_thresh is None else
                                    (filter_score_thresh if keep_threshold is None else keep_threshold))
        self.export_threshold = float(self.birth_threshold if export_threshold is None else export_threshold)
        self.score_thresh = self.birth_threshold
        self.filter_score_thresh = self.keep_threshold
        self.miss_tolerance = int(5 if miss_tolerance is None else miss_tolerance)
        self.max_obj_id = 0
        self.maximum_quantity = int(50 if maximum_quantity is None else maximum_quantity)

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, _track_discard=None, is_repeat=False):
        if not track_instances.has("suppressed_this_frame"):
            track_instances.suppressed_this_frame = torch.zeros(len(track_instances), dtype=torch.bool, device=track_instances.scores.device)
        suppressed = track_instances.suppressed_this_frame.bool()
        if _track_discard is not None and len(_track_discard):
            if _track_discard.dtype == torch.bool and len(_track_discard) == len(track_instances):
                suppressed = suppressed | _track_discard
            elif _track_discard.dtype != torch.bool:
                discard_mask = torch.zeros(len(track_instances), dtype=torch.bool, device=track_instances.scores.device)
                discard_mask[_track_discard] = True
                suppressed = suppressed | discard_mask
            track_instances.suppressed_this_frame = suppressed
        scores = track_instances.scores
        valid_box = (torch.isfinite(track_instances.pred_boxes).all(dim=-1)
                     if track_instances.has("pred_boxes") else torch.ones_like(scores, dtype=torch.bool))
        observed_before = (scores >= self.keep_threshold) & (~suppressed) & valid_box
        new_birth_before = (scores >= self.birth_threshold) & (~suppressed) & valid_box
        existing_before = track_instances.obj_idxes >= 0
        valid_indx = new_birth_before | existing_before | (track_instances.obj_idxes == -2)
        track_instances = track_instances[valid_indx]
        observed = observed_before[valid_indx]
        new_birth = new_birth_before[valid_indx]

        if len(track_instances) > self.maximum_quantity:
            top_indices = self.quantity_filter(track_instances, self.maximum_quantity)
            track_instances = track_instances[top_indices]
            observed = observed[top_indices]
            new_birth = new_birth[top_indices]

        if not track_instances.has("observed_this_frame"):
            track_instances.observed_this_frame = torch.zeros(
                len(track_instances), dtype=torch.bool, device=scores.device)
        if not track_instances.has("export_valid"):
            track_instances.export_valid = torch.zeros(
                len(track_instances), dtype=torch.bool, device=scores.device)
        if not track_instances.has("export_scores"):
            track_instances.export_scores = track_instances.scores.clone()
        for i in range(len(track_instances)):
            obj_id = int(track_instances.obj_idxes[i].item())
            if obj_id == -2:
                track_instances.observed_this_frame[i] = False
                track_instances.export_valid[i] = False
                continue
            if obj_id == -1 and bool(new_birth[i].item()) and not bool(track_instances.suppressed_this_frame[i].item()):
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
                obj_id = int(track_instances.obj_idxes[i].item())
            is_observed = bool(observed[i].item()) and not bool(track_instances.suppressed_this_frame[i].item())
            track_instances.observed_this_frame[i] = bool(obj_id >= 0 and is_observed)
            if obj_id >= 0:
                if is_observed:
                    track_instances.disappear_time[i] = 0
                elif not is_repeat:
                    # Suppression and low score each age once here; the
                    # duplicate helper never touches disappear_time.
                    track_instances.disappear_time[i] += 1
                    if int(track_instances.disappear_time[i].item()) >= self.miss_tolerance:
                        track_instances.obj_idxes[i] = -1
                        obj_id = -1
            track_instances.export_valid[i] = bool(
                obj_id >= 0 and is_observed and float(track_instances.scores[i].item()) >= self.export_threshold)
        return track_instances

        # Kept unreachable for source compatibility with the upstream patch;
        # the V3 implementation above owns all runtime state transitions.
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -2:
                continue
            elif track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh and not bool(track_instances.suppressed_this_frame[i]):
                # print("track {} has score {:.2f}, assign obj_id {}, cls is {}".format(i, track_instances.scores[i], self.max_obj_id, track_instances.cls_idxes[i]))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and bool(track_instances.suppressed_this_frame[i]):
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    track_instances.obj_idxes[i] = -1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh and is_repeat is False:
                track_instances.disappear_time[i] += 1
                # print(track_instances.obj_idxes[i])
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1
                    # print("track {} has score {:.2f}, disappear".format(i, track_instances.scores[i], self.max_obj_id))
            # elif (track_instances.obj_idxes[i] >= 0) and (track_instances.scores[i] >= self.filter_score_thresh) and (track_instances.keep_cls[i] == False):
            #     print("track {} keeps origin obj_id {}, cls changes to {}".format(i, track_instances.obj_idxes[i], track_instances.cls_idxes[i]))
                # track_instances.obj_idxes[i] = self.max_obj_id
                # self.max_obj_id += 1
        return track_instances

    @staticmethod
    def quantity_filter(track_instances, maximum_quantity):
        scores = track_instances.scores
        _, top_indices = torch.topk(scores, k=maximum_quantity, sorted=False)
        top_indices = torch.sort(top_indices).values
        return top_indices


class CausalMotionHead(nn.Module):
    """One-step causal residual head used by repair_v2.

    The output is a logit-box velocity.  The last layer is zero initialized so
    adding the head cannot change the original reference trajectory at step 0.
    """

    def __init__(self, hidden_dim, text_dim, mode="one_step_conditioned_v2", velocity_limit=1.25,
                 detach_features=True):
        super().__init__()
        aliases = {
            "class_conditioned": "one_step_conditioned_v2",
            "category_conditioned": "one_step_conditioned_v2",
            "class_agnostic": "one_step_agnostic_v2",
            "category_agnostic": "one_step_agnostic_v2",
        }
        mode = aliases.get(mode, mode)
        if mode not in ("one_step_conditioned_v2", "one_step_agnostic_v2"):
            raise ValueError("motion mode must be one_step_agnostic_v2 or one_step_conditioned_v2")
        self.mode = mode
        self.velocity_limit = float(velocity_limit)
        self.detach_features = bool(detach_features)
        input_dim = hidden_dim + 4 + text_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hs, boxes, logits, select_id, text_embeddings):
        if self.detach_features:
            hs, boxes, logits = hs.detach(), boxes.detach(), logits.detach()
        evidence = logits.sigmoid()
        evidence = evidence / evidence.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        if self.mode == "one_step_agnostic_v2":
            semantic = torch.zeros((*evidence.shape[:2], text_embeddings.shape[0]), device=hs.device, dtype=hs.dtype)
        else:
            active_text = text_embeddings[:, select_id].transpose(0, 1).to(hs.device, hs.dtype)
            semantic = torch.matmul(evidence.to(active_text), active_text)
        features = torch.cat([hs, boxes.detach() if self.detach_features else boxes, semantic], dim=-1)
        return self.velocity_limit * torch.tanh(self.net(features))


class OVFrameMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses,
                        random_drop=0,
                        calculate_negative_samples=True,
                        num_queries=900,
                        train_with_artificial_img_seqs=False,
                        label_mode='complete',
                        motion_mode='none',
                        class_registry=None,
                        protocol_role='cil',
                        pl_iou_min=0.5,
                        lambda_gt=1.0,
                        lambda_pl=0.25,
                        pl_warmup_steps=100,
                        ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses,)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.random_drop = random_drop

        self.num_queries = num_queries
        self.calculate_negative_samples = calculate_negative_samples
        self.train_with_artificial_img_seqs = train_with_artificial_img_seqs
        self.label_mode = label_mode
        self.motion_mode = motion_mode
        self.frame_metadata = []
        self.class_registry = class_registry
        self.protocol_role = str(protocol_role)
        self.pl_matcher = QualityAwarePLMatcher(
            cost_class=getattr(matcher, "cost_class", 1.0),
            cost_bbox=getattr(matcher, "cost_bbox", 1.0),
            cost_giou=getattr(matcher, "cost_giou", 1.0),
            iou_min=float(pl_iou_min),
        )
        self.pl_iou_min = float(pl_iou_min)
        self.lambda_gt = float(lambda_gt)
        self.lambda_pl = float(lambda_pl)
        self.pl_warmup_steps = int(pl_warmup_steps)
        self.training_step = 0
        self.loss_accumulator = LossAccumulator()
        self._normalizers = {}
        self._loss_layer_tag = 0
        self.last_supervision_diagnostics = {}

    def set_training_step(self, step: int) -> None:
        self.training_step = int(step)

    def effective_pl_lambda(self) -> float:
        if self.pl_warmup_steps <= 0:
            return self.lambda_pl
        return self.lambda_pl * min(1.0, float(self.training_step) / float(self.pl_warmup_steps))

    def initialize(self, gt_instances: List[Instances], frame_metadata=None):
        self.gt_instances = gt_instances
        self.frame_metadata = frame_metadata or []
        self.num_samples = 0
        self.motion_pairs = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}
        self.loss_accumulator.clear()
        self._normalizers = {}
        self._loss_layer_tag = 0
        self.last_supervision_diagnostics = {}

    def _step(self):
        self._current_frame_idx += 1

    def get_num_boxes(self, num_samples):
        device = self.sample_device or next(self.parameters(), torch.zeros(1)).device
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            "align": self.loss_align,
            "align_pre": self.loss_align_pre,
            "motion": self.loss_motion,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def _frame_metadata(self, index=None):
        index = self._current_frame_idx if index is None else int(index)
        return self.frame_metadata[index] if index < len(self.frame_metadata) else {}

    def build_classification_targets(self, logits, select_id, gt_instances, indices, frame_metadata, label_mode):
        """Build partial-label targets once for main and every auxiliary layer."""
        batch, queries, columns = logits.shape
        target = torch.zeros_like(logits)
        valid = torch.zeros_like(logits, dtype=torch.bool)
        weights = torch.ones_like(logits)
        select_id = select_id.to(logits.device)
        for batch_index in range(batch):
            meta = frame_metadata[batch_index] if batch_index < len(frame_metadata) else {}
            exhaustive = meta.get("exhaustive_global_ids", meta.get("supervised_global_ids", []))
            if label_mode == "complete" and not exhaustive:
                exhaustive = meta.get("active_global_ids", [])
            for global_id in exhaustive:
                columns_for_id = (select_id == int(global_id)).nonzero(as_tuple=False).flatten()
                if len(columns_for_id):
                    valid[batch_index, :, columns_for_id] = True
            src_idx, tgt_idx = indices[batch_index]
            for source_index, target_index in zip(src_idx.tolist(), tgt_idx.tolist()):
                if int(target_index) < 0 or int(target_index) >= len(gt_instances[batch_index]):
                    continue
                global_id = int(gt_instances[batch_index].labels[int(target_index)])
                columns_for_id = (select_id == global_id).nonzero(as_tuple=False).flatten()
                if not len(columns_for_id):
                    raise ValueError("matched global ID %d is absent from select_id" % global_id)
                column = int(columns_for_id[0])
                target[batch_index, int(source_index), column] = 1.0
                valid[batch_index, int(source_index), column] = True
                if gt_instances[batch_index].has("label_weights"):
                    value = float(gt_instances[batch_index].label_weights[int(target_index)].detach().item())
                    weights[batch_index, int(source_index), column] = max(0.0, min(1.0, value))
        return target, valid, weights

    def _apply_ignore_mask(self, valid, target, boxes, frame_metadata):
        if not frame_metadata:
            return valid
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach()).clamp(0, 1)
        for batch_index, meta in enumerate(frame_metadata):
            image_size = meta.get("image_size", [1, 1])
            ih, iw = float(image_size[0]), float(image_size[1])
            for region in meta.get("ignore_regions", []):
                values = region.get("bbox_xyxy", [])
                if len(values) != 4 or iw <= 0 or ih <= 0:
                    continue
                region_box = torch.tensor([values[0] / iw, values[1] / ih, values[2] / iw, values[3] / ih], device=boxes.device)
                ix0 = torch.maximum(boxes_xyxy[batch_index, :, 0], region_box[0])
                iy0 = torch.maximum(boxes_xyxy[batch_index, :, 1], region_box[1])
                ix1 = torch.minimum(boxes_xyxy[batch_index, :, 2], region_box[2])
                iy1 = torch.minimum(boxes_xyxy[batch_index, :, 3], region_box[3])
                inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
                area = (boxes_xyxy[batch_index, :, 2] - boxes_xyxy[batch_index, :, 0]).clamp(min=0) * (boxes_xyxy[batch_index, :, 3] - boxes_xyxy[batch_index, :, 1]).clamp(min=0)
                rarea = max(0.0, float(values[2] - values[0])) / iw * max(0.0, float(values[3] - values[1])) / ih
                iou = inter / (area + rarea - inter).clamp(min=1e-6)
                valid[batch_index, iou >= 0.5] &= target[batch_index, iou >= 0.5] > 0
        return valid

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        logits = outputs["pred_logits"]
        target, valid, positive_weights = self.build_classification_targets(
            logits, outputs["select_id"], gt_instances, indices,
            [self._frame_metadata()] if logits.shape[0] == 1 else self.frame_metadata,
            self.label_mode,
        )
        valid = self._apply_ignore_mask(valid, target, outputs["pred_boxes"],
                                        [self._frame_metadata()] if logits.shape[0] == 1 else self.frame_metadata)
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = 0.25 * target + 0.75 * (1.0 - target)
        elementwise = alpha_t * (1.0 - p_t).pow(2) * ce
        # Return numerator.  OVFrameMatcher.forward applies the common
        # detection normalizer once; there is intentionally no Q factor.
        return {"loss_ce": (elementwise * valid.to(elementwise) * positive_weights).sum()}

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        if src_boxes.numel() == 0:
            return {
                'loss_bbox': outputs['pred_boxes'].sum() * 0,
                'loss_giou': outputs['pred_boxes'].sum() * 0,
            }
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        mask = (target_obj_ids != -1)
        if not bool(mask.any()):
            return {
                'loss_bbox': outputs['pred_boxes'].sum() * 0,
                'loss_giou': outputs['pred_boxes'].sum() * 0,
            }

        target_weights = torch.ones((len(target_boxes),), device=target_boxes.device)
        for offset, (gt_per_img, (_, target_index)) in enumerate(zip(gt_instances, indices)):
            if gt_per_img.has("label_weights"):
                target_weights[offset:offset + len(target_index)] = gt_per_img.label_weights[target_index].to(target_weights)
        target_weights = target_weights[mask]
        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none') * target_weights[:, None]
        loss_giou = (1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask]))) * target_weights)

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum()
        losses['loss_giou'] = loss_giou.sum()

        return losses

    def loss_motion(self, outputs, targets, indices, num_boxes):
        """Supervise causal inverse-sigmoid deltas using the next GT frame."""
        motion = outputs.get('motion_velocity')
        if motion is None:
            return {'loss_motion': outputs['pred_boxes'].sum() * 0}
        next_index = self._current_frame_idx + 1
        if next_index >= len(self.gt_instances):
            return {'loss_motion': motion.sum() * 0}
        current_gt = self.gt_instances[self._current_frame_idx]
        next_gt = self.gt_instances[next_index]
        current_meta = self._frame_metadata(self._current_frame_idx)
        next_meta = self._frame_metadata(next_index)
        current_time = current_meta.get("timestamp_s")
        next_time = next_meta.get("timestamp_s")
        max_dt = float(current_meta.get("max_motion_dt", 2.0))
        if current_time is None or next_time is None:
            return {'loss_motion': motion.sum() * 0}
        dt = float(next_time) - float(current_time)
        if not math.isfinite(dt) or dt <= 0 or dt > max_dt:
            return {'loss_motion': motion.sum() * 0}
        next_by_id = {
            int(value): index for index, value in enumerate(next_gt.obj_ids.tolist())
            if int(value) >= 0 and (not next_gt.has("label_sources") or int(next_gt.label_sources[index]) in (0, 1))
        }
        predicted = []
        expected = []
        matched_sources, matched_targets = indices[0]
        for source_idx, target_idx in zip(matched_sources.tolist(), matched_targets.tolist()):
            source_idx = int(source_idx)
            target_idx = int(target_idx)
            if target_idx < 0 or target_idx >= len(current_gt):
                continue
            if current_gt.has("label_sources") and int(current_gt.label_sources[target_idx]) not in (0, 1):
                continue
            object_id = int(current_gt.obj_ids[target_idx])
            if object_id not in next_by_id:
                continue
            current_box = current_gt.boxes[target_idx].to(motion)
            next_box = next_gt.boxes[next_by_id[object_id]].to(motion)
            expected.append((inverse_sigmoid(next_box) - inverse_sigmoid(current_box)) / dt)
            predicted.append(motion[0, source_idx])
        if not predicted:
            return {'loss_motion': motion.sum() * 0}
        self.motion_pairs += len(predicted)
        return {'loss_motion': F.smooth_l1_loss(torch.stack(predicted), torch.stack(expected), reduction='sum')}

    def loss_align(self, outputs, targets, indices, num_boxes, l1_distillation=False):
        """Alignment mechanism guides generalization capabilities and aligned queries.
        """
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx

        idx = self._get_src_permutation_idx(indices)
        src_feature = outputs["pred_embed"][idx]
        if src_feature.numel() == 0 or outputs.get("image_feat") is None:
            return {"loss_align": outputs["pred_logits"].sum() * 0}

        select_id = outputs["select_id"]
        image_feat = outputs["image_feat"]
        target_feature = []
        target_weights = []
        for t, (_, i) in zip(targets, indices):
            for position, c in zip(i.tolist(), t.labels[i]):
                index = (select_id == c).nonzero(as_tuple=False)[0]
                target_feature.append(image_feat[int(index.item())])
                target_weights.append(float(t.label_weights[position].item()) if t.has("label_weights") else 1.0)
        if not target_feature:
            return {"loss_align": outputs["pred_logits"].sum() * 0}
        target_feature = torch.stack(target_feature, dim=0)
        # l1 normalize the feature
        src_feature = nn.functional.normalize(src_feature, dim=1)
        if l1_distillation:
            loss_feature = F.l1_loss(src_feature, target_feature, reduction="none")
        else:
            loss_feature = F.mse_loss(src_feature, target_feature, reduction="none")
        losses = {"loss_align": (loss_feature * torch.as_tensor(target_weights, device=loss_feature.device)[:, None]).sum()}
        return losses

    def loss_align_pre(self, outputs, targets, indices, num_boxes):
        """Preserve text features without sudden variations.
        """
        input_feat = outputs["input_feat"]
        loss_feature_all = []

        select_id = outputs["select_id"]
        uniq_labels = [torch.unique(t.labels) for t in targets]
        tgt_ids_all = []
        embed_bs_index = []
        for i, uniq_label in enumerate(uniq_labels):
            tgt_ids = []
            if len(uniq_label)==0:
                continue
            else:
                embed_bs_index.append(i)
                for lid in uniq_label:
                    index = (select_id == lid).nonzero(as_tuple=False)[0]
                    tgt_ids.append(index)
                tgt_ids = torch.cat(tgt_ids)
            tgt_ids_all.append(tgt_ids)

        input_feats = torch.cat([input_feat[i] for i in tgt_ids_all])
        encoder_embeds = outputs["text_embed"][:, embed_bs_index]

        for encoder_embed in encoder_embeds:
            src_feature = torch.cat([enc_embed[tgt_id] for enc_embed, tgt_id in zip(encoder_embed,tgt_ids_all)])
            # l2 normalize the feature
            src_feature = nn.functional.normalize(src_feature, dim=1)
            loss_feature = F.mse_loss(src_feature, input_feats, reduction="none")
            loss_feature_all.append(loss_feature.sum() / num_boxes)
        loss_feature_all = torch.stack(loss_feature_all)
        loss_encoder_align = loss_feature_all.sum()
        losses = {"loss_align_pre": loss_encoder_align}
        return losses

    def match_for_single_frame(self, outputs: dict, is_first=None):
        outputs_without_aux = {k: v for k, v in outputs.items() if
                               k != 'aux_outputs' and k != 'enc_outputs'}

        def select_unmatched_indexes(matched_indexes: torch.Tensor, num_total_indexes: int) -> torch.Tensor:
            matched_indexes_set = set(matched_indexes.detach().cpu().numpy().tolist())
            all_indexes_set = set(list(range(num_total_indexes)))
            unmatched_indexes_set = all_indexes_set - matched_indexes_set
            unmatched_indexes = torch.as_tensor(list(unmatched_indexes_set), dtype=torch.long).to(matched_indexes)
            return unmatched_indexes

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances_last: Instances = outputs_without_aux['track_instances']

        if self.train_with_artificial_img_seqs:
            shielded_ids = protect_det_preds(outputs_without_aux, num_queries=self.num_queries)
            keep_indices = torch.ones(len(track_instances_last), dtype=torch.bool, device=shielded_ids.device)
            keep_indices[shielded_ids] = False
            track_instances = track_instances_last[keep_indices]
        else:
            keep_indices = torch.ones(
                len(track_instances_last), dtype=torch.bool,
                device=track_instances_last.obj_idxes.device)
            track_instances = track_instances_last

        outputs_i = {
            'pred_logits': track_instances.pred_logits.unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes.unsqueeze(0),
            'pred_embed': outputs_without_aux['pred_embed'][0, keep_indices].unsqueeze(0),
            'select_id':outputs_without_aux['select_id'],
            'image_feat':outputs_without_aux['image_feat'],
            'motion_velocity': outputs_without_aux.get('motion_velocity', None)[0, keep_indices].unsqueeze(0) if outputs_without_aux.get('motion_velocity', None) is not None else None,
        }

        obj_idxes = gt_instances_i.obj_ids
        device = obj_idxes.device
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}

        # step1. inherit and update the previous tracks.
        frame_meta = self._frame_metadata()
        exhaustive_ids = {int(v) for v in frame_meta.get("exhaustive_global_ids", frame_meta.get("supervised_global_ids", []))}
        num_disappear_track = 0
        track_instances.matched_gt_idxes[:] = -1
        valid_track_mask = track_instances.obj_idxes >= 0
        valid_track_idxes = torch.arange(len(track_instances), device=device)[valid_track_mask]
        valid_obj_idxes = track_instances.obj_idxes[valid_track_idxes]
        for j in range(len(valid_obj_idxes)):
            obj_id = valid_obj_idxes[j].item()
            if obj_id in obj_idx_to_gt_idx:
                track_instances.matched_gt_idxes[valid_track_idxes[j]] = obj_idx_to_gt_idx[obj_id]
            elif exhaustive_ids:
                num_disappear_track += 1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=device)
        matched_track_idxes = (track_instances.obj_idxes >= 0) # occu
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(device)

        # step2. select the unmatched slots.
        # note that the fp tracks (obj_idxes == -2) will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select unmatched targets.  Real GT is matched before PL so a
        # low-confidence pseudo target cannot take a query from new-class GT.
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]
        unmatched_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        if gt_instances_i.has("label_sources"):
            gt_target_indexes = unmatched_tgt_indexes[gt_instances_i.label_sources[unmatched_tgt_indexes] != 2]
            pl_target_indexes = unmatched_tgt_indexes[gt_instances_i.label_sources[unmatched_tgt_indexes] == 2]
        else:
            gt_target_indexes, pl_target_indexes = unmatched_tgt_indexes, torch.empty(0, dtype=torch.long, device=device)

        def match_for_single_decoder_layer(unmatched_outputs, unmatched_track_idxes):
            matches = []
            available_queries = unmatched_track_idxes
            for target_indexes, is_pl in ((gt_target_indexes, False), (pl_target_indexes, True)):
                if len(available_queries) == 0 or len(target_indexes) == 0:
                    continue
                subset_targets = gt_instances_i[target_indexes]
                subset_outputs = {
                    "pred_logits": unmatched_outputs["pred_logits"][:, available_queries],
                    "pred_boxes": unmatched_outputs["pred_boxes"][:, available_queries],
                    "select_id": unmatched_outputs["select_id"],
                }
                local_indices = (self.pl_matcher(subset_outputs, [subset_targets])[0]
                                 if is_pl else self.matcher(subset_outputs, [subset_targets])[0])
                if is_pl:
                    self.last_supervision_diagnostics["pl_match_candidates"] = float(
                        self.last_supervision_diagnostics.get("pl_match_candidates", 0.0) + self.pl_matcher.last_diagnostics.get("candidates", 0))
                    self.last_supervision_diagnostics["pl_match_accepted"] = float(
                        self.last_supervision_diagnostics.get("pl_match_accepted", 0.0) + self.pl_matcher.last_diagnostics.get("accepted", 0))
                if len(local_indices[0]) == 0:
                    continue
                source = available_queries[local_indices[0]]
                target = target_indexes[local_indices[1]]
                matches.append(torch.stack([source, target], dim=1).to(device))
                keep = torch.ones(len(available_queries), dtype=torch.bool, device=device)
                keep[local_indices[0]] = False
                available_queries = available_queries[keep]
            return torch.cat(matches, dim=0) if matches else torch.empty((0, 2), dtype=torch.long, device=device)

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits.unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes.unsqueeze(0),
            'select_id':outputs_without_aux['select_id'],
        }

        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, unmatched_track_idxes)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += sum(
            1 for index in range(len(gt_instances_i))
            if (not gt_instances_i.has("label_sources") or int(gt_instances_i.label_sources[index].item()) in (0, 1, 2))
        )
        self.sample_device = device

        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # The normalizer namespace is (frame, decoder layer).  Keep
                # auxiliary supervision from overwriting the main-layer
                # source counts.
                self._loss_layer_tag = int(i) + 1

                # Shield and match individually for each layer.
                if self.train_with_artificial_img_seqs:
                    _shielded_ids_layer = protect_det_preds(aux_outputs, num_queries=self.num_queries)
                    _keep_indices_layer = torch.ones(
                        len(track_instances_last), dtype=torch.bool,
                        device=_shielded_ids_layer.device)
                    _keep_indices_layer[_shielded_ids_layer] = False
                    track_instances_layer = track_instances_last[_keep_indices_layer]
                else:
                    _keep_indices_layer = torch.ones(
                        len(track_instances_last), dtype=torch.bool,
                        device=track_instances_last.obj_idxes.device)
                    track_instances_layer = track_instances_last

                # step1*. compute inherited matches in a private tensor.  An
                # auxiliary layer must not mutate the main track state.
                layer_matched_gt = torch.full((len(track_instances_layer),), -1, dtype=torch.long, device=device)
                valid_track_mask = track_instances_layer.obj_idxes >= 0
                valid_track_idxes = torch.arange(len(track_instances_layer), device=device)[valid_track_mask]
                valid_obj_idxes = track_instances_layer.obj_idxes[valid_track_idxes]
                for j in range(len(valid_obj_idxes)):
                    obj_id = valid_obj_idxes[j].item()
                    if obj_id in obj_idx_to_gt_idx:
                        layer_matched_gt[valid_track_idxes[j]] = obj_idx_to_gt_idx[obj_id]

                full_track_idxes = torch.arange(len(track_instances_layer), dtype=torch.long, device=device)
                matched_track_idxes_layer = (track_instances_layer.obj_idxes >= 0)
                prev_matched_indices_layer = torch.stack(
                    [full_track_idxes[matched_track_idxes_layer], layer_matched_gt[matched_track_idxes_layer]], dim=1).to(device)

                # step2*. select the unmatched slots.
                unmatched_track_idxes_layer = full_track_idxes[track_instances_layer.obj_idxes == -1]

                # step3*. do matching between the unmatched slots and GTs.
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, _keep_indices_layer].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, _keep_indices_layer].unsqueeze(0),
                    'select_id': aux_outputs['select_id'],
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, unmatched_track_idxes_layer)

                # step4*. merge the unmatched pairs and the matched pairs.
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices_layer], dim=0)

                # step5*. calculate losses.
                _keep_aux_outputs = {
                    'pred_logits': aux_outputs['pred_logits'][0, _keep_indices_layer].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, _keep_indices_layer].unsqueeze(0),
                    'pred_embed': aux_outputs['pred_embed'][0, _keep_indices_layer].unsqueeze(0),
                    'select_id': aux_outputs['select_id'],
                    'image_feat': aux_outputs['image_feat'],
                }
                for loss in self.losses:
                    if loss == "motion":
                        continue
                    l_dict = self.get_loss(loss,
                                           _keep_aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})

        self._loss_layer_tag = 0
        self._step()
        return track_instances

    def forward(self, outputs):
        losses = outputs.pop("losses_dict")
        normalizer_det = self.get_num_boxes(max(1, self.num_samples))
        normalizer_motion = max(1, int(self.motion_pairs))
        loss_avg = {}
        for loss_name, _ in losses.items():
            loss_avg[loss_name] = losses[loss_name] / (normalizer_motion if "loss_motion" in loss_name else normalizer_det)
        return loss_avg


    @staticmethod
    def _pair_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = box_ops.box_cxcywh_to_xyxy(left)
        right = box_ops.box_cxcywh_to_xyxy(right)
        lt = torch.maximum(left[..., :2], right[..., :2])
        rb = torch.minimum(left[..., 2:], right[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area_l = (left[..., 2] - left[..., 0]).clamp(min=0) * (left[..., 3] - left[..., 1]).clamp(min=0)
        area_r = (right[..., 2] - right[..., 0]).clamp(min=0) * (right[..., 3] - right[..., 1]).clamp(min=0)
        return inter / (area_l + area_r - inter).clamp(min=1e-8)

    def build_classification_targets(self, logits, select_id, gt_instances, indices, frame_metadata, label_mode):
        """Build V3 masks once; GT mutex negatives never come from PL."""
        batch, queries, columns = logits.shape
        target = torch.zeros_like(logits)
        gt_mask = torch.zeros_like(logits, dtype=torch.bool)
        gt_weights = torch.ones_like(logits)
        pl_mask = torch.zeros_like(logits, dtype=torch.bool)
        pl_weights = torch.zeros_like(logits)
        matched_iou = torch.zeros((batch, queries), dtype=logits.dtype, device=logits.device)
        trusted = torch.zeros((batch, queries), dtype=torch.bool, device=logits.device)
        matched_source = torch.full((batch, queries), -1, dtype=torch.long, device=logits.device)
        select_id = select_id.to(logits.device) if torch.is_tensor(select_id) else torch.as_tensor(select_id, device=logits.device)
        gt_counts = []
        pl_counts = {}
        for batch_index in range(batch):
            meta = frame_metadata[batch_index] if batch_index < len(frame_metadata) else {}
            if not bool(meta.get("annotation_valid", True)):
                gt_counts.append(0)
                continue
            exhaustive = meta.get("exhaustive_global_ids", meta.get("supervised_global_ids", []))
            if label_mode == "complete" and not exhaustive:
                exhaustive = meta.get("active_global_ids", [])
            for global_id in exhaustive:
                match = (select_id == int(global_id)).nonzero(as_tuple=False).flatten()
                if match.numel():
                    gt_mask[batch_index, :, match] = True
            gt_count = 0
            for index in range(len(gt_instances[batch_index])):
                source = int(gt_instances[batch_index].label_sources[index].item()) if gt_instances[batch_index].has("label_sources") else 0
                if source in (0, 1):
                    gt_count += 1
            src_idx, tgt_idx = indices[batch_index]
            for source_index, target_index in zip(src_idx.tolist(), tgt_idx.tolist()):
                source_index = int(source_index)
                target_index = int(target_index)
                if target_index < 0 or target_index >= len(gt_instances[batch_index]) or source_index >= queries:
                    continue
                source = int(gt_instances[batch_index].label_sources[target_index].item()) if gt_instances[batch_index].has("label_sources") else 0
                if source not in (0, 1, 2):
                    continue
                global_id = int(gt_instances[batch_index].labels[target_index].item())
                match = (select_id == global_id).nonzero(as_tuple=False).flatten()
                if not match.numel():
                    raise ValueError("matched global ID %d is absent from select_id" % global_id)
                column = int(match[0].item())
                query_box = outputs_box = None
                # The current loss call supplies the prediction boxes through
                # the owning outputs object; a detached zero IoU is replaced
                # by the real query/GT IoU in loss_labels below.
                if source in (0, 1):
                    target[batch_index, source_index, column] = 1.0
                    gt_mask[batch_index, source_index, column] = True
                    if gt_instances[batch_index].has("label_weights"):
                        value = float(gt_instances[batch_index].label_weights[target_index].detach().item())
                        gt_weights[batch_index, source_index, column] = max(0.0, min(1.0, value))
                    matched_source[batch_index, source_index] = source
                else:
                    # A PL row is a positive for its own class only.  It is
                    # never a universal negative for other active classes.
                    target[batch_index, source_index, column] = 1.0
                    pl_mask[batch_index, source_index, column] = True
                    value = float(gt_instances[batch_index].label_weights[target_index].detach().item()) if gt_instances[batch_index].has("label_weights") else 1.0
                    pl_weights[batch_index, source_index, column] = max(0.0, min(1.0, value))
                    matched_source[batch_index, source_index] = source
                    pl_counts[global_id] = pl_counts.get(global_id, 0) + 1
            gt_counts.append(gt_count)
        self._pending_gt_counts = gt_counts
        self._pending_pl_counts = pl_counts
        return ClassificationSupervision(target, gt_mask, gt_weights, pl_mask, pl_weights, matched_iou, trusted, matched_source)

    def _complete_matched_supervision(self, supervision, outputs, gt_instances, indices, frame_metadata):
        """Add detached IoU-gated reliable-GT mutex cells after matching."""
        select_id = outputs["select_id"].to(outputs["pred_logits"].device)
        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if batch_index >= len(frame_metadata) or not bool(frame_metadata[batch_index].get("annotation_valid", True)):
                continue
            instance = gt_instances[batch_index]
            for source_index, target_index in zip(src_idx.tolist(), tgt_idx.tolist()):
                source_index, target_index = int(source_index), int(target_index)
                if target_index < 0 or target_index >= len(instance) or source_index >= outputs["pred_boxes"].shape[1]:
                    continue
                source = int(instance.label_sources[target_index].item()) if instance.has("label_sources") else 0
                if source not in (0, 1):
                    continue
                global_id = int(instance.labels[target_index].item())
                iou = self._pair_iou(outputs["pred_boxes"][batch_index, source_index], instance.boxes[target_index].to(outputs["pred_boxes"]))
                supervision.matched_iou[batch_index, source_index] = iou.detach()
                if float(iou.detach().item()) < self.pl_iou_min:
                    continue
                supervision.trusted_gt_queries[batch_index, source_index] = True
                if self.class_registry is None:
                    mutex_ids = [int(value) for value in select_id.tolist() if int(value) != global_id]
                else:
                    mutex_ids = self.class_registry.mutually_exclusive_ids(global_id, [int(value) for value in select_id.tolist()])
                for mutex_id in mutex_ids:
                    mutex_columns = (select_id == int(mutex_id)).nonzero(as_tuple=False).flatten()
                    if mutex_columns.numel():
                        supervision.gt_mask[batch_index, source_index, mutex_columns] = True
                        supervision.gt_weights[batch_index, source_index, mutex_columns] = 1.0
        return supervision

    def _apply_ignore_mask(self, valid, target, boxes, frame_metadata):
        if not frame_metadata:
            return valid
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach()).clamp(0, 1)
        result = valid.clone()
        for batch_index, meta in enumerate(frame_metadata):
            image_size = meta.get("image_size", [1, 1])
            ih, iw = float(image_size[0]), float(image_size[1])
            for region in meta.get("ignore_regions", []):
                values = region.get("bbox_xyxy", [])
                if len(values) != 4 or iw <= 0 or ih <= 0:
                    continue
                region_box = torch.tensor([values[0] / iw, values[1] / ih, values[2] / iw, values[3] / ih], device=boxes.device)
                lt = torch.maximum(boxes_xyxy[batch_index, :, :2], region_box[:2])
                rb = torch.minimum(boxes_xyxy[batch_index, :, 2:], region_box[2:])
                inter = (rb - lt).clamp(min=0)
                inter = inter[:, 0] * inter[:, 1]
                area = (boxes_xyxy[batch_index, :, 2] - boxes_xyxy[batch_index, :, 0]).clamp(min=0) * (boxes_xyxy[batch_index, :, 3] - boxes_xyxy[batch_index, :, 1]).clamp(min=0)
                region_area = max(0.0, float(values[2] - values[0])) / iw * max(0.0, float(values[3] - values[1])) / ih
                iou = inter / (area + region_area - inter).clamp(min=1e-6)
                overlap = iou >= 0.5
                # Ignore screens only negative GT cells.  Reliable GT/PL
                # positives remain supervised when they fall in the region.
                result[batch_index, overlap] &= target[batch_index, overlap] > 0
        return result

    def loss_labels(self, outputs, gt_instances, indices, num_boxes, log=False):
        logits = outputs["pred_logits"]
        metadata = [self._frame_metadata()] if logits.shape[0] == 1 else self.frame_metadata
        supervision = self.build_classification_targets(logits, outputs["select_id"], gt_instances, indices, metadata, self.label_mode)
        supervision = self._complete_matched_supervision(supervision, outputs, gt_instances, indices, metadata)
        supervision.gt_mask = self._apply_ignore_mask(supervision.gt_mask, supervision.targets, outputs["pred_boxes"], metadata)
        gt_loss = focal_binary_loss(logits, supervision.targets, supervision.gt_mask, supervision.gt_weights)
        class_losses = []
        class_counts = self._pending_pl_counts
        for global_id, count in sorted(class_counts.items()):
            column = (outputs["select_id"] == int(global_id)).nonzero(as_tuple=False).flatten()
            if not column.numel() or count <= 0:
                continue
            class_losses.append(focal_binary_loss(
                logits[:, :, column], supervision.targets[:, :, column], supervision.pl_mask[:, :, column], supervision.pl_weights[:, :, column]
            ) / float(count))
        pl_loss = torch.stack(class_losses).mean() if class_losses else logits.sum() * 0.0
        frame = int(self._current_frame_idx)
        self._normalizers[(frame, int(self._loss_layer_tag))] = {
            "gt": float(sum(self._pending_gt_counts)),
            "pl": 1.0 if class_losses else 0.0,
        }
        self.last_supervision_diagnostics = {
            "gt_valid_objects": float(sum(self._pending_gt_counts)),
            "pl_valid_classes": float(len(class_losses)),
            "pl_valid_objects": float(sum(class_counts.values())),
            "gt_trusted_queries": float(supervision.trusted_gt_queries.sum().item()),
        }
        return {
            "loss_gt_ce": gt_loss * self.lambda_gt,
            "loss_pl_ce": pl_loss * self.effective_pl_lambda(),
        }

    def loss_boxes(self, outputs, gt_instances, indices, num_boxes):
        gt_bbox_values, gt_giou_values = [], []
        pl_by_class = {}
        pl_giou_by_class = {}
        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            instance = gt_instances[batch_index]
            for source_index, target_index in zip(src_idx.tolist(), tgt_idx.tolist()):
                source_index, target_index = int(source_index), int(target_index)
                if target_index < 0 or target_index >= len(instance) or source_index >= outputs["pred_boxes"].shape[1]:
                    continue
                if instance.has("obj_ids") and int(instance.obj_ids[target_index].item()) < 0:
                    continue
                src_box = outputs["pred_boxes"][batch_index, source_index]
                tgt_box = instance.boxes[target_index].to(src_box)
                weight = float(instance.label_weights[target_index].item()) if instance.has("label_weights") else 1.0
                source = int(instance.label_sources[target_index].item()) if instance.has("label_sources") else 0
                bbox_value = F.l1_loss(src_box, tgt_box, reduction="sum") * weight
                giou_value = 1.0 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_box[None]), box_ops.box_cxcywh_to_xyxy(tgt_box[None])))[0]
                giou_value = giou_value * weight
                if source in (0, 1):
                    gt_bbox_values.append(bbox_value)
                    gt_giou_values.append(giou_value)
                elif source == 2:
                    gid = int(instance.labels[target_index].item())
                    pl_by_class.setdefault(gid, []).append(bbox_value)
                    pl_giou_by_class.setdefault(gid, []).append(giou_value)
        reference = outputs["pred_boxes"].sum() * 0.0
        gt_bbox = torch.stack(gt_bbox_values).sum() if gt_bbox_values else reference
        gt_giou = torch.stack(gt_giou_values).sum() if gt_giou_values else reference
        pl_bbox = torch.stack([torch.stack(values).sum() / float(len(values)) for values in pl_by_class.values()]).mean() if pl_by_class else reference
        pl_giou = torch.stack([torch.stack(values).sum() / float(len(values)) for values in pl_giou_by_class.values()]).mean() if pl_giou_by_class else reference
        self._normalizers.setdefault((int(self._current_frame_idx), int(self._loss_layer_tag)), {
            "gt": float(len(gt_bbox_values)), "pl": 1.0 if pl_by_class else 0.0,
        })
        return {
            "loss_gt_bbox": gt_bbox * self.lambda_gt,
            "loss_gt_giou": gt_giou * self.lambda_gt,
            "loss_pl_bbox": pl_bbox * self.effective_pl_lambda(),
            "loss_pl_giou": pl_giou * self.effective_pl_lambda(),
        }

    def forward(self, outputs):
        losses = outputs.pop("losses_dict")
        loss_avg = {}
        for loss_name, value in losses.items():
            parts = loss_name.split("_")
            frame = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            layer = int(parts[2][3:]) + 1 if len(parts) > 2 and parts[2].startswith("aux") else 0
            normalizers = self._normalizers.get((frame, layer), {"gt": 1.0, "pl": 1.0})
            if "loss_gt_" in loss_name or loss_name.endswith("loss_align"):
                denominator = normalizers.get("gt", 0.0)
            elif "loss_pl_" in loss_name:
                denominator = normalizers.get("pl", 0.0)
            elif "loss_motion" in loss_name:
                denominator = float(max(1, self.motion_pairs))
            else:
                denominator = 1.0
            loss_avg[loss_name] = value / float(denominator) if denominator > 0 else value * 0.0
        return loss_avg


class OVTR(nn.Module):
    def __init__(self, backbone, transformer, num_feature_levels, criterion, track_embed,
                    aux_loss=True, with_box_refine=False, two_stage=False,
                    two_stage_bbox_embed_share=False,
                    dec_pred_bbox_embed_share=True,
                    use_checkpoint=None,
                    distribution_based_sampling=None,
                    text_embeddings=None,
                    image_embeddings=None,
                    max_len=None,
                    novel_cls_cpu=None,
                    computed_aux=None,
                    score_thresh=None,
                    filter_score_thresh=None,
                    miss_tolerance=None,
                    train_with_artificial_img_seqs=False,
                    class_registry=None,
                    semantic_bank=None,
                    continual_cfg=None,
                    motion_velocity_limit=1.25,
                    motion_warmup_steps=20,
                    motion_detach_features=True,
                    motion_detach_reference=True,
                    motion_max_dt=2.0,
                    inference_dedup_enabled=True,
                    duplicate_iou=0.9,
                    birth_threshold=None,
                    keep_threshold=None,
                    export_threshold=None,
                    duplicate_feature_cos=0.95,
                    dedup_new_new=True,
                    dedup_new_track=True,
                    merge_existing_ids=False,
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        self.num_queries = transformer.num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.max_pad_len = max_len
        if text_embeddings is None:
            raise ValueError("text_embeddings are required for OVTR")
        self.text_embeddings = text_embeddings.t().contiguous()
        self.image_embeddings = None if image_embeddings is None else image_embeddings.t().contiguous()
        self.class_registry = class_registry
        self.semantic_bank = semantic_bank
        self.continual_cfg = continual_cfg or {}
        self.motion_mode = self.continual_cfg.get("motion_mode", "none")
        self.motion_velocity_limit = float(motion_velocity_limit)
        self.motion_warmup_steps = int(motion_warmup_steps)
        self.motion_detach_features = bool(motion_detach_features)
        self.motion_detach_reference = bool(motion_detach_reference)
        self.motion_max_dt = float(motion_max_dt)
        self.motion_training_step = 0
        self.inference_dedup_enabled = bool(inference_dedup_enabled)
        self.duplicate_iou = float(duplicate_iou)
        self.duplicate_feature_cos = float(duplicate_feature_cos)
        self.dedup_new_new = bool(dedup_new_new)
        self.dedup_new_track = bool(dedup_new_track)
        self.merge_existing_ids = bool(merge_existing_ids)
        self.runtime_stats = {
            "input_detection_queries": 0,
            "active_track_count": 0,
            "suppressed_duplicate_count": 0,
            "new_id_count": 0,
            "output_box_count": 0,
            "track_capacity_hits": 0,
            "invalid_dt_count": 0,
            "motion_advance_count": 0,
            "birth_count": 0,
            "keep_count": 0,
            "export_count": 0,
        }
        self.patch2query = nn.Linear(512, 256)
        self.all_ids = torch.tensor(range(self.text_embeddings.shape[-1]))
        self.all_ids = [i + 1 for i in self.all_ids]
        self.select_id = list(range(0, len(Frequency_list_total_1)))

        self.frequency = torch.tensor(Frequency_list_70, dtype=torch.float32, device='cpu')
        print("0.7 power sampling | Training excludes rare categories.")
        self.frequency_eval = torch.tensor(Frequency_list_total_1, dtype=torch.float32, device='cpu')
        self.novel_cls_cpu = novel_cls_cpu
        self.computed_aux = computed_aux

        for layer in [self.patch2query]:
            nn.init.xavier_uniform_(self.patch2query.weight)
            nn.init.constant_(self.patch2query.bias, 0)

        # feature alignment
        self.feature_align = nn.Linear(256, 512) # alignment head
        nn.init.xavier_uniform_(self.feature_align.weight)
        nn.init.constant_(self.feature_align.bias, 0)
        num_pred = len(self.computed_aux)
        if with_box_refine:
            self.feature_align = _get_clones(self.feature_align, num_pred)
        else:
            self.feature_align = nn.ModuleList([self.feature_align for _ in range(num_pred)])

        if self.motion_mode != "none":
            self.motion_head = CausalMotionHead(
                hidden_dim, self.text_embeddings.shape[0], mode=self.motion_mode,
                velocity_limit=self.motion_velocity_limit,
                detach_features=self.motion_detach_features,
            )

        # bbox
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed

        if two_stage:
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.refpoint_embed = None

        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.post_process = TrackerPostProcess()
        def _scalar(value, default):
            if value is None:
                return default
            if isinstance(value, (list, tuple)):
                return value[0] if value else default
            return value

        self.track_base = RuntimeTrackerBase(
            score_thresh=_scalar(score_thresh, 0.6),
            filter_score_thresh=_scalar(filter_score_thresh, 0.6),
            miss_tolerance=_scalar(miss_tolerance, 5),
            maximum_quantity=self.continual_cfg.get("maximum_quantity", 50),
            birth_threshold=birth_threshold,
            keep_threshold=keep_threshold,
            export_threshold=export_threshold,
        )
        self.ious_thresh = float(self.continual_cfg.get("ious_thresh", 0.3))

        self.use_checkpoint = use_checkpoint
        self.distribution_based_sampling = distribution_based_sampling
        self.criterion = criterion
        self.train_with_artificial_img_seqs = train_with_artificial_img_seqs

    def _generate_empty_tracks(self, cls_pad_len=None):
        track_instances = Instances((1, 1))
        num_queries = self.num_queries
        dim_h = self.transformer.d_model
        device = self.transformer.level_embed.device
        if cls_pad_len is None:
            cls_pad_len = self.text_embeddings.shape[-1]

        track_instances.ref_pts = torch.zeros((num_queries, 4), device=device)
        track_instances.query_tgt = self.transformer.tgt_embed.weight
        track_instances.query_pos = torch.zeros((num_queries, dim_h), device=device)

        track_instances.obj_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((num_queries, 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((num_queries, cls_pad_len), dtype=torch.float, device=device)
        track_instances.suppressed_this_frame = torch.zeros((num_queries,), dtype=torch.bool, device=device)
        track_instances.observed_this_frame = torch.zeros((num_queries,), dtype=torch.bool, device=device)
        track_instances.export_valid = torch.zeros((num_queries,), dtype=torch.bool, device=device)
        track_instances.export_scores = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.last_timestamp_s = torch.full((num_queries,), float("nan"), dtype=torch.float32, device=device)
        track_instances.motion_valid = torch.zeros((num_queries,), dtype=torch.bool, device=device)
        if self.motion_mode != "none":
            track_instances.motion_velocity = torch.zeros((num_queries, 4), device=device)

        if not self.training:
            track_instances.cls_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
            track_instances.disappear_time = torch.zeros((num_queries, ), dtype=torch.long, device=device)
        return track_instances.to(device)

    def set_training_step(self, step: int):
        self.motion_training_step = int(step)
        if hasattr(self.criterion, "set_training_step"):
            self.criterion.set_training_step(int(step))

    def _advance_motion_references(self, track_instances, current_timestamp_s):
        """Apply a detached velocity once before the current transformer call."""
        if self.motion_mode == "none" or not track_instances.has("motion_velocity"):
            return track_instances
        timestamp = current_timestamp_s
        base = inverse_sigmoid(track_instances.pred_boxes[:, :4].detach().clamp(1e-5, 1.0 - 1e-5))
        valid = track_instances.motion_valid & (track_instances.obj_idxes >= 0)
        if timestamp is None:
            track_instances.ref_pts = base
            self.runtime_stats["invalid_dt_count"] += int(valid.sum().item())
            return track_instances
        current = torch.full_like(track_instances.last_timestamp_s, float(timestamp))
        dt = current - track_instances.last_timestamp_s
        usable = valid & torch.isfinite(dt) & (dt > 0) & (dt <= self.motion_max_dt)
        if not bool(usable.any()):
            track_instances.ref_pts = base
            self.runtime_stats["invalid_dt_count"] += int(valid.sum().item())
            return track_instances
        gate = 0.0 if self.training and self.motion_training_step <= self.motion_warmup_steps else 1.0
        ref = base.clone()
        velocity = track_instances.motion_velocity.detach() if self.motion_detach_reference else track_instances.motion_velocity
        ref[usable] = base[usable] + gate * velocity[usable] * dt[usable, None]
        track_instances.ref_pts = ref
        self.runtime_stats["motion_advance_count"] += int(usable.sum().item())
        return track_instances

    def consume_runtime_stats(self):
        value = dict(self.runtime_stats)
        self.runtime_stats = {key: 0 for key in self.runtime_stats}
        return value

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_embed):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_embed':c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_embed[:-1])]

    def _distribution_based_sampling(self, pad_len, uniq_labels=None):
        frequency = self.frequency.clone()
        frequency[uniq_labels] = 0
        extra_labels = torch.multinomial(frequency, pad_len)
        extra_labels = extra_labels[torch.isin(extra_labels,self.novel_cls_cpu, invert=True)]
        extra_labels = extra_labels[torch.randperm(len(extra_labels))]
        return extra_labels

    def get_select_id(self, cls_num, labels_list, extra_labels, is_first):
        if self.class_registry is not None:
            select_id = list(self.class_registry.active_select_ids())
            if not select_id:
                raise ValueError("C-MOT stage has no active semantic classes")
            return select_id, None
        max_pad_len = max(cls_num, self.max_pad_len)
        # get input categories
        uniq_labels = torch.unique(labels_list).to("cpu")
        if is_first: # first frame detection
            if len(uniq_labels) < max_pad_len:
                pad_len = max_pad_len - len(uniq_labels)
                if self.distribution_based_sampling: # Sample negative categories based on the distribution.
                    extra_labels = self._distribution_based_sampling(pad_len, uniq_labels=uniq_labels)
                else:
                    extra_list = torch.tensor([i for i in self.all_ids if i not in uniq_labels])
                    extra_labels = extra_list[torch.randperm(len(extra_list))][:pad_len]
                select_id = uniq_labels.tolist() + extra_labels.tolist()
            else:
                select_id = uniq_labels.tolist()
                extra_labels = torch.LongTensor([])
        else: # subsequent frame tracking
            extra_label_notin = torch.isin(extra_labels, uniq_labels, invert=True)
            extra_labels_cur = extra_labels[extra_label_notin]
            select_id = uniq_labels.tolist() + extra_labels_cur.tolist()
            if len(select_id) < max_pad_len:
                sampled_labels = self._distribution_based_sampling(max_pad_len-len(select_id), uniq_labels=torch.tensor(select_id))
                if extra_labels is not None:
                    extra_labels = torch.cat([extra_labels_cur, sampled_labels])
                else:
                    extra_labels = sampled_labels
                select_id = uniq_labels.tolist() + extra_labels.tolist()
            elif len(select_id) > max_pad_len:
                select_id = select_id[:max_pad_len]
        return select_id, extra_labels

    def _forward_single_image(self, samples, track_instances: Instances, targets=None, extra_labels=None ,is_first=True, cls_num=0, *, frame_context=None):
        if frame_context is not None and not is_first:
            self._advance_motion_references(track_instances, frame_context.get("timestamp_s"))
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # Get the selected category id
        if self.training:
            labels_list = torch.cat([targets.labels]) if targets is not None and len(targets) > 0 else torch.empty(0, dtype=torch.long, device=masks[0].device)
            select_id, extra_labels = self.get_select_id(cls_num, labels_list, extra_labels, is_first)
        else:
            if self.class_registry is not None:
                select_id = list(self.class_registry.active_select_ids())
            else:
                select_id, extra_labels = self.select_id, None

        # Prepare queries and embeddings for alignment
        text_query = self.text_embeddings[:, select_id].to(masks[0].device).t()
        if self.image_embeddings is None:
            image_feat_ori = None
        else:
            image_align = self.image_embeddings[:, select_id].to(masks[0].device).t()
            image_feat_ori = image_align.float().detach()

        dtype = self.patch2query.weight.dtype
        text_query = self.patch2query(text_query.type(dtype))
        select_id = torch.tensor(select_id).to(text_query.device)
        text_dict = preprocess_for_masks(srcs[0].shape[0], select_id, text_query)

        (hs_cti, hs_ofa, init_reference, inter_references, pre_outputs_classes, query_pos_track) = self.transformer(
            srcs, masks, pos, track_instances.query_pos, track_instances.query_tgt, ref_pts=track_instances.ref_pts, text_dict=text_dict)

        outputs_coords = []
        outputs_embeds = []

        for lvl in range(hs_cti.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs_ofa[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_embeds.append(self.feature_align[lvl](hs_ofa[lvl]))
        outputs_class = pre_outputs_classes
        outputs_coord = torch.stack(outputs_coords)
        outputs_embed = torch.stack(outputs_embeds)
        motion_velocity = None
        if self.motion_mode != "none":
            motion_velocity = self.motion_head(
                hs_ofa[-1], outputs_coord[-1], outputs_class[-1], select_id, self.text_embeddings)

        if init_reference.shape[-1]==4:
            ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :4]], dim=0)
        else:
            ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'ref_pts': ref_pts_all[-2],
            "pred_embed": outputs_embed[-1],
            "select_id": select_id,
            "image_feat": image_feat_ori,
            "extra_labels": extra_labels,
            "motion_velocity": motion_velocity,
            }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_embed)
            for temp in out["aux_outputs"]:
                temp["select_id"] = select_id
                temp["image_feat"] = image_feat_ori

        out['query_pos_track'] = query_pos_track.transpose(0, 1)
        out['hs_ofa'] = hs_ofa[-1]
        out['hs_cti'] = hs_cti[-1]
        return out

    def _post_process_single_image(self, frame_res, track_instances, is_last, is_repeat=None, is_first=False, target_size=None, *, frame_context=None):
        with torch.no_grad():
            track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding_txt = frame_res['hs_cti'][0]
        track_instances.output_embedding_img = frame_res['hs_ofa'][0]
        track_instances.query_pos = frame_res["query_pos_track"][0]
        if frame_res.get('motion_velocity') is not None:
            track_instances.motion_velocity = frame_res['motion_velocity'][0]
        track_instances.suppressed_this_frame = torch.zeros_like(track_instances.obj_idxes, dtype=torch.bool)

        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res, is_first)
        else:
            _track_discard = torch.zeros(
                0, dtype=torch.long, device=track_instances.scores.device)
            # Resolve global semantic IDs before duplicate protection; the
            # protection rule is same-class only and must not use the
            # initial -1 placeholders.
            track_instances = self.post_process_pre(track_instances, frame_res['select_id'], is_first)
            self.runtime_stats["input_detection_queries"] += int(min(self.num_queries, len(track_instances)))
            if self.inference_dedup_enabled:
                track_instances, dedup_stats = protect_track_preds(
                    track_instances, num_queries=self.num_queries,
                    birth_threshold=self.track_base.birth_threshold,
                    keep_threshold=self.track_base.keep_threshold,
                    duplicate_iou=self.duplicate_iou,
                    duplicate_feature_cos=self.duplicate_feature_cos,
                    dedup_new_new=self.dedup_new_new,
                    dedup_new_track=self.dedup_new_track,
                    merge_existing_ids=self.merge_existing_ids,
                )
                self.runtime_stats["suppressed_duplicate_count"] += int(dedup_stats.get("suppressed", 0))
            # each track will be assigned an unique global id by the track base.
            if is_first:
                self.track_base.clear()
            old_max_obj_id = int(self.track_base.max_obj_id)
            track_instances = self.track_base.update(track_instances, _track_discard, is_repeat=is_repeat)
            self.runtime_stats["new_id_count"] += int(self.track_base.max_obj_id - old_max_obj_id)
            self.runtime_stats["active_track_count"] += int((track_instances.obj_idxes >= 0).sum().item())
            self.runtime_stats["birth_count"] += int(self.track_base.max_obj_id - old_max_obj_id)
            if track_instances.has("observed_this_frame"):
                self.runtime_stats["keep_count"] += int(track_instances.observed_this_frame.sum().item())
            if track_instances.has("export_valid"):
                self.runtime_stats["export_count"] += int(track_instances.export_valid.sum().item())
            if len(track_instances) >= self.track_base.maximum_quantity:
                self.runtime_stats["track_capacity_hits"] += 1

        if frame_context is not None and frame_context.get("timestamp_s") is not None:
            timestamp = float(frame_context["timestamp_s"])
            track_instances.last_timestamp_s = torch.full_like(track_instances.last_timestamp_s, timestamp)
            if track_instances.has("motion_valid"):
                track_instances.motion_valid = track_instances.obj_idxes >= 0
        if track_instances.has("export_valid"):
            self.runtime_stats["output_box_count"] += int(track_instances.export_valid.sum().item())
        elif track_instances.has("suppressed_this_frame"):
            self.runtime_stats["output_box_count"] += int(((track_instances.obj_idxes >= 0) & (~track_instances.suppressed_this_frame)).sum().item())

        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks(cls_pad_len=track_instances.pred_logits.shape[1])
        tmp['track_instances'] = track_instances

        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        frame_res['track_instances_pre'] = track_instances
        return frame_res

    def post_process_pre(self, track_instances, select_id, is_first):
        out_logits = track_instances.pred_logits

        prob = out_logits.sigmoid()
        max_scores, labels = prob.max(-1)
        cur_cls_idxes = select_id[labels]
        # track_instances.keep_cls = torch.eq(cur_cls_idxes, track_instances.cls_idxes)

        if is_first:
            track_instances.cls_idxes = cur_cls_idxes
        else:
            accepted = max_scores >= self.track_base.keep_threshold
            track_instances.cls_idxes[accepted] = cur_cls_idxes[accepted]
        # The runtime score is always the score of the actually assigned
        # global semantic class, never an unrelated winning select column.
        assigned_columns = []
        for global_id in track_instances.cls_idxes.tolist():
            matches = (select_id == int(global_id)).nonzero(as_tuple=False).flatten()
            assigned_columns.append(int(matches[0].item()) if matches.numel() else 0)
        assigned_columns = torch.as_tensor(assigned_columns, dtype=torch.long, device=prob.device)
        track_instances.assigned_column = assigned_columns
        track_instances.scores = prob[torch.arange(len(prob), device=prob.device), assigned_columns]
        track_instances.export_scores = track_instances.scores.clone()
        return track_instances

    @torch.no_grad()
    def inference_single_image(self, data, track_instances=None, is_repeat=False, frame_id=None, ori_img_size=None, extra_labels=None):
        img = nested_tensor_from_tensor_list([data['imgs'][0]])
        if (track_instances is None) or (frame_id == 0):
            track_instances = self._generate_empty_tracks()
        if frame_id == 0:
            is_first = True
        else:
            is_first = False

        res = self._forward_single_image(img, track_instances, None, extra_labels, is_first, cls_num=None, frame_context={"timestamp_s": None})
        res = self._post_process_single_image(res, track_instances, False, is_repeat=is_repeat, is_first=is_first, target_size=ori_img_size[:-1], frame_context={"timestamp_s": None})

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size[:-1])
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size[:-1]
            # scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    @torch.no_grad()
    def inference_video_frame(self, data, runtime_state=None, frame_context=None):
        """Run one frame while keeping raw state separate from export output.

        ``inference_single_image`` is retained for upstream callers.  C-MOT
        uses this method so ``TrackerPostProcess`` cannot remove
        ``pred_logits``/``pred_boxes`` from the state fed to the next frame.
        """
        frame_context = frame_context or {}
        frame_id = int(frame_context.get("frame_id", 0))
        is_first = runtime_state is None or frame_id == 0
        if is_first:
            runtime_state = self._generate_empty_tracks()
        model_device = next(self.parameters()).device
        frame_tensor = data["imgs"][0].to(model_device)
        img = nested_tensor_from_tensor_list([frame_tensor])
        target_size = frame_context.get("target_size")
        if target_size is None:
            h, w = frame_tensor.shape[-2:]
            target_size = (h, w)
        res = self._forward_single_image(
            img, runtime_state, None, None, is_first, cls_num=None, frame_context=frame_context)
        res = self._post_process_single_image(
            res, runtime_state, is_last=False, is_first=is_first,
            target_size=target_size, frame_context=frame_context)
        next_state = res["track_instances"]
        export_state = copy.deepcopy(next_state)
        exported = self.post_process(export_state, target_size)
        prediction_record = {
            "frame_key": frame_context.get("frame_key"),
            "timestamp_s": frame_context.get("timestamp_s"),
            "predictions": [],
        }
        for i in range(len(exported)):
            if int(exported.obj_idxes[i]) < 0:
                continue
            if exported.has("export_valid") and not bool(exported.export_valid[i].item()):
                continue
            if exported.has("suppressed_this_frame") and bool(exported.suppressed_this_frame[i].item()):
                continue
            global_id = int(exported.cls_idxes[i]) if exported.has("cls_idxes") else int(exported.labels[i])
            prediction_record["predictions"].append({
                "track_id": int(exported.obj_idxes[i]),
                "global_id": global_id,
                "column_id": int(exported.labels[i]),
                "score": float(exported.scores[i]),
                "bbox_xyxy": [float(v) for v in exported.boxes[i]],
            })
        return next_state, prediction_record, res

    def forward(self, data):
        if self.training:
            self.criterion.initialize(data['gt_instances'], data.get('frame_metadata'))
        frames = data['imgs']
        cls_num = max([len(torch.unique(gt_instance.labels)) for gt_instance in data['gt_instances']] or [0])
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
            'select_id': [],
            'track_instances': []
        }
        track_instances = self._generate_empty_tracks()

        keys = list(track_instances._fields.keys())
        for frame_index, (frame, targets) in enumerate(zip(frames, data['gt_instances'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            is_first = frame_index == 0
            if is_first:
                extra_labels = None
            else:
                extra_labels = frame_res["extra_labels"]
            if self.use_checkpoint and frame_index < len(frames) - 3:
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp, targets, extra_labels, is_first, cls_num, frame_context=None)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['ref_pts'],
                        frame_res['pred_embed'],
                        frame_res['select_id'],
                        frame_res['image_feat'],
                        frame_res['extra_labels'],
                        frame_res['query_pos_track'],
                        frame_res['hs_cti'],
                        frame_res['hs_ofa'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_embed'] for aux in frame_res['aux_outputs']],
                        *[aux['select_id'] for aux in frame_res['aux_outputs']],
                        *[aux['image_feat'] for aux in frame_res['aux_outputs']],
                    )
                args = [frame] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'ref_pts': tmp[2],
                    'pred_embed': tmp[3],
                    'select_id': tmp[4],
                    'image_feat': tmp[5],
                    'extra_labels': tmp[6],
                    'query_pos_track': tmp[7],
                    'hs_cti': tmp[8],
                    'hs_ofa': tmp[9],
                    'aux_outputs': [{
                        'pred_logits': tmp[10+i],
                        'pred_boxes': tmp[10+5+i],
                        'pred_embed': tmp[10+10+i],
                        'select_id': tmp[10+15+i],
                        'image_feat': tmp[10+20+i],
                    } for i in range(len(self.computed_aux)-1)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_context = data.get("frame_metadata", [{}] * len(frames))[frame_index]
                frame_context = dict(frame_context)
                frame_context["max_motion_dt"] = self.motion_max_dt
                frame_res = self._forward_single_image(frame, track_instances, targets, extra_labels, is_first, cls_num, frame_context=frame_context)
            if self.use_checkpoint and frame_index < len(frames) - 3:
                frame_context = None
            else:
                frame_context = data.get("frame_metadata", [{}] * len(frames))[frame_index]
                frame_context = dict(frame_context)
                frame_context["max_motion_dt"] = self.motion_max_dt
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last, is_first=is_first, frame_context=frame_context)

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])
            outputs['select_id'].append(frame_res['select_id'])
            outputs['track_instances'].append(frame_res['track_instances_pre'])

        outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args, cfg):
    text_path = getattr(cfg, 'Clip_text_embeddings', None)
    image_path = getattr(cfg, 'Clip_image_embeddings', None)
    if not text_path:
        raise ValueError("Clip_text_embeddings is required")
    if getattr(cfg, 'cmot_disable_alignment', False):
        image_path = None
    text_embeddings, image_embeddings = load_embeddings(text_path, image_path)

    device = torch.device(args.device)
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)
    d_model = transformer.d_model
    hidden_dim = cfg.dim_feedforward
    updater = build_updater(args, args.track_query_iteration, d_model, hidden_dim, d_model*2)
    matcher = build_matcher(args)

    num_frames_per_batch = max(getattr(args, 'sampler_lengths', None) or [2])
    motion_mode = getattr(cfg, 'cmot_motion_mode', 'none')
    alignment_enabled = image_embeddings is not None
    motion_loss_coef = float(getattr(cfg, 'cmot_motion_loss_coef', 1.0))
    continual_cfg = dict(getattr(cfg, 'cmot_continual_cfg', None) or {})
    continual_cfg.setdefault('motion_mode', motion_mode)
    weight_dict = {}

    for i in range(0, num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_gt_ce".format(i): args.cls_loss_coef,
                            "frame_{}_loss_pl_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_gt_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_pl_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_gt_giou'.format(i): args.giou_loss_coef,
                            'frame_{}_loss_pl_giou'.format(i): args.giou_loss_coef,
                            })
        if alignment_enabled:
            weight_dict['frame_{}_loss_align'.format(i)] = args.align_loss_coef
        if motion_mode != 'none':
            weight_dict['frame_{}_loss_motion'.format(i)] = motion_loss_coef

    if args.aux_loss:
        for i in range(0, num_frames_per_batch):
            for j in range(cfg.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_gt_ce".format(i, j): args.cls_loss_coef,
                                    "frame_{}_aux{}_loss_pl_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_gt_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_pl_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_gt_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_aux{}_loss_pl_giou'.format(i, j): args.giou_loss_coef,
                                    })
                if alignment_enabled:
                    weight_dict['frame_{}_aux{}_loss_align'.format(i, j)] = args.align_loss_coef

    losses = ['labels', 'boxes']
    if alignment_enabled:
        losses.append('align')
    if motion_mode != 'none':
        losses.append('motion')

    criterion = OVFrameMatcher(None, matcher=matcher, weight_dict=weight_dict, losses=losses, random_drop=args.random_drop,
                                train_with_artificial_img_seqs=cfg.train_with_artificial_img_seqs,
                                calculate_negative_samples=args.calculate_negative_samples,
                                num_queries=cfg.num_queries,
                                label_mode=getattr(cfg, 'cmot_label_mode', 'complete'),
                                motion_mode=motion_mode,
                                class_registry=getattr(cfg, 'cmot_class_registry', None),
                                protocol_role=getattr(cfg, 'cmot_protocol_role', 'cil'),
                                pl_iou_min=float(getattr(cfg, 'cmot_pl_iou_min', 0.5)),
                                lambda_gt=float(getattr(cfg, 'cmot_lambda_gt', 1.0)),
                                lambda_pl=float(getattr(cfg, 'cmot_lambda_pl', 0.25)),
                                pl_warmup_steps=int(getattr(cfg, 'cmot_pl_warmup_steps', 100)),
                                )
    criterion.to(device)

    model = OVTR(
        backbone,
        transformer,
        track_embed=updater,
        num_feature_levels=cfg.num_feature_levels,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        max_len=args.max_len,
        use_checkpoint=cfg.use_checkpoint_track,
        train_with_artificial_img_seqs=cfg.train_with_artificial_img_seqs,
        distribution_based_sampling=cfg.distribution_based_sampling,
        novel_cls_cpu=novel_class,
        computed_aux=cfg.computed_aux,
        score_thresh=args.score_thresh,
        filter_score_thresh=args.filter_score_thresh,
        miss_tolerance=args.miss_tolerance,
        class_registry=getattr(cfg, 'cmot_class_registry', None),
        continual_cfg=continual_cfg,
        motion_velocity_limit=float(getattr(cfg, 'cmot_motion_velocity_limit', continual_cfg.get('velocity_limit', 1.25))),
        motion_warmup_steps=int(getattr(cfg, 'cmot_motion_warmup_steps', continual_cfg.get('warmup_steps', 20))),
        motion_detach_features=bool(getattr(cfg, 'cmot_motion_detach_features', continual_cfg.get('detach_features', True))),
        motion_detach_reference=bool(getattr(cfg, 'cmot_motion_detach_reference', continual_cfg.get('detach_reference', True))),
        motion_max_dt=float(getattr(cfg, 'cmot_motion_max_dt', continual_cfg.get('max_dt', 2.0))),
        inference_dedup_enabled=bool(getattr(cfg, 'cmot_inference_dedup_enabled', True)),
        duplicate_iou=float(getattr(cfg, 'cmot_duplicate_iou', continual_cfg.get('duplicate_iou', 0.9))),
        birth_threshold=float(getattr(cfg, 'cmot_birth_threshold', continual_cfg.get('birth_threshold', 0.5))),
        keep_threshold=float(getattr(cfg, 'cmot_keep_threshold', continual_cfg.get('keep_threshold', 0.2))),
        export_threshold=float(getattr(cfg, 'cmot_export_threshold', continual_cfg.get('export_threshold', 0.5))),
        duplicate_feature_cos=float(getattr(cfg, 'cmot_duplicate_feature_cos', 0.95)),
        dedup_new_new=bool(getattr(cfg, 'cmot_dedup_new_new', True)),
        dedup_new_track=bool(getattr(cfg, 'cmot_dedup_new_track', True)),
        merge_existing_ids=bool(getattr(cfg, 'cmot_merge_existing_ids', False)),
    )
    return model, criterion
