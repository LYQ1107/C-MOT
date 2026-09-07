# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Grounding DINO (https://github.com/IDEA-Research/GroundingDINO)
# Copyright (c) 2023 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from util.box_ops import box_cxcywh_to_xyxy
try:
    from mmdet.core import bbox_overlaps
except ImportError:
    # C-MOT's public contract tests only need the small IoU helper, and the
    # OVTR runtime already carries all other model dependencies.  Keep the
    # fallback local so a clean evaluator environment without MMDetection can
    # still import the adapted model; the validated OVTR environment uses the
    # upstream implementation above.
    def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
        if mode != "iou":
            raise ValueError("the local fallback only supports IoU")
        if is_aligned:
            lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
            rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
            wh = (rb - lt).clamp(min=0)
            overlap = wh[..., 0] * wh[..., 1]
            area1 = (bboxes1[..., 2] - bboxes1[..., 0]).clamp(min=0) * (bboxes1[..., 3] - bboxes1[..., 1]).clamp(min=0)
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]).clamp(min=0) * (bboxes2[..., 3] - bboxes2[..., 1]).clamp(min=0)
            return overlap / (area1 + area2 - overlap).clamp(min=eps)
        if bboxes1.dim() == 2 and bboxes2.dim() == 2:
            lhs = bboxes1[:, None, :]
            rhs = bboxes2[None, :, :]
            area1_shape = (slice(None), None)
            area2_shape = (None, slice(None))
        else:
            lhs = bboxes1.unsqueeze(-2)
            rhs = bboxes2.unsqueeze(-3)
            area1_shape = (..., slice(None), None)
            area2_shape = (..., None, slice(None))
        lt = torch.max(lhs[..., :2], rhs[..., :2])
        rb = torch.min(lhs[..., 2:], rhs[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        area1 = ((bboxes1[..., 2] - bboxes1[..., 0]).clamp(min=0) * (bboxes1[..., 3] - bboxes1[..., 1]).clamp(min=0))
        area2 = ((bboxes2[..., 2] - bboxes2[..., 0]).clamp(min=0) * (bboxes2[..., 3] - bboxes2[..., 1]).clamp(min=0))
        return overlap / (area1[area1_shape] + area2[area2_shape] - overlap).clamp(min=eps)


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res


def gen_encoder_output_proposals(
    memory: Tensor, memory_padding_mask: Tensor, spatial_shapes: Tensor, learnedwh=None
):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        # import ipdb; ipdb.set_trace()

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            # import ipdb; ipdb.set_trace()
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        # scale = torch.cat([W_[None].unsqueeze(-1), H_[None].unsqueeze(-1)], 1).view(1, 1, 1, 2).repeat(N_, 1, 1, 1)
        # grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        # wh = torch.ones_like(grid) / scale
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_
    # import ipdb; ipdb.set_trace()
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
        -1, keepdim=True
    )
    output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    # output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    # output_memory = output_memory.masked_fill(~output_proposals_valid, float('inf'))

    return output_memory, output_proposals


class RandomBoxPerturber:
    def __init__(
        self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2
    ) -> None:
        self.noise_scale = torch.Tensor(
            [x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale]
        )

    def __call__(self, refanchors: Tensor) -> Tensor:
        nq, bs, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = torch.rand_like(refanchors)
        noise_scale = self.noise_scale.to(device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.clamp_(0, 1)


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, no_reduction=False
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if no_reduction:
        return loss

    return loss.mean(1).sum() / num_boxes


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def attention_protection(outputs_class, num_queries, layer_id, isol_ratio=None):
    # Category Isolation Strategy.
    outputs_class = F.softmax(outputs_class, dim=-1)
    class_probs_row = outputs_class.unsqueeze(2)
    class_probs_col = outputs_class.unsqueeze(1)
    kl_div_matrix = torch.sum(F.kl_div(class_probs_col.log(), class_probs_row, reduction='none'), dim=-1)
    S_cls = kl_div_matrix + kl_div_matrix.transpose(-1, -2)
    isolate_threshold = S_cls.mean() * isol_ratio

    if layer_id != 5 and layer_id != -1:
        isolate_mask = S_cls > isolate_threshold
        if outputs_class.shape[1] != num_queries:
            isolate_mask[:, num_queries: , num_queries: ] = False
    else:
        isolate_mask = None
    return isolate_mask

def protect_track_preds(track_instances, num_queries=900, duplicate_iou=0.9):
    """Mark duplicate detections without mutating tracker age or IDs.

    The previous helper both removed query rows and incremented
    ``disappear_time``.  That made one duplicate suppression event look like a
    missing observation and also shifted the query/ID correspondence.  The
    tracker now owns ageing; this function only marks rows which must not be
    exported or assigned a fresh ID for the current frame.
    """
    count = len(track_instances)
    device = track_instances.scores.device
    suppressed = torch.zeros(count, dtype=torch.bool, device=device)
    if count == 0:
        track_instances.suppressed_this_frame = suppressed
        return track_instances, {"suppressed": 0, "track_duplicates": 0, "detection_shields": 0}

    boxes = box_cxcywh_to_xyxy(track_instances.pred_boxes.unsqueeze(0))[0]
    ious = bbox_overlaps(boxes.unsqueeze(0), boxes.unsqueeze(0), mode="iou")[0]
    scores = track_instances.scores
    obj_idxes = track_instances.obj_idxes
    classes = track_instances.cls_idxes if track_instances.has("cls_idxes") else torch.full_like(obj_idxes, -1)
    track_end = min(int(num_queries), count)

    # Keep the strongest already-tracked query for each highly-overlapping
    # same-class group.  Existing IDs are deliberately not aged here.
    tracked = [
        int(index) for index in range(track_end, count)
        if int(obj_idxes[index]) >= 0 and float(scores[index]) >= 0.19
    ]
    tracked.sort(key=lambda index: (-float(scores[index]), index))
    accepted = []
    for index in tracked:
        duplicate = any(
            int(classes[index]) == int(classes[other])
            and float(ious[index, other]) >= float(duplicate_iou)
            for other in accepted
        )
        if duplicate:
            suppressed[index] = True
        else:
            accepted.append(index)

    # A new detection which overlaps an accepted track must not shield the
    # track or create a duplicate ID.  Do not compare against suppressed
    # tracks, and do not suppress low-confidence detections before the normal
    # tracker threshold has a chance to handle them.
    for index in range(track_end):
        if float(scores[index]) < 0.19:
            continue
        if any(
            int(classes[index]) == int(classes[other])
            and float(ious[index, other]) >= float(duplicate_iou)
            for other in accepted
        ):
            suppressed[index] = True

    track_instances.suppressed_this_frame = suppressed
    return track_instances, {
        "suppressed": int(suppressed.sum().item()),
        "track_duplicates": int(suppressed[track_end:].sum().item()),
        "detection_shields": int(suppressed[:track_end].sum().item()),
    }

def protect_det_preds(outputs, num_queries=900):  
    '''Shield detection predictions close to tracking predictions to preserve the perception of 
    newly emerging targets.
    '''
    pred_boxes = outputs['pred_boxes']
    pred_boxes_xy = box_cxcywh_to_xyxy(pred_boxes)
    ious = bbox_overlaps(pred_boxes_xy, pred_boxes_xy, mode='iou')
    valid_index = ious > 0.8

    track_index = valid_index[0, :num_queries, num_queries:]
    true_positions = torch.nonzero(track_index, as_tuple=False)
    row_indices = true_positions[:, 0]
    shielded_ids = torch.unique(row_indices)

    return shielded_ids

def preprocess_for_masks(bs, select_id, text_query):
    text_attention_mask = torch.full([1, len(select_id)], True, device=text_query.device).repeat(bs, 1) # bs, 195
    text_features = text_query.unsqueeze(0).expand(bs, -1, -1)
    text_dict={}
    text_dict["text_features"] = text_features
    text_dict["text_token_mask"] = text_attention_mask
    text_dict["select_text_num"] = len(select_id)
    return text_dict
    

class ContrastiveEmbed(nn.Module):
    def __init__(self, max_text_len=1203):
        """
        Args:
            max_text_len: max length of text.
        """
        super().__init__()
        self.max_text_len = max_text_len

    def forward(self, x, text_dict):
        """_summary_

        Args:
            x (_type_): _description_
            text_dict (_type_): _description_
            {
                'encoded_text': encoded_text, # bs, 195, d_model
                'text_token_mask': text_token_mask, # bs, 195
                        # True for used tokens. False for padding tokens
            }
        Returns:
            _type_: _description_
        """
        assert isinstance(text_dict, dict)

        y = text_dict["encoded_text"]   
        text_token_mask = text_dict["text_token_mask"]

        res = x @ y.transpose(-1, -2)
        res.masked_fill_(~text_token_mask[:, None, :], float("-inf"))

        # padding to max_text_len
        new_res = torch.full((*res.shape[:-1], self.max_text_len), float("-inf"), device=res.device)
        new_res[..., : res.shape[-1]] = res

        return new_res
