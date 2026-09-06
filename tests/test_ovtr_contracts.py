import torch


def test_motion_head_uses_query_aligned_text_and_agnostic_mode():
    from models.ovtr import CausalMotionHead

    torch.manual_seed(3)
    conditioned = CausalMotionHead(hidden_dim=8, text_dim=5, mode="class_conditioned")
    agnostic = CausalMotionHead(hidden_dim=8, text_dim=5, mode="class_agnostic")
    agnostic.load_state_dict(conditioned.state_dict())
    hs = torch.randn(1, 4, 8)
    boxes = torch.rand(1, 4, 4)
    logits = torch.randn(1, 4, 2)
    select_id = torch.tensor([2, 7])
    text = torch.randn(5, 10)
    out = conditioned(hs, boxes, logits, select_id, text)
    assert out.shape == (1, 4, 4)
    out_a = agnostic(hs, boxes, logits, select_id, text)
    out_b = agnostic(hs, boxes, logits, select_id, text + 100.0)
    assert torch.allclose(out_a, out_b)


def test_partial_label_mask_blocks_hidden_old_class_gradient():
    from detectron2.structures import Instances
    from models.ovtr import OVFrameMatcher

    criterion = OVFrameMatcher(
        None, matcher=None, weight_dict={}, losses=["labels"], label_mode="partial"
    )
    criterion.frame_metadata = [{"label_scope": "partial", "supervised_global_ids": [792]}]
    criterion._current_frame_idx = 0
    target = Instances(
        (100, 100),
        labels=torch.tensor([792]),
        boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        obj_ids=torch.tensor([1]),
    )
    logits = torch.zeros((1, 3, 2), requires_grad=True)
    loss = criterion.loss_labels(
        {"pred_logits": logits, "select_id": torch.tensor([206, 792])},
        [target],
        [(torch.tensor([0]), torch.tensor([0]))],
        1.0,
    )["loss_ce"]
    loss.backward()
    assert torch.allclose(logits.grad[:, :, 0], torch.zeros_like(logits.grad[:, :, 0]))
    assert torch.any(logits.grad[:, :, 1].abs() > 0)
