import torch

from cmot.losses.motion import MotionNLLLoss, target_velocity_from_boxes
from cmot.models.motion_prior import CategoryConditionedMotionPrior, MotionForecast


def _history():
    boxes = torch.tensor(
        [[[0.10, 0.20, 0.20, 0.10], [0.12, 0.20, 0.20, 0.10], [0.14, 0.20, 0.20, 0.10], [0.16, 0.20, 0.20, 0.10]],
         [[0.50, 0.50, 0.20, 0.20], [0.50, 0.50, 0.20, 0.20], [0.50, 0.50, 0.20, 0.20], [0.50, 0.50, 0.20, 0.20]]],
        dtype=torch.float32,
    )
    times = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    valid = torch.ones((2, 4), dtype=torch.bool)
    quality = torch.ones((2, 4))
    return boxes, times, valid, quality


def test_prior_shapes_and_invalid_history_are_causal():
    torch.manual_seed(4)
    prior = CategoryConditionedMotionPrior(history_length=4, hidden_dim=16, num_modes=3, semantic_dim=8)
    boxes, times, valid, quality = _history()
    valid[:, -1] = False
    semantic = torch.randn(2, 8)
    first = prior(boxes, times, valid, quality, semantic, torch.ones(2))
    changed = boxes.clone()
    changed[:, -1] = torch.tensor([[0.9, 0.9, 0.8, 0.8], [0.1, 0.1, 0.8, 0.8]])
    invalid = valid.clone()
    invalid[:, -1] = False
    second = prior(changed, times, invalid, quality, semantic, torch.ones(2))
    assert first.velocity_mean.shape == (2, 3, 4)
    assert first.velocity_log_std.shape == (2, 3, 4)
    assert first.mixture_logits.shape == (2, 3)
    assert first.next_boxes.shape == (2, 4)
    assert torch.allclose(first.next_boxes, second.next_boxes, atol=1e-6)
    assert torch.equal(first.valid, torch.ones(2, dtype=torch.bool))


def test_class_agnostic_control_ignores_semantic_context_with_same_capacity():
    torch.manual_seed(5)
    prior = CategoryConditionedMotionPrior(history_length=4, hidden_dim=16, num_modes=3, semantic_dim=8)
    agnostic = CategoryConditionedMotionPrior(
        history_length=4, hidden_dim=16, num_modes=3, semantic_dim=8, mode="class_agnostic"
    )
    agnostic.load_state_dict(prior.state_dict())
    inputs = _history()
    out_a = agnostic(*inputs, torch.randn(2, 8), torch.ones(2))
    out_b = agnostic(*inputs, torch.randn(2, 8), torch.ones(2))
    assert torch.allclose(out_a.next_boxes, out_b.next_boxes)
    assert torch.allclose(out_a.velocity_mean, out_b.velocity_mean)


def test_motion_nll_has_one_valid_pair_normalization_and_gradient():
    means = torch.zeros((2, 3, 4), requires_grad=True)
    log_std = torch.zeros((2, 3, 4), requires_grad=True)
    mixture = torch.zeros((2, 3), requires_grad=True)
    forecast = MotionForecast(means, log_std, mixture, torch.zeros((2, 4)), torch.ones(2, dtype=torch.bool))
    targets = torch.zeros((2, 4))
    valid = torch.tensor([True, False])
    weights = torch.tensor([1.0, 1.0])
    loss_module = MotionNLLLoss()
    loss = loss_module(forecast, targets, valid, weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert loss_module.last_count == 1
    assert means.grad is not None and torch.isfinite(means.grad).all()
    zero = loss_module(forecast, targets, torch.zeros(2, dtype=torch.bool), weights)
    assert float(zero.detach()) == 0.0


def test_target_velocity_uses_log_size_state_and_delta_t():
    current = torch.tensor([[0.1, 0.2, 0.2, 0.1]])
    following = torch.tensor([[0.3, 0.2, 0.2, 0.1]])
    velocity = target_velocity_from_boxes(current, following, torch.tensor([2.0]))
    assert torch.allclose(velocity[0, :2], torch.tensor([0.1, 0.0]))
    assert torch.allclose(velocity[0, 2:], torch.zeros(2))
