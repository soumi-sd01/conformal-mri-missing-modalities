import torch
from src.training.losses import DiceCELoss
from src.training.metrics import per_class_dice, region_dice


def test_dice_ce_loss_finite_and_backward():
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    targets = torch.randint(0, 4, (2, 8, 8, 8))
    loss = DiceCELoss()(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_region_dice_hand_computation():
    targets = torch.tensor([[[[0, 1, 1, 2, 2, 3, 3, 0]]]])
    preds = torch.tensor([[[[0, 1, 2, 2, 2, 3, 0, 0]]]])
    scores = region_dice(preds, targets)
    assert abs(scores["WT"] - (10 / 11)) < 1e-5
    assert abs(scores["TC"] - (4 / 6)) < 1e-5
    assert abs(scores["ET"] - (2 / 3)) < 1e-5


def test_absent_region_both_empty_is_one():
    t = torch.tensor([[[[0, 1, 2, 0]]]])
    assert region_dice(t, t)["ET"] == 1.0


def test_absent_region_false_positive_is_zero():
    t = torch.tensor([[[[0, 1, 2, 0]]]])
    p = torch.tensor([[[[0, 1, 3, 0]]]])
    assert region_dice(p, t)["ET"] < 1e-4


def test_per_class_dice_keys():
    t = torch.tensor([[[[0, 1, 2, 3]]]])
    assert set(per_class_dice(t, t)) == {"class_0", "class_1", "class_2", "class_3"}
