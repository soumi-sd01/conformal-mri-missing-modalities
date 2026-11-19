import torch
from scripts.evaluate_missing_modalities import predict_volume_sliding_window, get_patterns


def test_all_15_nonempty_patterns():
    assert len(get_patterns()) == 15


def test_sliding_window_stitches_identity_model():
    image = torch.randn(4, 12, 14, 16)
    output_bs1 = predict_volume_sliding_window(
        torch.nn.Identity(), image, torch.device("cpu"),
        patch_size=(8, 8, 8), overlap=0.5, amp=False, batch_size=1
    )
    output_bs16 = predict_volume_sliding_window(
        torch.nn.Identity(), image, torch.device("cpu"),
        patch_size=(8, 8, 8), overlap=0.5, amp=False, batch_size=16
    )
    assert output_bs1.shape == image.shape
    assert torch.allclose(output_bs1, image, atol=1e-6)
    assert torch.allclose(output_bs16, output_bs1, atol=1e-6)
