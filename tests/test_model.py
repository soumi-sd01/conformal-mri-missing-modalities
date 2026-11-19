import torch
from src.models.unet3d import UNet3D


def test_unet_preserves_patch_shape():
    model = UNet3D(in_channels=4, out_channels=4, base_filters=4)
    model.eval()
    x = torch.randn(1, 4, 16, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 16, 32, 32)
    assert torch.isfinite(y).all()
