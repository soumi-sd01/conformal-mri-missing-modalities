import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """GPU-conscious 3D U-Net; checkpoint architecture must remain unchanged."""

    def __init__(self, in_channels=4, out_channels=4, base_filters=16):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(f * 4, f * 8)
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, 2, stride=2)
        self.dec3 = DoubleConv3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, 2, stride=2)
        self.dec2 = DoubleConv3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, 2, stride=2)
        self.dec1 = DoubleConv3D(f * 2, f)
        self.final_conv = nn.Conv3d(f, out_channels, 1)

    @staticmethod
    def _pad_and_cat(up, skip):
        dd = skip.size(2) - up.size(2)
        dh = skip.size(3) - up.size(3)
        dw = skip.size(4) - up.size(4)
        if dd or dh or dw:
            up = F.pad(
                up,
                [
                    dw // 2, dw - dw // 2,
                    dh // 2, dh - dh // 2,
                    dd // 2, dd - dd // 2,
                ],
            )
        return torch.cat([up, skip], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(self._pad_and_cat(self.up3(b), e3))
        d2 = self.dec2(self._pad_and_cat(self.up2(d3), e2))
        d1 = self.dec1(self._pad_and_cat(self.up1(d2), e1))
        return self.final_conv(d1)
