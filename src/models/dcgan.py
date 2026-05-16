import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _make_norm(norm_type: str, channels: int, spatial_size: int, num_groups: int = 8) -> nn.Module:
    norm = norm_type.lower()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        groups = min(num_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(num_groups=groups, num_channels=channels)
    if norm == "layer":
        return nn.LayerNorm([channels, spatial_size, spatial_size])
    raise ValueError(f"Unknown norm_type={norm_type!r}. Expected one of: batch, group, layer")


class DCGANGenerator(nn.Module):
    def __init__(self, z_dim: int = 100, channels: int = 4, features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features * 8, 4, 1, 0),
            self._block(features * 8, features * 4, 4, 2, 1),
            self._block(features * 4, features * 2, 4, 2, 1),
            self._block(features * 2, features, 4, 2, 1),
            nn.ConvTranspose2d(features, channels, 4, 2, 1),
            nn.Tanh(),
        )

    def _block(self, in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, z):
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, channels: int = 4, features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, features, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 2, 1),
            spectral_norm(nn.Conv2d(features * 8, 1, 4, 1, 0)),
        )

    def _block(self, in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x).view(-1)


class DCGANImprovedGenerator(DCGANGenerator):
    def _block(self, in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )


class DCGANImprovedDiscriminator(nn.Module):
    def __init__(self, channels: int = 4, features: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, features, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 2, 1),
            spectral_norm(nn.Conv2d(features * 8, 1, 4, 1, 0)),
        )

    def _block(self, in_c, out_c, kernel, stride, padding):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)),
            nn.GroupNorm(num_groups=8, num_channels=out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x).view(-1)


class ResBlockDown(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)),
            nn.AvgPool2d(kernel_size=2),
        )
        self.skip = nn.Sequential(
            spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)),
            nn.AvgPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class ResNetDiscriminator64(nn.Module):
    def __init__(self, channels: int = 4, features: int = 64, pool: str = "avg"):
        super().__init__()
        if pool not in {"avg", "sum"}:
            raise ValueError(f"pool must be 'avg' or 'sum', got: {pool}")
        self.pool = pool
        self.stem = spectral_norm(nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.Sequential(
            ResBlockDown(features, features * 2),
            ResBlockDown(features * 2, features * 4),
            ResBlockDown(features * 4, features * 8),
            ResBlockDown(features * 8, features * 16),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.head = spectral_norm(nn.Linear(features * 16, 1))

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.activation(h)
        if self.pool == "avg":
            h = F.adaptive_avg_pool2d(h, output_size=1)
        else:
            h = F.adaptive_avg_pool2d(h, output_size=1) * h.shape[-1] * h.shape[-2]
        h = h.view(h.size(0), -1)
        return self.head(h).view(-1)

class ResConvGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int = 100,
        channels: int = 4,
        features: int = 64,
        upsample_mode: str = "nearest",
        norm_type: str = "group",
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features * 8, kernel_size=4, stride=1, padding=0, bias=False),
            _make_norm(norm_type=norm_type, channels=features * 8, spatial_size=4),
            nn.ReLU(True),
        )
        self.up1 = self._up_block(features * 8, features * 4, spatial_size=8, upsample_mode=upsample_mode, norm_type=norm_type)
        self.up2 = self._up_block(features * 4, features * 2, spatial_size=16, upsample_mode=upsample_mode, norm_type=norm_type)
        self.up3 = self._up_block(features * 2, features, spatial_size=32, upsample_mode=upsample_mode, norm_type=norm_type)
        self.up4 = self._up_block(features, features, spatial_size=64, upsample_mode=upsample_mode, norm_type=norm_type)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def _up_block(self, in_c: int, out_c: int, spatial_size: int, upsample_mode: str, norm_type: str) -> nn.Module:
        align_corners = False if upsample_mode == "bilinear" else None
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=align_corners),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm(norm_type=norm_type, channels=out_c, spatial_size=spatial_size),
            nn.ReLU(True),
        )

    def forward(self, z):
        x = self.proj(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.to_rgb(x)
