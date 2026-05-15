import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


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
