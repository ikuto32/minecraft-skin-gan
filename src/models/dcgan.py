import torch.nn as nn
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
