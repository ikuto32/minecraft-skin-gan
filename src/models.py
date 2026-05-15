import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=4, features=64):
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


class Discriminator(nn.Module):
    def __init__(self, channels=4, features=64):
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
