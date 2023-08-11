import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_d) -> None:
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            # Blocks
            self._block(z_dim, features_d * 16, 4, 1, 0),
            self._block(features_d * 16, features_d * 8, 4, 2, 1),
            self._block(features_d * 8, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 2, 4, 2, 1),

            # Final Layer
            nn.ConvTranspose2d(features_d * 2, img_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)
