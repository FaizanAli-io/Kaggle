import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, img_channels, features_d) -> None:
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            # Initial Layer
            nn.Conv2d(img_channels, features_d,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Blocks
            self._block(features_d * 1, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),

            # Final Layer
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.critic(x)
