import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=stride,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                4, 2, 1,
                padding_mode="reflect",
            ), nn.LeakyReLU(0.2),
        )

        layers = []
        for i in range(1, len(features)):
            layers.append(CNNBlock(
                in_channels=features[i-1],
                out_channels=features[i],
                stride=(1 if i == len(features) - 1 else 2),
            ))

        layers.append(nn.Conv2d(
            features[-1], 1, 4, 1, 1,
            padding_mode="reflect",
        ))

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(self.initial(x))


def test():
    N = 32
    x = torch.randn(N, 3, 256, 256)
    y = torch.randn(N, 3, 256, 256)
    model = Discriminator()
    assert model(x, y).shape == (N, 1, 26, 26)


if __name__ == "__main__":
    test()
