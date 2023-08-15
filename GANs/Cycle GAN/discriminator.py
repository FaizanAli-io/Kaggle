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
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
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

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return torch.sigmoid(x)


def test():
    N = 32
    x = torch.randn(N, 3, 256, 256)
    model = Discriminator()
    print(model(x).shape)


if __name__ == "__main__":
    test()
