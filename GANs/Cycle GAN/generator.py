import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True,
                 **kwargs):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                padding_mode="reflect",
                **kwargs,
            ) if down else nn.ConvTranspose2d(
                in_channels,
                out_channels,
                **kwargs,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            CNNBlock(channels, channels,
                     kernel_size=3, padding=1),
            CNNBlock(channels, channels, use_act=False,
                     kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, image_channels, num_features, num_residuals):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                image_channels,
                num_features,
                7, 1, 3,
                padding_mode="reflect",
            ), nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList(
            [
                CNNBlock(num_features*1, num_features*2,
                         kernel_size=3, stride=2, padding=1),
                CNNBlock(num_features*2, num_features*4,
                         kernel_size=3, stride=2, padding=1),
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                CNNBlock(num_features*4, num_features*2, down=False,
                         kernel_size=3, stride=2, padding=1, output_padding=1),
                CNNBlock(num_features*2, num_features*1, down=False,
                         kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Sequential(
            nn.Conv2d(
                num_features,
                image_channels,
                7, 1, 3,
                padding_mode="reflect",
            ), nn.Tanh(),
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return self.last(x)


def test():
    N = 32
    x = torch.randn(N, 3, 256, 256)
    model = Generator()
    print(model(x).shape)


if __name__ == "__main__":
    test()
