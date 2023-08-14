import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down, relu, drop):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4, 2, 1,
                bias=False,
                padding_mode="reflect",
            ) if down else nn.ConvTranspose2d(
                in_channels,
                out_channels,
                4, 2, 1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if relu else nn.LeakyReLU(0.2),
        )

        self.drop = drop
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.drop else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features,
                4, 2, 1,
                padding_mode="reflect",
            ), nn.LeakyReLU(0.2),
        )

        self.down1 = Block(features*1, features*2, True, False, False)
        self.down2 = Block(features*2, features*4, True, False, False)
        self.down3 = Block(features*4, features*8, True, False, False)
        self.down4 = Block(features*8, features*8, True, False, False)
        self.down5 = Block(features*8, features*8, True, False, False)
        self.down6 = Block(features*8, features*8, True, False, False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features*8, features*8, 4, 2, 1,
                padding_mode="reflect",
            ), nn.ReLU(),
        )

        self.up1 = Block(features*8, features*8, False, True, True)
        self.up2 = Block(features*8*2, features*8, False, True, True)
        self.up3 = Block(features*8*2, features*8, False, True, True)
        self.up4 = Block(features*8*2, features*8, False, True, False)
        self.up5 = Block(features*8*2, features*4, False, True, False)
        self.up6 = Block(features*4*2, features*2, False, True, False)
        self.up7 = Block(features*2*2, features*1, False, True, False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final_up(torch.cat([u7, d1], 1))


def test():
    N = 32
    x = torch.randn(N, 3, 256, 256)
    model = Generator()
    assert model(x).shape == (N, 3, 256, 256)


if __name__ == "__main__":
    test()
