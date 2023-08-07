import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_dims) -> None:
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dims, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)
