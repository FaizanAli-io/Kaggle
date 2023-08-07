import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim) -> None:
        super().__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)
