import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from gen import Generator
from critic import Critic
from utils import gradient_penalty


class WGAN:
    def __init__(self):
        self.lr = 1e-4
        self.image_size = 64
        self.image_channels = 1
        self.gen_features = 16
        self.critic_features = 16

        self.z_dim = 100
        self.lambda_gp = 10
        self.batch_size = 64
        self.critic_iterations = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mytransforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [.5 for _ in range(self.image_channels)],
                [.5 for _ in range(self.image_channels)],
            ),
        ])

        self.loader = DataLoader(
            datasets.MNIST(
                root="dataset/",
                train=True,
                transform=self.mytransforms,
                download=True,
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.gen = Generator(self.z_dim, self.image_channels,
                             self.gen_features).to(self.device)
        self.critic = Critic(self.image_channels,
                             self.critic_features).to(self.device)

        self.initialize_weights(self.gen)
        self.initialize_weights(self.critic)
        self.test_shape()

        self.opt_gen = optim.Adam(
            self.gen.parameters(), lr=self.lr, betas=(0.0, 0.9))
        self.opt_critic = optim.Adam(
            self.critic.parameters(), lr=self.lr, betas=(0.0, 0.9))

        self.fixed_noise = torch.randn(32, self.z_dim, 1, 1)
        self.writer_fake = SummaryWriter("runs/GAN_MNIST/fake")
        self.writer_real = SummaryWriter("runs/GAN_MNIST/real")
        self.step = 0

        self.gen.train()
        self.critic.train()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for batch_idx, (real, _) in enumerate(tqdm(self.loader)):
                real = real.to(self.device)
                cur_batch_size = real.shape[0]

                # Critic Loss Function: maximize [log(D(x)) + log(1 - D(G(z)))]
                for _ in range(self.critic_iterations):
                    noise = torch.randn(
                        (cur_batch_size, self.z_dim, 1, 1)).to(self.device)
                    fake = self.gen(noise)

                    critic_real = self.critic(real).reshape(-1)
                    critic_fake = self.critic(fake).reshape(-1)
                    gp = gradient_penalty(self.critic, real, fake, self.device)

                    loss_critic = torch.mean(
                        critic_fake) - torch.mean(critic_real)
                    loss_critic += self.lambda_gp * gp

                    self.critic.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    self.opt_critic.step()

                # Generator Loss Function: min [log(1 - D(G(z)))]
                # => max [log(D(G(z)))]
                critic_fake = self.critic(fake).reshape(-1)
                loss_gen = -torch.mean(critic_fake)

                self.gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()

                # Tensor Board Code:
                if batch_idx % 100 == 0:
                    self.log_and_save(epoch, num_epochs,
                                      loss_critic, loss_gen, real)

    def log_and_save(self, epoch, num_epochs,
                     loss_critic, loss_gen, real):
        print(f"Epoch: {epoch}/{num_epochs}")
        print(f"Critic Loss: {loss_critic:.4f}")
        print(f"Generator Loss: {loss_gen:.4f}")

        with torch.no_grad():
            fake = self.gen(self.fixed_noise)

            fake_grid = torchvision.utils.make_grid(
                fake[:32], normalize=True)
            real_grid = torchvision.utils.make_grid(
                real[:32], normalize=True)

            self.writer_fake.add_image(
                "MNIST Fake", fake_grid, global_step=self.step)
            self.writer_real.add_image(
                "MNIST Real", real_grid, global_step=self.step)

        self.step += 1

    def test_shape(self):
        (N, in_channel, H, W) = (
            self.batch_size,
            self.image_channels,
            self.image_size,
            self.image_size,
        )

        x = torch.randn((N, in_channel, H, W))
        assert self.critic(x).shape == (N, 1, 1, 1)

        x = torch.randn((N, self.z_dim, 1, 1))
        assert self.gen(x).shape == (N, in_channel, H, W)

        print("Shape Check Passed")

    @staticmethod
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":
    mygan = WGAN()
    mygan.train(10)
