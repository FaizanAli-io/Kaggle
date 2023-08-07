import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from disc import Discriminator
from gen import Generator


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test_shape():
    N, in_channel, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channel, H, W))
    disc = Discriminator(in_channel, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)

    x = torch.randn((N, z_dim, 1, 1))
    gen = Generator(z_dim, in_channel, 8)
    initialize_weights(gen)
    assert gen(x).shape == (N, in_channel, H, W)

    print("Shape Check Passed")


def run():
    test_shape()

    lr = 2e-4
    z_dim = 100
    num_epochs = 5
    image_size = 64
    batch_size = 128
    gen_features = 64
    disc_features = 64
    image_channels = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mytransforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [.5 for _ in range(image_channels)],
            [.5 for _ in range(image_channels)],
        ),
    ])

    dataset = datasets.MNIST(
        root="dataset/", train=True, transform=mytransforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    gen = Generator(z_dim, image_channels, gen_features).to(device)
    disc = Discriminator(image_channels, disc_features).to(device)
    initialize_weights(gen)
    initialize_weights(disc)
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(32, z_dim, 1, 1)

    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
    step = 0

    gen.train()
    disc.train()

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)

            noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
            fake = gen(noise)

            # Discriminator Loss Function: maximize [log(D(x)) + log(1 - D(G(z)))]
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            # Generator Loss Function: min [log(1 - D(G(z)))] <--> max [log(D(G(z)))]
            gen_out = disc(fake).reshape(-1)
            loss_gen = criterion(gen_out, torch.ones_like(gen_out))

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            print(f"Batch Index: {batch_idx}")

            # Tensorboard Code:
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}/{num_epochs}")
                print(f"Discriminator Loss: {loss_disc:.4f}")
                print(f"Generator Loss: {loss_gen:.4f}")

                with torch.no_grad():
                    fake = gen(fixed_noise)

                    fake_grid = torchvision.utils.make_grid(
                        fake[:32], normalize=True)
                    real_grid = torchvision.utils.make_grid(
                        real[:32], normalize=True)

                    writer_fake.add_image(
                        "MNIST Fake", fake_grid, global_step=step)
                    writer_real.add_image(
                        "MNIST Real", real_grid, global_step=step)

                step += 1


run()
