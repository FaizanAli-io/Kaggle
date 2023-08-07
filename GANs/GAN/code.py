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


def run():
    lr = 3e-4
    z_dim = 64
    batch_size = 32
    num_epochs = 50
    image_dim = 28 * 28 * 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    disc = Discriminator(image_dim).to(device)
    gen = Generator(z_dim, image_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    mytransforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,)),
    ])
    dataset = datasets.MNIST(
        root="dataset/", transform=mytransforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCELoss()
    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
    step = 0

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)

            # Discriminator Loss Function: maximize [log(D(x)) + log(1 - D(G(z)))]
            disc_real = disc(real).view(-1)
            disc_fake = disc(fake).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            # Generator Loss Function: min [log(1 - D(G(z)))] <--> max [log(D(G(z)))]
            gen_out = disc(fake).view(-1)
            loss_gen = criterion(gen_out, torch.ones_like(gen_out))

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Tensorboard Code:
            if batch_idx == 1:
                print(f"Epoch: {epoch}/{num_epochs}")
                print(f"Discriminator Loss: {loss_disc:.4f}")
                print(f"Generator Loss: {loss_gen:.4f}")

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    real = real.reshape(-1, 1, 28, 28)

                    fake_grid = torchvision.utils.make_grid(
                        fake, normalize=True)
                    real_grid = torchvision.utils.make_grid(
                        real, normalize=True)

                    writer_fake.add_image(
                        "MNIST Fake", fake_grid, global_step=step)
                    writer_real.add_image(
                        "MNIST Real", real_grid, global_step=step)

                step += 1


run()
