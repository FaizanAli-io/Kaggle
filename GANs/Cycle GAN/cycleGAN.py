import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import albumentations as A

from utils import save_checkpoint, load_checkpoint, seed_everything
from dataset import CustomDataset
from discriminator import Discriminator
from generator import Generator

from tqdm import tqdm


class CycleGAN:
    def __init__(self):
        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")

        self.SAVE = True
        self.LOAD = False

        self.NUM_EPOCHS = 10
        self.LEARNING_RATE = 1e-5
        self.LAMBDA_CYCLE = 10.0
        self.LAMBDA_IDENTITY = 0.0

        self.CHECKPOINT_DSC_X = "checkpoints/dscx.pth.tar"
        self.CHECKPOINT_DSC_Y = "checkpoints/dscy.pth.tar"
        self.CHECKPOINT_GEN_X = "checkpoints/genx.pth.tar"
        self.CHECKPOINT_GEN_Y = "checkpoints/geny.pth.tar"

        IMAGE_CHANNELS = 3
        DSC_FEATURES = [64, 128, 256, 512]
        GEN_FEATURES = 64
        NUM_RESIDUALS = 9

        BATCH_SIZE = 1
        NUM_WORKERS = 4
        ROOT_PATH_1 = ""
        ROOT_PATH_2 = ""

        transforms = A.Compose(
            [
                A.Resize(width=256, height=256),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.5 for _ in range(IMAGE_CHANNELS)],
                    std=[0.5 for _ in range(IMAGE_CHANNELS)],
                    max_pixel_value=255,
                ),
                A.pytorch.ToTensorV2(),
            ],
            additional_targets={"image1": "image2"},
        )

        self.dscX = Discriminator(IMAGE_CHANNELS, DSC_FEATURES).to(self.DEVICE)
        self.dscY = Discriminator(IMAGE_CHANNELS, DSC_FEATURES).to(self.DEVICE)
        self.genX = Generator(IMAGE_CHANNELS, GEN_FEATURES,
                              NUM_RESIDUALS).to(self.DEVICE)
        self.genY = Generator(IMAGE_CHANNELS, GEN_FEATURES,
                              NUM_RESIDUALS).to(self.DEVICE)

        self.opt_dsc = optim.Adam(
            list(self.dscX.parameters()) + list(self.dscY.parameters()),
            lr=self.LEARNING_RATE,
            betas=(0.5, 0.999),
        )

        self.opt_gen = optim.Adam(
            list(self.genX.parameters()) + list(self.genY.parameters()),
            lr=self.LEARNING_RATE,
            betas=(0.5, 0.999),
        )

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        dataset = CustomDataset(
            root1=ROOT_PATH_1,
            root2=ROOT_PATH_2,
            trans=transforms
        )

        self.dataloader = DataLoader(
            shuffle=True,
            pin_memory=True,
            dataset=dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        self.gen_scaler = torch.cuda.amp.GradScaler()
        self.dsc_scaler = torch.cuda.amp.GradScaler()

        if self.LOAD:
            self.load_model()

    def train_model(self):
        looper = tqdm(self.dataloader)

        for idx, (X, Y) in enumerate(looper):
            pass

    def load_model(self):
        load_checkpoint(self.CHECKPOINT_DSC_X, self.dscX,
                        self.opt_dsc, self.LEARNING_RATE)
        load_checkpoint(self.CHECKPOINT_DSC_Y, self.dscY,
                        self.opt_dsc, self.LEARNING_RATE)
        load_checkpoint(self.CHECKPOINT_GEN_X, self.genX,
                        self.opt_gen, self.LEARNING_RATE)
        load_checkpoint(self.CHECKPOINT_GEN_Y, self.genY,
                        self.opt_gen, self.LEARNING_RATE)

    def save_model(self):
        save_checkpoint(self.CHECKPOINT_DSC_X, self.dscX, self.opt_dsc)
        save_checkpoint(self.CHECKPOINT_DSC_Y, self.dscY, self.opt_dsc)
        save_checkpoint(self.CHECKPOINT_GEN_X, self.genX, self.opt_gen)
        save_checkpoint(self.CHECKPOINT_GEN_Y, self.genY, self.opt_gen)
