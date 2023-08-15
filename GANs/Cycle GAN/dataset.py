import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root1, root2, trans=None):
        self.root1 = root1
        self.root2 = root2
        self.trans = trans

        self.images1 = os.listdir(root1)
        self.images2 = os.listdir(root2)
        self.len1 = len(self.images1)
        self.len2 = len(self.images2)

    def __len__(self):
        return max(self.len1, self.len2)

    def __getitem__(self, index):
        img1 = self.images1[index % self.len1]
        img2 = self.images2[index % self.len2]

        path1 = os.path.join(self.root1, img1)
        path2 = os.path.join(self.root2, img2)

        img1 = np.array(Image.open(path1).convert('RGB'))
        img2 = np.array(Image.open(path2).convert('RGB'))

        if self.trans:
            augments = self.trans(image1=img1, images2=img2)
            img1 = augments["image1"]
            img2 = augments["image2"]

        return img1, img2
