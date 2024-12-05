import os
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset

from conv_cp.imagenet.classes import IMAGENET2012_CLASSES


class ImageNet(Dataset):
    def __init__(self, root_dir: str, transform: nn.Module):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(self.root_dir)
        self.label_map = {k: i for i, k in enumerate(IMAGENET2012_CLASSES)}

    def __len__(self) -> int:
        return len(self.files)

    def _load_image(self, idx: int) -> torch.Tensor:
        image = Image.open(os.path.join(self.root_dir, self.files[idx]))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.transform(image)

    def _get_label(self, idx: int) -> int:
        root, _ = self.files[idx].rsplit(".", maxsplit=1)
        _, label_id = root.rsplit("_", maxsplit=1)
        return self.label_map[label_id]

    def split(self, ratio: float):
        train_size = int(len(self) * ratio)

        train_dataset = ImageNet(self.root_dir, self.transform)
        train_dataset.files = self.files[:train_size]

        test_dataset = ImageNet(self.root_dir, self.transform)
        test_dataset.files = self.files[train_size:]

        return train_dataset, test_dataset

    def get_raw_img(self, idx: int) -> Image.Image:
        image = Image.open(os.path.join(self.root_dir, self.files[idx]))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def __getitem__(self, idx):
        return self._load_image(idx), self._get_label(idx)
