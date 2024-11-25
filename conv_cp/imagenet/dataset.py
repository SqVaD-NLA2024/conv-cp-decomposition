import os
from PIL import Image
from torch.utils.data import Dataset

from conv_cp.imagenet.classes import IMAGENET2012_CLASSES


class ImageNet(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)

    def __len__(self) -> int:
        return len(self.files)

    def _load_image(self, idx: int) -> Image.Image:
        return Image.open(os.path.join(self.root_dir, self.files[idx]))

    def _get_label(self, idx: int) -> int:
        root, _ = self.files[idx].rsplit(".", maxsplit=1)
        _, label_id = root.split("_", maxsplit=1)
        return IMAGENET2012_CLASSES[label_id]

    def __getitem__(self, idx):
        return {"image": self._load_image(idx), "label": self._get_label(idx)}
