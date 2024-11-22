from typing import Iterator, Dict, Union
from PIL import Image

from datasets import load_dataset


def get_dataset_iterator() -> Iterator[Dict[str, Union[int, Image.Image]]]:
    dataset = load_dataset(
        "imagenet-1k",
        split="validation",
        trust_remote_code=True,
        streaming=True,
    )

    return iter(dataset)
