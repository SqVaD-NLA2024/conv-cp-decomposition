import time
from typing import Tuple, Union

import torch
from torch import nn

from conv_cp.dataset import get_dataset
from conv_cp.models import EvaluationResult
from conv_cp.utils import count_parameters


@torch.no_grad()
def process_images(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.LongTensor,
    device: Union[str, torch.DeviceObjType] = "cpu",
) -> Tuple[torch.Tensor, float, int]:
    images = images.to(device)

    start_time = time.time()
    output = model.forward(images)
    end_time = time.time()

    pred_label = torch.argmax(output, dim=1).cpu()
    return output, end_time - start_time, torch.sum(pred_label == labels).item()


def preprocess_image(image, transform):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = transform(image)
    return image


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    transform: nn.Module,
    device: Union[str, torch.DeviceObjType] = "cpu",
    batch_size: int = 32,
    total_samples: int = 50000,
) -> EvaluationResult:
    dataset = get_dataset()
    total_samples = min(total_samples, len(dataset))

    model.to(device)

    total_elapsed_time = 0
    running_n_correct_preds = 0
    n_images = 0

    for idx in range(0, total_samples, batch_size):
        samples = dataset[idx : min(idx + batch_size, total_samples)]
        images = [preprocess_image(sample["image"], transform) for sample in samples]
        labels = [sample["label"] for sample in samples]

        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels).to(dtype=torch.long)

        _, elapsed_time, is_correct = process_images(model, images, labels, device)

        total_elapsed_time += elapsed_time
        running_n_correct_preds += is_correct
        n_images += len(images)

    return EvaluationResult(
        accuracy_score=running_n_correct_preds / n_images,
        inference_time=total_elapsed_time,
        n_params=count_parameters(model),
    )
