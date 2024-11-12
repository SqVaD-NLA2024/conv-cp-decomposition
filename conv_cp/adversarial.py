from typing import Optional
import torch
from torch import nn
from conv_cp.models import FGSMOutput


def FGSM(
    model: nn.Module,
    images: torch.Tensor,
    labels: Optional[torch.LongTensor],
    epsilon: float = 0.05,
    threshold: Optional[float] = None,
) -> FGSMOutput:
    """
    Perform the Fast Gradient Sign Method attack on the model using the given images and labels.

    Parameters:
        model: The model to attack.
        images: The images of shape (B, C, H, W) to attack. Expected to be in the range [0, 1].
        labels: The target labels of shape (B,) for the images.
        If None, function will stop when any other class is the most probable
        epsilon: The step size for the attack.
        threshold: The minimal probability for the target class. If None, will stop when the target class is the most probable.

    Returns:
        A FGSMOutput object containing the results of the attack.
    """
    images = images.clone().detach().requires_grad_(True)

    with torch.no_grad():
        correct_labels = model(images).argmax(dim=1)

    for step in range(100):
        images.grad = None
        images.requires_grad = True
        logits = model(images)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, correct_labels)

        loss.backward()

        with torch.no_grad():
            if labels is not None:
                images = images - epsilon * images.grad.sign()
            else:
                images = images + epsilon * images.grad.sign()
            images = torch.clamp(images, 0, 1)
            new_logits = model(images)

        probs = torch.nn.functional.softmax(new_logits, dim=1)
        new_labels = new_logits.argmax(dim=1)
        if labels is not None:
            if threshold is not None and torch.all(probs[:, labels] >= threshold):
                return FGSMOutput(
                    success=True,
                    adversarial_images=images,
                    adversarial_labels=new_labels,
                    n_steps=step + 1,
                )
            elif threshold is None and torch.all(new_labels == labels):
                return FGSMOutput(
                    success=True,
                    adversarial_images=images,
                    adversarial_labels=new_labels,
                    n_steps=step + 1,
                )
        elif torch.all(new_labels != correct_labels):
            return FGSMOutput(
                success=True,
                adversarial_images=images,
                adversarial_labels=new_labels,
                n_steps=step + 1,
            )

    return FGSMOutput(
        success=False,
        adversarial_images=images,
        adversarial_labels=new_labels,
        n_steps=step + 1,
    )
