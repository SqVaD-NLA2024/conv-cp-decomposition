from pydantic import BaseModel
import torch


class FGSMOutput(BaseModel, arbitrary_types_allowed=True):
    success: bool
    adversarial_images: torch.Tensor
    adversarial_labels: torch.LongTensor
    n_steps: int


class EvaluationResult(BaseModel):
    accuracy_score: float
    n_params: int
    inference_time: float
