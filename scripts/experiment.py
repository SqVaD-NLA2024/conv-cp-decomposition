# region imports
from typing import Callable, Tuple, Any, Dict
from copy import deepcopy
import json
import gc
import logging

import numpy as np
import torch
from torch import nn
from torchvision.models import Weights
from torchvision.models import (
    alexnet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vgg11,
    vgg16,
    vgg19,
    vgg11_bn,
    vgg16_bn,
    vgg19_bn,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    mobilenet_v2,
    mobilenet_v3_small,
    mobilenet_v3_large,
    inception_v3,
    googlenet,
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
    mnasnet0_5,
    mnasnet0_75,
    mnasnet1_0,
    mnasnet1_3,
    swin_b,
    swin_t,
    swin_s,
    swin_v2_b,
    swin_v2_t,
    swin_v2_s,
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    wide_resnet50_2,
    wide_resnet101_2,
    resnext50_32x4d,
    resnext101_32x8d,
    resnext101_64x4d,
    squeezenet1_0,
    squeezenet1_1,
    AlexNet_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    VGG11_Weights,
    VGG16_Weights,
    VGG19_Weights,
    VGG11_BN_Weights,
    VGG16_BN_Weights,
    VGG19_BN_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
    DenseNet121_Weights,
    DenseNet161_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
    Inception_V3_Weights,
    GoogLeNet_Weights,
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
    ShuffleNet_V2_X1_5_Weights,
    ShuffleNet_V2_X2_0_Weights,
    MNASNet0_5_Weights,
    MNASNet0_75_Weights,
    MNASNet1_0_Weights,
    MNASNet1_3_Weights,
    Swin_B_Weights,
    Swin_T_Weights,
    Swin_S_Weights,
    Swin_V2_B_Weights,
    Swin_V2_T_Weights,
    Swin_V2_S_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    ResNeXt101_64X4D_Weights,
    SqueezeNet1_0_Weights,
    SqueezeNet1_1_Weights,
)

from conv_cp.conv_cp import decompose_model
from conv_cp.evaluation import evaluate_model
from conv_cp.adversarial import FGSM
from conv_cp.dataset import get_dataset_iterator


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# endregion

# region constants
MODELS = [
    {"model": alexnet, "weights": AlexNet_Weights.IMAGENET1K_V1},
    {"model": resnet18, "weights": ResNet18_Weights.IMAGENET1K_V1},
    {"model": resnet34, "weights": ResNet34_Weights.IMAGENET1K_V1},
    {"model": resnet50, "weights": ResNet50_Weights.IMAGENET1K_V2},
    {"model": resnet101, "weights": ResNet101_Weights.IMAGENET1K_V2},
    {"model": resnet152, "weights": ResNet152_Weights.IMAGENET1K_V2},
    {"model": vgg11, "weights": VGG11_Weights.IMAGENET1K_V1},
    {"model": vgg16, "weights": VGG16_Weights.IMAGENET1K_V1},
    {"model": vgg19, "weights": VGG19_Weights.IMAGENET1K_V1},
    {"model": vgg11_bn, "weights": VGG11_BN_Weights.IMAGENET1K_V1},
    {"model": vgg16_bn, "weights": VGG16_BN_Weights.IMAGENET1K_V1},
    {"model": vgg19_bn, "weights": VGG19_BN_Weights.IMAGENET1K_V1},
    {"model": efficientnet_b0, "weights": EfficientNet_B0_Weights.IMAGENET1K_V1},
    {"model": efficientnet_b1, "weights": EfficientNet_B1_Weights.IMAGENET1K_V1},
    {"model": efficientnet_b2, "weights": EfficientNet_B2_Weights.IMAGENET1K_V1},
    {"model": efficientnet_b3, "weights": EfficientNet_B3_Weights.IMAGENET1K_V1},
    {"model": efficientnet_b4, "weights": EfficientNet_B4_Weights.IMAGENET1K_V1},
    {"model": efficientnet_b5, "weights": EfficientNet_B5_Weights.IMAGENET1K_V1},
    {"model": efficientnet_b6, "weights": EfficientNet_B6_Weights.IMAGENET1K_V1},
    {"model": efficientnet_b7, "weights": EfficientNet_B7_Weights.IMAGENET1K_V1},
    {"model": efficientnet_v2_l, "weights": EfficientNet_V2_L_Weights.IMAGENET1K_V1},
    {"model": efficientnet_v2_m, "weights": EfficientNet_V2_M_Weights.IMAGENET1K_V1},
    {"model": efficientnet_v2_s, "weights": EfficientNet_V2_S_Weights.IMAGENET1K_V1},
    {"model": densenet121, "weights": DenseNet121_Weights.IMAGENET1K_V1},
    {"model": densenet161, "weights": DenseNet161_Weights.IMAGENET1K_V1},
    {"model": densenet169, "weights": DenseNet169_Weights.IMAGENET1K_V1},
    {"model": densenet201, "weights": DenseNet201_Weights.IMAGENET1K_V1},
    {"model": mobilenet_v2, "weights": MobileNet_V2_Weights.IMAGENET1K_V2},
    {"model": mobilenet_v3_small, "weights": MobileNet_V3_Small_Weights.IMAGENET1K_V1},
    {"model": mobilenet_v3_large, "weights": MobileNet_V3_Large_Weights.IMAGENET1K_V2},
    {"model": inception_v3, "weights": Inception_V3_Weights.IMAGENET1K_V1},
    {"model": googlenet, "weights": GoogLeNet_Weights.IMAGENET1K_V1},
    {"model": shufflenet_v2_x0_5, "weights": ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1},
    {"model": shufflenet_v2_x1_0, "weights": ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1},
    {"model": shufflenet_v2_x1_5, "weights": ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1},
    {"model": shufflenet_v2_x2_0, "weights": ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1},
    {"model": mnasnet0_5, "weights": MNASNet0_5_Weights.IMAGENET1K_V1},
    {"model": mnasnet0_75, "weights": MNASNet0_75_Weights.IMAGENET1K_V1},
    {"model": mnasnet1_0, "weights": MNASNet1_0_Weights.IMAGENET1K_V1},
    {"model": mnasnet1_3, "weights": MNASNet1_3_Weights.IMAGENET1K_V1},
    {"model": swin_b, "weights": Swin_B_Weights.IMAGENET1K_V1},
    {"model": swin_t, "weights": Swin_T_Weights.IMAGENET1K_V1},
    {"model": swin_s, "weights": Swin_S_Weights.IMAGENET1K_V1},
    {"model": swin_v2_b, "weights": Swin_V2_B_Weights.IMAGENET1K_V1},
    {"model": swin_v2_t, "weights": Swin_V2_T_Weights.IMAGENET1K_V1},
    {"model": swin_v2_s, "weights": Swin_V2_S_Weights.IMAGENET1K_V1},
    {"model": convnext_tiny, "weights": ConvNeXt_Tiny_Weights.IMAGENET1K_V1},
    {"model": convnext_small, "weights": ConvNeXt_Small_Weights.IMAGENET1K_V1},
    {"model": convnext_base, "weights": ConvNeXt_Base_Weights.IMAGENET1K_V1},
    {"model": convnext_large, "weights": ConvNeXt_Large_Weights.IMAGENET1K_V1},
    {"model": wide_resnet50_2, "weights": Wide_ResNet50_2_Weights.IMAGENET1K_V2},
    {"model": wide_resnet101_2, "weights": Wide_ResNet101_2_Weights.IMAGENET1K_V2},
    {"model": resnext50_32x4d, "weights": ResNeXt50_32X4D_Weights.IMAGENET1K_V2},
    {"model": resnext101_32x8d, "weights": ResNeXt101_32X8D_Weights.IMAGENET1K_V2},
    {"model": resnext101_64x4d, "weights": ResNeXt101_64X4D_Weights.IMAGENET1K_V1},
    {"model": squeezenet1_0, "weights": SqueezeNet1_0_Weights.IMAGENET1K_V1},
    {"model": squeezenet1_1, "weights": SqueezeNet1_1_Weights.IMAGENET1K_V1},
]

CONFIGS = [
    {"coef": 1},
    {"coef": 0.7, "min_rank": 3, "max_rank": 700},
    {"coef": 0.6, "min_rank": 3, "max_rank": 650},
    {"coef": 0.5, "min_rank": 3, "max_rank": 600},
    {"coef": 0.45, "min_rank": 3, "max_rank": 550},
    {"coef": 0.4, "min_rank": 3, "max_rank": 500},
    {"coef": 0.35, "min_rank": 3, "max_rank": 500},
    {"coef": 0.3, "min_rank": 3, "max_rank": 475},
    {"coef": 0.25, "min_rank": 3, "max_rank": 450},
    {"coef": 0.2, "min_rank": 3, "max_rank": 425},
    {"coef": 0.15, "min_rank": 3, "max_rank": 375}
]
# endregion


# region code
class CombModel(nn.Module):
    def __init__(self, model: nn.Module, transform: nn.Module):
        super().__init__()
        self.model = model
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.transform(x))


def get_model(
    model_name: Callable[[Weights], nn.Module], weights: Weights
) -> Tuple[nn.Module, nn.Module]:
    model = model_name(weights=weights)
    transform = weights.transforms()

    return model, transform


def evaluate_adv_resistance(
    model: nn.Module, transform: nn.Module, n_samples: int
) -> float:
    comb_model = CombModel(model, transform)
    dataset_iter = get_dataset_iterator()

    samples = [next(dataset_iter) for _ in range(n_samples)]
    images = [np.array(sample["image"]) for sample in samples]
    num_steps = []
    for image in images:
        image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255

        result = FGSM(
            comb_model,
            image,
            epsilon=0.0005,
            device="cuda",
        )

        num_steps.append(result.n_steps)

    return np.mean(num_steps).item()


def process_model(
    model: nn.Module, transform: nn.Module, config: Dict[str, Any]
) -> Dict[str, Any]:
    decomp_model = deepcopy(model)
    if config["coef"] < 1:
        decomp_model = decompose_model(decomp_model, **config)

    result = evaluate_model(
        decomp_model,
        transform,
        device="cuda",
        batch_size=32,
        total_samples=1000,
    )

    experiment_info = deepcopy(config)
    experiment_info.update(
        {
            "accuracy_score": result.accuracy_score,
            "n_params": result.n_params,
            "inference_time": result.inference_time,
        }
    )
    experiment_info["adv_resistance"] = evaluate_adv_resistance(
        decomp_model, transform, 64
    )

    decomp_model.cpu()
    del decomp_model
    torch.cuda.empty_cache()
    gc.collect()

    return experiment_info


def main():
    experiment_results = []
    for model_conf in MODELS:
        logging.info(f"processing {model_conf['model'].__name__}")
        model, transform = get_model(model_conf["model"], model_conf["weights"])
        model.eval()

        for config in CONFIGS:
            logging.info(f"processing config {config}")
            try:
                experiment_info = process_model(model, transform, config)
                experiment_info["model"] = model_conf["model"].__name__
                logging.info(f"experiment result: {experiment_info}")
                experiment_results.append(experiment_info)
            except Exception:
                logging.exception(f"failed to process {model_conf['model'].__name__}")
                experiment_info = {
                    "model": model_conf["model"].__name__,
                    "error": True,
                    **config,
                }

    with open("result.json", "w") as f:
        json.dump(experiment_results, f)


# endregion

if __name__ == "__main__":
    main()
