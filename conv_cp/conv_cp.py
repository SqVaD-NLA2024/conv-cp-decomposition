from typing import Optional, Literal

from torch import nn
import tensorly as tl

tl.set_backend("pytorch")


class CPConv2d(nn.Module):
    def __init__(
        self, layer: nn.Conv2d, rank: int, cp_init: Literal["svd", "random"] = "svd"
    ):
        super().__init__()
        if layer.groups != 1:
            raise ValueError("Grouped convolutions are not supported")

        _, (t, s, y, x) = tl.decomposition.parafac(
            layer.weight.data, rank=rank, init=cp_init
        )
        in_c = s.shape[0]
        out_c = t.shape[0]
        k_y = y.shape[0]
        k_x = x.shape[0]
        pad_y = layer.padding[0]
        pad_x = layer.padding[1]
        stride_y = layer.stride[0]
        stride_x = layer.stride[1]
        dilation_y = layer.dilation[0]
        dilation_x = layer.dilation[1]

        self.model = nn.Sequential(
            nn.Conv2d(in_c, rank, 1, bias=False),
            nn.Conv2d(
                rank,
                rank,
                (k_y, 1),
                bias=False,
                groups=rank,
                padding=(pad_y, 0),
                stride=(stride_y, 1),
                dilation=(dilation_y, 1),
            ),
            nn.Conv2d(
                rank,
                rank,
                (1, k_x),
                bias=False,
                groups=rank,
                padding=(0, pad_x),
                stride=(1, stride_x),
                dilation=(1, dilation_x),
            ),
            nn.Conv2d(rank, out_c, 1, bias=True),
        )
        self.model[0].weight.data = s.T[:, :, None, None]
        self.model[1].weight.data = y.T[:, None, :, None]
        self.model[2].weight.data = x.T[:, None, None, :]
        self.model[3].weight.data = t[:, :, None, None]
        self.model[3].bias.data = layer.bias

    def forward(self, x):
        return self.model.forward(x)


def decompose_model(
    model: nn.Module,
    rank: Optional[int] = None,
    coef: Optional[float] = None,
    min_rank: int = 3,
    max_rank: int = 20,
    cp_init: Literal["svd", "random"] = "random",
) -> nn.Module:
    if rank is None and coef is None:
        raise ValueError("Either rank or coef should be provided")
    if rank is not None and coef is not None:
        raise ValueError("Only one of rank or coef should be provided")

    if rank is not None:
        for i, layer in model.features.named_children():
            if isinstance(layer, nn.Conv2d):
                layer = CPConv2d(layer, rank, cp_init)
                model.features._modules[i] = layer

    if coef is not None:
        for i, layer in model.features.named_children():
            if isinstance(layer, nn.Conv2d):
                rank_ = int(coef * layer.weight.numel() / sum(layer.weight.shape))
                rank_ = min(max(rank_, min_rank), max_rank)
                layer = CPConv2d(layer, rank_, cp_init)
                model.features._modules[i] = layer

    return model
