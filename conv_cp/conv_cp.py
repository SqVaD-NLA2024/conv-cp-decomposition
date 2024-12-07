from typing import Literal, Generator, Tuple, Callable, Optional, Dict
from copy import deepcopy

import torch
from torch import nn
import tensorly as tl

tl.set_backend("pytorch")


class SVDLinear(nn.Module):
    def __init__(self, orig_layer: nn.Module, rank: int):
        super().__init__()
        self.bias = None
        if isinstance(orig_layer, nn.Linear):
            u, s, v = torch.linalg.svd(orig_layer.weight.T, full_matrices=False)
            sqrt_s = torch.sqrt(s)
            if orig_layer.bias is not None:
                self.bias = nn.Parameter(orig_layer.bias)
            self.rank = rank

            self.u = nn.Parameter(u[:, :rank] * sqrt_s[:rank])
            self.v = nn.Parameter(sqrt_s[:rank, None] * v[:rank])

        else:
            raise NotImplementedError

    def forward(self, x):
        if self.bias is not None:
            return x @ self.u @ self.v + self.bias
        return x @ self.u @ self.v


class CPLinear(nn.Module):
    def __init__(
        self,
        layer: nn.Linear,
        rank: int,
        do_decomp: bool = True,
        cp_init: Literal["svd", "random"] = "random",
    ):
        super().__init__()
        out_dim, in_dim = layer.weight.shape

        self.model = nn.Sequential(
            nn.Linear(in_dim, rank, bias=False),
            nn.Linear(rank, out_dim, bias=layer.bias is not None),
        )

        if do_decomp:
            _, (w2, w1) = tl.decomposition.parafac(
                layer.weight.data, rank=rank, init=cp_init
            )
            self.model[0].weight.data = w1.T
            self.model[1].weight.data = w2
            if layer.bias is not None:
                self.model[1].bias.data = layer.bias

    def forward(self, x):
        return self.model.forward(x)


class CPConv2d(nn.Module):
    def __init__(
        self,
        layer: nn.Conv2d,
        rank: int,
        do_cp: bool = True,
        cp_init: Literal["svd", "random"] = "random",
    ):
        super().__init__()
        if layer.groups != 1:
            raise ValueError("Grouped convolutions are not supported")

        out_c, in_c, k_y, k_x = layer.weight.shape
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
            nn.Conv2d(rank, out_c, 1, bias=layer.bias is not None),
        )

        if do_cp:
            _, (t, s, y, x) = tl.decomposition.parafac(
                layer.weight.data, rank=rank, init=cp_init
            )
            self.model[0].weight.data = s.T[:, :, None, None]
            self.model[1].weight.data = y.T[:, None, :, None]
            self.model[2].weight.data = x.T[:, None, None, :]
            self.model[3].weight.data = t[:, :, None, None]
            if layer.bias is not None:
                self.model[3].bias.data = layer.bias

    def forward(self, x):
        return self.model.forward(x)


def decompose_model(
    model: nn.Module,
    conv_rank: int,
    fc_rank: int,
    loss_fn: Callable[[nn.Module], float],
    train_fn: Callable[[nn.Module], None],
    trial_rank: int = 5,
    layer_size_regularization: float = 0.0,
    linear_decomp_type: Literal["cp", "svd"] = "cp",
    freeze_decomposed: bool = False,
    verbose: bool = False,
    predefined_ranks: Optional[Dict[str, int]] = None,
) -> nn.Module:
    def debug(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if predefined_ranks is None:
        predefined_ranks = {}

    conv_layers = list(_get_conv_layers(model))
    fc_layers = list(_get_fc_layers(model))

    conv_losses = {}
    conv_ranks = {}
    conv_sizes = {}
    total_conv_rank = conv_rank
    debug("Computing losses for conv layers")
    for name, conv_layer in conv_layers:
        if name in predefined_ranks:
            debug(f"Using predefined rank {predefined_ranks[name]} for {name}")
            total_conv_rank -= predefined_ranks[name]
            conv_ranks[name] = predefined_ranks[name]
            continue
        debug(f"Processing module {name}")
        cp_conv_layer = CPConv2d(conv_layer, rank=trial_rank)
        cp_model = deepcopy(model)
        cp_model.set_submodule(name, cp_conv_layer)
        conv_losses[name] = loss_fn(cp_model)
        conv_sizes[name] = _get_n_params(cp_conv_layer)

    total_conv_loss = sum(conv_losses.values())
    total_conv_size = sum(conv_sizes.values())
    for name, loss in conv_losses.items():
        loss_prop = loss / total_conv_loss
        size_prop = conv_sizes[name] / total_conv_size
        rank_prop = (
            1 - layer_size_regularization
        ) * loss_prop + layer_size_regularization * size_prop

        rank = int(rank_prop * total_conv_rank)
        conv_ranks[name] = rank

    conv_names = list(conv_ranks.keys())
    for i in range(total_conv_rank - sum(conv_ranks.values())):
        key = conv_names[i % len(conv_layers)]
        conv_ranks[key] += 1

    fc_losses = {}
    fc_ranks = {}
    fc_sizes = {}
    total_fc_rank = fc_rank
    debug("Computing losses for fc layers")
    for name, fc_layer in fc_layers:
        if name in predefined_ranks:
            debug(f"Using predefined rank {predefined_ranks[name]} for {name}")
            total_fc_rank -= predefined_ranks[name]
            fc_ranks[name] = predefined_ranks[name]
            continue
        debug(f"Processing module {name}")
        cp_fc_layer = CPLinear(fc_layer, rank=trial_rank)
        cp_model = deepcopy(model)
        cp_model.set_submodule(name, cp_fc_layer)
        fc_losses[name] = loss_fn(cp_model)
        fc_sizes[name] = _get_n_params(cp_fc_layer)

    total_fc_loss = sum(fc_losses.values())
    total_fc_size = sum(fc_sizes.values())
    for name, loss in fc_losses.items():
        loss_prop = loss / total_fc_loss
        size_prop = fc_sizes[name] / total_fc_size
        rank_prop = (
            1 - layer_size_regularization
        ) * loss_prop + layer_size_regularization * size_prop

        rank = int(rank_prop * total_fc_rank)
        fc_ranks[name] = rank

    fc_names = list(fc_ranks.keys())
    for i in range(total_fc_rank - sum(fc_ranks.values())):
        key = fc_names[i % len(fc_layers)]
        fc_ranks[key] += 1

    debug("Initializing CP model")
    for name, rank in conv_ranks.items():
        debug(f"Decomposing {name} with rank {rank}")
        conv_layer = model.get_submodule(name)
        cp_conv_layer = CPConv2d(conv_layer, rank=rank)
        if freeze_decomposed:
            _freeze(cp_conv_layer)
        model.set_submodule(name, cp_conv_layer)
        debug(f"Training model with {name} decomposed")
        train_fn(model)

    for i, (name, rank) in enumerate(fc_ranks.items()):
        debug(f"Decomposing {name} with rank {rank}")
        fc_layer = model.get_submodule(name)
        cp_fc_layer = _get_linear_decomp(fc_layer, rank, linear_decomp_type)
        if freeze_decomposed:
            _freeze(cp_fc_layer)
        model.set_submodule(name, cp_fc_layer)
        if i < len(fc_layers) - 1:
            if not freeze_decomposed:
                debug(f"Training model with {name} decomposed")
                train_fn(model)

    return model


def _get_conv_layers(
    model: nn.Module, prefix: str = ""
) -> Generator[Tuple[str, nn.Module], None, None]:
    for name, child in model.named_children():
        full_name = name
        if prefix:
            full_name = f"{prefix}.{name}"
        if isinstance(child, nn.Conv2d):
            yield full_name, child
        else:
            yield from _get_conv_layers(child, prefix=full_name)


def _get_fc_layers(
    model: nn.Module, prefix: str = ""
) -> Generator[Tuple[str, nn.Module], None, None]:
    for name, child in model.named_children():
        full_name = name
        if prefix:
            full_name = f"{prefix}.{name}"
        if isinstance(child, nn.Linear):
            yield full_name, child
        else:
            yield from _get_fc_layers(child, prefix=full_name)


def _get_linear_decomp(layer: nn.Linear, rank: int, decomp_type: Literal["cp", "svd"]):
    if decomp_type == "cp":
        return CPLinear(layer, rank=rank)
    elif decomp_type == "svd":
        return SVDLinear(layer, rank=rank)
    else:
        raise ValueError(f"Unknown decomposition type: {decomp_type}")


def _freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def _get_n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
