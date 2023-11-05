import math
from typing import List, Tuple, Type

import torch
import torch.nn as nn


def append_dims(tensor, target_dims):
    assert isinstance(target_dims, int), f"Expected 'target_dims' to be an integer, but received {type(target_dims)}."
    tensor_dims = tensor.ndim
    assert tensor_dims <= target_dims, f"Tensor has {tensor_dims} dimensions, but target has {target_dims} dimensions."
    return tensor[(...,) + (None,) * (target_dims - tensor_dims)]


def get_weight_decay_parameters(
        model: nn.Module,
        whitelist_modules: Tuple[Type[nn.Module]] = (nn.Linear,)
) -> Tuple[List, List]:
    decay_params = set()

    for module_name, module in model.named_modules():
        if not isinstance(module, whitelist_modules):
            continue
        for param_name, param in module.named_parameters():
            if param_name.endswith('weight') and param.requires_grad:
                full_param_name = f'{module_name}.{param_name}' if module_name else param_name
                decay_params.add(full_param_name)

    param_dict = {param_name: param for param_name, param in model.named_parameters() if param.requires_grad}
    no_decay_params = set(param_dict.keys()) - decay_params

    decay = [param_dict[name] for name in sorted(decay_params)]
    no_decay = [param_dict[name] for name in sorted(no_decay_params)]

    return decay, no_decay


def get_named_float_tensors(model: nn.Module, include_buffers: bool = True) -> List[Tuple[str, torch.Tensor]]:
    unique_names_and_tensors = []

    for name, param in model.named_parameters():
        if param.dtype in [torch.float, torch.float16]:
            unique_names_and_tensors.append((name, param))

    if include_buffers:
        for name, buffer in model.named_buffers():
            if buffer.dtype in [torch.float, torch.float16]:
                unique_names_and_tensors.append((name, buffer))

    return unique_names_and_tensors


@torch.no_grad()
def update_ema_parameters(target_parameters, source_parameters, mu: float = 0.95) -> None:
    for target_weight, source_weight in zip(target_parameters, source_parameters):
        target_weight.mul_(mu).add_(source_weight, alpha=1 - mu)


@torch.no_grad()
def compute_anisotropy(embedding_weight):
    num_embeddings, _ = embedding_weight.shape
    num_pairs = (num_embeddings - 1) * num_embeddings
    norm_embeddings = torch.nn.functional.normalize(embedding_weight, dim=-1)
    cossim = norm_embeddings @ norm_embeddings.transpose(0, 1)
    return (cossim.sum() - num_embeddings) / num_pairs


def get_text(path: str) -> List[str]:
    with open(path, "r", encoding='utf-8') as file:
        return [line.strip() for line in file]


def get_line_offsets(path: str, chunk_size: int = 2 ** 20) -> List[int]:
    offsets = [0]
    with open(path, "rb") as file:
        chunk = file.readlines(chunk_size)
        while chunk:
            for line in chunk:
                offsets.append(offsets[-1] + len(line))
            chunk = file.readlines(chunk_size)
    return offsets


def linear_decay_with_warmup(
        step: int,
        max_learning_rate: float,
        warmup_steps: int,
        hold_steps: int,
        decay_steps: int,
        min_learning_rate: float = 1e-8
) -> float:
    scaled_lr = max_learning_rate * (step / warmup_steps)

    if step < warmup_steps:
        return scaled_lr
    elif step < warmup_steps + hold_steps:
        return max_learning_rate
    else:
        offset = warmup_steps + hold_steps
        return max(max_learning_rate * (1 - (step - offset) / decay_steps), min_learning_rate)


def cosine_decay_with_warmup(
        step: int,
        max_learning_rate: float,
        warmup_steps: int,
        decay_steps: int
) -> float:
    if step < warmup_steps:
        return max_learning_rate * step / warmup_steps

    step -= warmup_steps
    decay_fraction = (math.cos(step / decay_steps * math.pi) + 1) / 2
    return decay_fraction * max_learning_rate
