from typing import List
import math


def append_dims(tensor, target_dims):
    assert isinstance(target_dims, int), f"Expected 'target_dims' to be an integer, but received {type(target_dims)}."
    tensor_dims = tensor.ndim
    assert tensor_dims <= target_dims, f"Tensor has {tensor_dims} dimensions, but target has {target_dims} dimensions."
    return tensor[(...,) + (None,) * (target_dims - tensor_dims)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_text(path: str) -> str:
    with open(path, "r", encoding='utf-8') as file:
        return file.read()


def get_line_offsets(path: str, chunk_size: int = 2 ** 20) -> List[int]:
    offsets = [0]
    with open(path, "rb") as file:
        chunk = file.readlines(chunk_size)
        while chunk:
            for line in chunk:
                offsets.append(offsets[-1] + len(line))
            print(f"Lines found: {len(offsets):,}", end='\r')
            chunk = file.readlines(chunk_size)
    return offsets


def linear_decay_with_warmup(step, max_learning_rate, warmup_steps, hold_steps, decay_steps, min_learning_rate=1e-8):
    scaled_lr = max_learning_rate * (step / warmup_steps)

    if step < warmup_steps:
        return scaled_lr
    elif step < warmup_steps + hold_steps:
        return max_learning_rate
    else:
        offset = warmup_steps + hold_steps
        return max(max_learning_rate * (1 - (step - offset) / decay_steps), min_learning_rate)


def cosine_decay_with_warmup(step, max_learning_rate, warmup_steps, decay_steps):
    if step < warmup_steps:
        return max_learning_rate * step / warmup_steps

    step -= warmup_steps
    decay_fraction = (math.cos(step / decay_steps * math.pi) + 1) / 2
    return decay_fraction * max_learning_rate