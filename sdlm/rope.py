from typing import Tuple

import torch


def compute_freqs_cis(t: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=t.device)[: (dim // 2)].float() / dim))
    freqs = (t.unsqueeze(-1) * freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(-2)
    q_out = torch.view_as_real(q_complex * freqs_cis).flatten(q.ndim - 1)
    k_out = torch.view_as_real(k_complex * freqs_cis).flatten(k.ndim - 1)
    return q_out.type_as(q), k_out.type_as(k)
