import random
from typing import Any, Callable, List, Tuple, TypeVar, Union

import torch
import torch.nn as nn

from utils import append_dims

T = TypeVar('T', float, torch.Tensor)


class MultiStepScoreDiffusion:
    def __init__(
            self,
            sigma_min: float = 1.0,
            sigma_max: float = 10.0,
            sigma_data: float = 1.0,
            rho: float = 1.0,
    ) -> None:
        super(MultiStepScoreDiffusion).__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

    def rho_schedule(self, u: T) -> T:
        # u [0,1], linear schedule when rho is 1.0
        rho_inv = 1.0 / self.rho
        sigma_max_pow_rho_inv = self.sigma_max ** rho_inv
        sigmas = (sigma_max_pow_rho_inv + u * (self.sigma_min ** rho_inv - sigma_max_pow_rho_inv)) ** self.rho
        return sigmas

    @staticmethod
    def get_snr(sigma: T) -> T:
        return sigma ** -2.0

    def get_weights(self, snr: T) -> T:
        return snr + 1.0 / self.sigma_data ** 2.0

    def get_scaling(self, sigma: T, epsilon: float = 0.0) -> Tuple[T, T, T]:
        c_skip = (self.sigma_data ** 2) / (((sigma - epsilon) ** 2) + self.sigma_data ** 2)
        c_out = ((sigma - epsilon) * self.sigma_data) / ((sigma ** 2 + self.sigma_data ** 2) ** 0.5)
        c_in = 1 / ((sigma ** 2 + self.sigma_data ** 2) ** 0.5)
        return c_skip, c_out, c_in

    def loss_weight(self, sigma: T) -> T:
        return (sigma ** 2.0 + self.sigma_data ** 2.0) / (sigma * self.sigma_data) ** 2.0

    def denoise(
            self,
            model: nn.Module,
            x_t: torch.Tensor,
            sigmas: torch.Tensor,
            **model_kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip, c_out, c_in = self.get_scaling(sigmas)
        rescaled_t = 0.25 * torch.log(sigmas + 1e-3)
        model_output, latent = model(append_dims(c_in, x_t.ndim) * x_t, rescaled_t, **model_kwargs)
        denoised = append_dims(c_out, model_output.ndim) * model_output + append_dims(c_skip, x_t.ndim) * x_t.detach()
        return model_output, denoised, latent

    def loss_t(
            self,
            model: nn.Module,
            x_target: torch.Tensor,
            t: torch.Tensor,
            **model_kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_target, x = x_target.detach(), x_target

        z = torch.randn_like(x, device=x.device)
        x_t = x + (z * append_dims(t, z.ndim))

        model_output, x_denoised, latent = self.denoise(model, x_t, t, **model_kwargs)

        weights = self.loss_weight(t)

        return ((x_denoised - x_target) ** 2.0).mean(-1), x_denoised, latent, weights

    def compute_loss(
            self,
            model: nn.Module,
            x_target: torch.Tensor,
            conditional_mask: torch.Tensor = None,
            **model_kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if conditional_mask is None:
            conditional_mask = torch.zeros(x_target.shape[:2], dtype=torch.bool)
        else:
            assert conditional_mask.shape == x_target.shape[:2], \
                f"Mask shape {conditional_mask.shape} mismatch with first 2 dimensions of x_target {x_target.shape}"

        if random.uniform(0, 1) < 0.5:
            u = torch.rand(x_target.size()[:2], dtype=x_target.dtype, device=x_target.device, requires_grad=False)
        else:
            u = torch.rand((x_target.shape[0], 1), dtype=x_target.dtype, device=x_target.device, requires_grad=False)

        t = self.rho_schedule(u)
        t = t.masked_fill(conditional_mask, 0.0)

        return self.loss_t(model, x_target, t, **model_kwargs)

    @torch.no_grad()
    def sample_euler(
            self,
            model: nn.Module,
            x_start: torch.Tensor,
            ts: torch.Tensor,
            conditional_mask: torch.Tensor = None,
            interpolate: Callable = None,
            return_all: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if conditional_mask is None:
            conditional_mask = torch.zeros(x_start.shape[:2], dtype=torch.bool)
        else:
            assert conditional_mask.shape == x_start.shape[:2], \
                f"Mask shape {conditional_mask.shape} mismatch with first 2 dimensions of x_start {x_start.shape}"

        x = x_start
        x_list = []
        t = ts[0]
        logits = None

        for i in range(len(ts)):
            t = t.masked_fill(conditional_mask, 0.0)

            _, denoised, latent = self.denoise(model, x, t)

            if interpolate is not None:
                denoised, logits = interpolate(latent)

            t_next = ts[i + 1] if i + 1 != len(ts) else 0.0
            d = (x - denoised) / append_dims(t, x.ndim)
            dt = append_dims(t_next - t, d.ndim)
            x = x + (d * dt).masked_fill(append_dims(conditional_mask, d.ndim), 0.0)

            if return_all:
                x_list.append(x)

            t = t_next

        return x_list if return_all else logits if interpolate else x

    @torch.no_grad()
    def sample_iterative(
            self,
            model: nn.Module,
            x_start: torch.Tensor,
            ts: torch.Tensor,
            conditional_mask: torch.Tensor = None,
            interpolate: Callable = None,
            return_all: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if conditional_mask is None:
            conditional_mask = torch.zeros(x_start.shape[:2], dtype=torch.bool)
        else:
            assert conditional_mask.shape == x_start.shape[:2], \
                f"Mask shape {conditional_mask.shape} mismatch with first 2 dimensions of x_start {x_start.shape}"

        x = x_start
        x_list = []
        logits = None

        for i, t in enumerate(ts):
            t = t.masked_fill(conditional_mask, 0.0)

            if i != 0:
                z = torch.randn_like(x)
                x = x + z * append_dims(t, z.ndim)

            _, x, latent = self.denoise(model, x, t)

            if interpolate is not None:
                x, logits = interpolate(latent)

            x = torch.where(append_dims(conditional_mask, x.ndim), x, x)

            if return_all:
                x_list.append(x)

        return x_list if return_all else logits if interpolate else x
