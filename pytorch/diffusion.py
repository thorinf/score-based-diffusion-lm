import random
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

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
            scale_output: bool = True,
            **model_kwargs: Any
    ) -> torch.Tensor:
        c_skip, c_out, c_in = self.get_scaling(sigmas)
        model_output = model(append_dims(c_in, x_t.ndim) * x_t, sigmas, **model_kwargs)
        if not scale_output:
            return model_output
        return append_dims(c_out, model_output.ndim) * model_output + append_dims(c_skip, x_t.ndim) * x_t.detach()

    def score_loss(
            self,
            model: nn.Module,
            x_target: torch.Tensor,
            **model_kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        u_shape = x_target.shape[:1] if random.uniform(0, 1) < 0.5 else x_target.shape[:2]
        u = torch.rand(u_shape, dtype=x_target.dtype, device=x_target.device, requires_grad=False)
        t = self.rho_schedule(u)

        z = torch.randn_like(x_target, device=x_target.device)
        x_t = x_target + (z * append_dims(t, z.ndim))

        model_output = self.denoise(model, x_t, t, **model_kwargs)

        weights = append_dims(self.loss_weight(t), x_target.ndim)
        mse = (x_target.detach() - model_output) ** 2.0

        terms = {}
        terms["xs_mse"] = mse.mean()
        terms["mse"] = (weights * mse).mean()
        terms["loss"] = terms["mse"]

        return terms

    def ce_score_loss(
            self,
            model: nn.Module,
            x_target: torch.Tensor,
            ids_target: torch.Tensor,
            ignore_index: int = -100,
            **model_kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        u_shape = x_target.shape[:1] if random.uniform(0, 1) < 0.5 else x_target.shape[:2]
        u = torch.rand(u_shape, dtype=x_target.dtype, device=x_target.device, requires_grad=False)
        t = self.rho_schedule(u)

        z = torch.randn_like(x_target, device=x_target.device)
        x_t = x_target + (z * append_dims(t, z.ndim))

        logits = self.denoise(model, x_t, t, False, **model_kwargs)

        loss = torch.nn.functional.cross_entropy(logits.transpose(1, -1), ids_target, reduction='none',
                                                 ignore_index=ignore_index)

        terms = {}
        terms["n_elem"] = (ids_target != ignore_index).sum()
        terms["ce"] = loss.sum() / terms["n_elem"]
        terms["wce"] = (loss * self.loss_weight(t)).sum() / terms["n_elem"]
        terms["loss"] = terms["wce"]

        return terms

    @torch.no_grad()
    def sample_euler(
            self,
            model: nn.Module,
            x_start: torch.Tensor,
            ts: torch.Tensor,
            interpolate: Callable = None,
            **model_kwargs: Any
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = x_start
        logits = None

        indices = range(len(ts))
        for i in indices:
            t, t_next = ts[i], ts[i + 1] if i + 1 != len(ts) else 0.0

            denoised = self.denoise(model, x, t, interpolate is None, **model_kwargs)

            if interpolate is not None:
                denoised, logits = interpolate(denoised), denoised

            d = (x - denoised) / append_dims(t, x.ndim)
            dt = append_dims(t_next - t, d.ndim)
            x = x + (d * dt)

        return logits if interpolate else x

    @torch.no_grad()
    def sample_edm(
            self,
            model: nn.Module,
            x_start: torch.Tensor,
            ts: torch.Tensor,
            interpolate: Callable = None,
            heun: bool = False,
            **model_kwargs: Any
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = x_start
        logits = None

        indices = range(len(ts))
        for i in indices:
            t, t_next = ts[i], ts[i + 1] if i + 1 != len(ts) else 0.0
            x_cur = x

            denoised = self.denoise(model, x_cur, t, interpolate is None, **model_kwargs)

            if interpolate is not None:
                denoised, logits = interpolate(denoised), denoised

            d = (x_cur - denoised) / t
            x = x_cur + (t_next - t) * d

            if i < len(ts) - 1 and heun:
                denoised = self.denoise(model, x, t_next, interpolate is None, **model_kwargs)

                if interpolate is not None:
                    denoised, logits = interpolate(denoised), denoised

                d_prime = (x - denoised) / t_next
                x = x_cur + (t_next - t) * (0.5 * d + 0.5 * d_prime)

        return logits if interpolate else x

    @torch.no_grad()
    def sample_iterative(
            self,
            model: nn.Module,
            x_start: torch.Tensor,
            ts: torch.Tensor,
            interpolate: Callable = None,
            **model_kwargs: Any
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = x_start
        logits = None

        indices = range(len(ts))
        for i in indices:
            t = ts[i]
            if i != 0:
                z = torch.randn_like(x)
                x = x + z * append_dims(t, z.ndim)

            x = self.denoise(model, x, t, interpolate is None, **model_kwargs)

            if interpolate is not None:
                x, logits = interpolate(x), x

        return logits if interpolate else x
