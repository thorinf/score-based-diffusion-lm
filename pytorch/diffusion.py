import random

import torch

from utils import append_dims


class ScoreDiffusion:
    def __init__(self, score_model, interpolate=None, masking=None, sigma_min=1.0, sigma_max=10.0, sigma_data=1.0,
                 rho=1.0):
        super(ScoreDiffusion).__init__()
        self.score_model = score_model
        self.interpolate = interpolate
        self.masking = masking

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

    def rho_schedule(self, u):
        # u [0,1], linear schedule when rho is 1.0
        rho_inv = 1.0 / self.rho
        sigma_max_pow_rho_inv = self.sigma_max ** rho_inv
        sigmas = (sigma_max_pow_rho_inv + u * (self.sigma_min ** rho_inv - sigma_max_pow_rho_inv)) ** self.rho
        return sigmas

    @staticmethod
    def get_snr(sigma):
        return sigma ** -2.0

    def get_weights(self, snr):
        return snr + 1.0 / self.sigma_data ** 2.0

    def get_scaling(self, sigma, epsilon=0.0):
        c_skip = (self.sigma_data ** 2) / (((sigma - epsilon) ** 2) + self.sigma_data ** 2)
        c_out = ((sigma - epsilon) * self.sigma_data) / ((sigma ** 2 + self.sigma_data ** 2) ** 0.5)
        c_in = 1 / ((sigma ** 2 + self.sigma_data ** 2) ** 0.5)
        return c_skip, c_out, c_in

    def loss_weight(self, sigma):
        return (sigma ** 2.0 + self.sigma_data ** 2.0) / (sigma * self.sigma_data) ** 2.0

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        c_skip, c_out, c_in = self.get_scaling(sigmas)
        rescaled_t = 0.25 * torch.log(sigmas + 1e-3)
        model_output, latent = model(append_dims(c_in, x_t.ndim) * x_t, rescaled_t, **model_kwargs)
        denoised = append_dims(c_out, model_output.ndim) * model_output + append_dims(c_skip, x_t.ndim) * x_t.detach()
        return model_output, denoised, latent

    def loss_t(self, x, t, **model_kwargs):
        x_target = x.detach()

        z = torch.randn_like(x, device=x.device)
        x_t = x + (z * t.unsqueeze(-1))

        model_output, x_denoised, latent = self.denoise(self.score_model, x_t, t, **model_kwargs)

        weights = self.loss_weight(t)

        return ((x_denoised - x_target) ** 2.0).mean(-1), x_denoised, latent, weights

    def compute_loss(self, x, conditional_mask, **model_kwargs):
        if random.uniform(0, 1) < 0.5:
            u = torch.rand(x.size()[:2], dtype=x.dtype, device=x.device, requires_grad=False)
        else:
            u = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device, requires_grad=False)
        t = self.rho_schedule(u)
        t = t.masked_fill(conditional_mask, 0.0)

        return self.loss_t(x, t, **model_kwargs)

    @torch.no_grad()
    def sample_euler(self, x, ts, conditional_mask=None, return_list=False):
        if conditional_mask is None:
            conditional_mask = torch.zeros_like(x, dtype=torch.bool)

        x_list = []
        t = ts[0]
        logits = None

        for i in range(len(ts)):
            t = t.masked_fill(conditional_mask, 0.0)

            _, denoised, latent = self.denoise(self.score_model, x, t)

            if self.interpolate is not None:
                denoised, logits = self.interpolate(latent)

            t_next = ts[i + 1] if i + 1 != len(ts) else 0.0
            d = (x - denoised) / t.unsqueeze(-1)
            dt = (t_next - t).unsqueeze(-1)
            x = x + (d * dt).masked_fill(conditional_mask.unsqueeze(-1), 0.0)

            if return_list:
                x_list.append(x)

            t = t_next

        return x_list if return_list else logits if self.interpolate else x

    @torch.no_grad()
    def sample_iterative(self, x, ts, conditional_mask=None, return_list=False):
        if conditional_mask is None:
            conditional_mask = torch.zeros_like(x, dtype=torch.bool)

        x_list = []
        logits = None

        for i, t in enumerate(ts):
            t = t.masked_fill(conditional_mask, 0.0)

            if i != 0:
                z = torch.randn_like(x)
                x = x + t.unsqueeze(-1) * z

            _, x, latent = self.denoise(self.score_model, x, t)

            if self.interpolate is not None:
                x, logits = self.interpolate(latent)

            x = torch.where(conditional_mask.unsqueeze(-1), x, x)

            if return_list:
                x_list.append(x)

        return x_list if return_list else logits if self.interpolate else x
