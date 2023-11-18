import os.path as osp

import numpy as np
import torch
from unidecode import unidecode

from . import logger
from .utils import (
    get_named_float_tensors,
    get_weight_decay_parameters,
    update_ema_parameters,
    compute_anisotropy,
    cosine_decay_with_warmup
)

INITIAL_LOG_LOSS_SCALE = 20.0


class Trainer:
    def __init__(
            self,
            model,
            diffusion,
            tokenizer,
            data,
            batch_size,
            accumulation_steps,
            learning_rate,
            ema_rate,
            model_dir,
            log_interval,
            save_interval,
            sample_interval,
            sample_size,
            sample_conditioning,
            sample_iterations,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            warmup_steps=1e5,
            weight_decay=0.0,
            gradient_clipping=-1.0
    ):
        self.model = model
        self.diffusion = diffusion
        self.tokenizer = tokenizer
        self.data = data
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.learning_rate = learning_rate
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        self.model_dir = model_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.sample_interval = sample_interval
        self.sample_size = sample_size
        self.sample_conditioning = sample_conditioning
        self.sample_iterations = sample_iterations
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
        self.max_updates = 1e6
        self.global_step = 0
        self.log_loss_scale = INITIAL_LOG_LOSS_SCALE

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)
        self.load_model_checkpoint()

        decay, no_decay = get_weight_decay_parameters(self.model)
        optim_groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        self.opt = torch.optim.AdamW(optim_groups, lr=self.learning_rate)
        self.load_optim_checkpoint()

        self.ema_named_tensors = []
        self.load_ema_checkpoints()

    def load_model_checkpoint(self):
        checkpoint_path = osp.join(self.model_dir, "model.pt")
        if osp.exists(checkpoint_path):
            logger.info(f"loading checkpoint from {checkpoint_path}")
            model_state_dict = torch.load(checkpoint_path)
            self.model.load_state_dict(model_state_dict, strict=False)

    def load_optim_checkpoint(self):
        checkpoint_path = osp.join(self.model_dir, "optim.pt")
        if osp.exists(checkpoint_path):
            logger.info(f"loading checkpoint from {checkpoint_path}")
            optim_state_dict = torch.load(checkpoint_path)
            self.global_step = optim_state_dict.pop('global_step', self.global_step)
            self.opt.load_state_dict(optim_state_dict)

    def load_ema_checkpoints(self):
        all_named_tensors = get_named_float_tensors(self.model, include_buffers=True)
        for rate in self.ema_rate:
            ema_checkpoint_path = osp.join(self.model_dir, f"ema_{rate}.pt")
            if osp.exists(ema_checkpoint_path):
                logger.info(f"loading EMA checkpoint: {ema_checkpoint_path}")
                ema_state_dict = torch.load(ema_checkpoint_path)
                # Update or get tensors from the EMA state dict using model tensors names
                ema_named_tensors = [(key, ema_state_dict[key]) for key, value in all_named_tensors]
            else:
                logger.info(f"initializing new EMA model with EMA rate of {rate}")
                # Initialize EMA tensors with model's named tensors
                ema_named_tensors = [(key, value.data.clone()) for key, value in all_named_tensors]
            self.ema_named_tensors.append(ema_named_tensors)

    def save(self):
        for ema_named_tensors, rate in zip(self.ema_named_tensors, self.ema_rate):
            ema_state_dict = self.model.state_dict()
            for name, tensor in ema_named_tensors:
                ema_state_dict[name] = tensor
            checkpoint_path = osp.join(self.model_dir, f"ema_{rate}.pt")
            logger.info(f"saving checkpoint: {checkpoint_path}")
            torch.save(ema_state_dict, checkpoint_path)

        model_state_dict = self.model.state_dict()
        checkpoint_path = osp.join(self.model_dir, "model.pt")
        logger.info(f"saving checkpoint: {checkpoint_path}")
        torch.save(model_state_dict, checkpoint_path)

        optim_state_dict = self.opt.state_dict()
        optim_state_dict['global_step'] = self.global_step
        checkpoint_path = osp.join(self.model_dir, "optim.pt")
        logger.info(f"saving checkpoint: {checkpoint_path}")
        torch.save(optim_state_dict, checkpoint_path)

    def update_ema_parameters(self):
        model_state_dict = self.model.state_dict()
        for ema_named_tensors, rate in zip(self.ema_named_tensors, self.ema_rate):
            ema_parameters_list, model_parameters_list = zip(*[
                (ema_parameter, model_state_dict[key])
                for key, ema_parameter in ema_named_tensors
            ])

            update_ema_parameters(ema_parameters_list, model_parameters_list, rate)

    def run_loop(self):
        logger.info(f"training loop started...")
        while True:
            self.run_step()
            if self.global_step % self.log_interval == 0:
                logger.dump_kvs()
            if self.global_step % self.save_interval == 0:
                self.save()
            if self.global_step % self.sample_interval == 0:
                self.sample()
                logger.info(f"resuming training...")
            if self.max_updates is not None and self.global_step >= self.max_updates:
                logger.info(f"training completed {self.max_updates} maximum steps...")
                break

    def run_step(self):
        self.forward_backward()
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimise()
        self.update_ema_parameters()
        self._log_anisotropy()
        self._log_step()

    def forward_backward(self):
        self.opt.zero_grad()
        for _ in range(self.accumulation_steps):
            ids, length_mask, attention_mask, conditioning_mask = [tensor.to(self.device) for tensor in next(self.data)]

            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                encoded = self.model.encode(ids)

                output = self.diffusion.forward_reverse(
                    model=self.model,
                    x_target=encoded,
                    length_mask=length_mask,
                    attention_mask=attention_mask,
                    conditioning=encoded,
                    conditioning_mask=conditioning_mask
                )

                logits = self.model.decode(output)

                loss_mask = torch.logical_and(length_mask, ~conditioning_mask)
                ids = ids.masked_fill(~loss_mask, -100)
                n_elem = loss_mask.sum()

                ce = torch.nn.functional.cross_entropy(logits.transpose(1, -1), ids)
                z_loss = 1e-4 * torch.square(output.logsumexp(dim=-1) * loss_mask).sum() / n_elem
                loss = ce

                accuracy = torch.eq(logits.argmax(-1), ids).float().sum() / n_elem

            if self.use_fp16:
                loss_scale = 2 ** self.log_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

            # Losses
            n_elem = n_elem.item()
            logger.log_kv_mean("loss", loss.item(), n_elem)
            logger.log_kv_mean("ce", ce.item(), n_elem)
            logger.log_kv_mean("z_loss", z_loss.item(), n_elem)

            # Model Metrics
            logits_mean = (logits.mean(-1) * loss_mask).sum() / n_elem
            logger.log_kv_mean("logits_mean", logits_mean.item(), n_elem)
            logger.log_kv_mean("accuracy", accuracy.item(), n_elem)

            # Performance Metrics
            n_token, n_mask = length_mask.sum().item(), conditioning_mask.sum().item()
            logger.log_kv_mean("n_token", n_token)
            logger.log_kv_mean("p_mask", n_mask / n_token, n_token)

    def optimise(self):
        n_loss_elem = self._get_n_loss_elem()

        if n_loss_elem == 0:
            logger.error("tried to optimize, but number of elements in loss was 0")
            return

        self._scale_grads(1.0 / n_loss_elem)
        self._log_norms()
        self._anneal_lr()
        if self.gradient_clipping > 0:
            self._grad_clip()
        self.opt.step()
        self.global_step += 1

    def optimize_fp16(self):
        if not self._grad_is_finite():
            # In a DistributedDataParallel (DDP) setting, gradients are synchronized,
            # so scaling (and consequently weight updates) should be consistent across all ranks.
            self.log_loss_scale -= 1
            logger.log(f"found NaN, decreased log_loss_scale to {self.log_loss_scale}")
            return

        n_loss_elem = self._get_n_loss_elem()
        if n_loss_elem == 0:
            logger.log("tried to optimize, but number of elements in loss was 0")
            return

        self._scale_grads(1.0 / (n_loss_elem * (2 ** self.log_loss_scale)))
        self._log_norms()
        self._anneal_lr()
        if self.gradient_clipping > 0:
            self._grad_clip()
        self.opt.step()
        self.log_loss_scale += self.fp16_scale_growth
        self.global_step += 1

    def _grad_is_finite(self):
        for p in self.model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    return False
        return True

    def _scale_grads(self, scale):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    def _grad_clip(self):
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers have a specific way to do gradient clipping.
            self.opt.clip_grad_norm(self.gradient_clipping)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

    def _get_n_loss_elem(self):
        # The number of 'loss elements' can optionally be set during forward-backward.
        # If used, the loss for computing the backward pass should be the sum of the loss, not the mean.
        # Scaling the gradients by this value later is typically equivalent to taking the mean.
        # Specifying the number of loss elements is useful when various elements contribute to the overall loss,
        # preventing smaller batches from disproportionately affecting weight updates.
        # For scenarios with multiple accumulation steps, sum the number loss elements across iterations.
        n_loss_elem = torch.tensor(getattr(self, "n_loss_elem", self.accumulation_steps)).float()
        if hasattr(self, "n_loss_elem"):
            setattr(self, "n_loss_elem", 0)

        return n_loss_elem

    def _log_norms(self):
        grad_sq_sum = 0.0
        param_sq_sum = 0.0
        for p in self.model.parameters():
            param_sq_sum += (p ** 2).sum().item()
            if p.grad is not None:
                grad_sq_sum += (p.grad ** 2).sum().item()
        logger.log_kv_mean("grad_norm", np.sqrt(grad_sq_sum))
        logger.log_kv_mean("param_norm", np.sqrt(param_sq_sum))

    def _anneal_lr(self):
        lr = cosine_decay_with_warmup(self.global_step, self.learning_rate, int(self.warmup_steps),
                                      int(self.max_updates))
        [param_group.update({"lr": lr}) for param_group in self.opt.param_groups]
        logger.log_kv("lr", lr)

    def _log_anisotropy(self):
        logger.log_kv("anisotropy", compute_anisotropy(self.model.autoencoder.embedding.weight).item())

    def _log_step(self):
        logger.log_kv("step", self.global_step)
        if self.use_fp16:
            logger.log_kv("log_loss_scale", self.log_loss_scale)

    @torch.inference_mode()
    def sample(self):
        self.model.eval()
        conditioning_ids, conditioning_mask = self._prepare_conditioning()
        conditioning = self.model.encode(conditioning_ids)

        z = torch.randn((*conditioning_mask.size(), self.model.embedding_dim), device=self.device)
        us = torch.arange(self.sample_iterations, device=z.device) / (self.sample_iterations - 1)
        ts = self.diffusion.rho_schedule(us.unsqueeze(-1))

        logger.info(f"sampling started...")
        _, probs = self.diffusion.sample_euler(
            model=self.model,
            x_start=z * ts[0],
            ts=ts,
            postprocessing=self.model.recode,
            conditioning=conditioning,
            conditioning_mask=conditioning_mask,
        )
        self.model.train()

        output_ids = torch.where(conditioning_mask, conditioning_ids, probs.argmax(-1)).cpu().tolist()
        decoded = self.tokenizer.decode(output_ids)
        [logger.info(f"sample {i}:\t{unidecode(text)}") for i, text in enumerate(decoded)]

    def _prepare_conditioning(self):
        conditioning_ids = torch.zeros(self.sample_size, dtype=torch.int64, device=self.device)
        conditioning_mask = torch.zeros_like(conditioning_ids, dtype=torch.bool)

        if self.sample_conditioning:
            for i, text in enumerate(self.sample_conditioning):
                sample_conditioning = self.tokenizer.encode(text, bos=True, eos=False)
                sublist_len = len(sample_conditioning)
                conditioning_ids[i, :sublist_len] = torch.tensor(sample_conditioning, device=self.device)
                conditioning_mask[i, :sublist_len] = True

        return conditioning_ids, conditioning_mask
