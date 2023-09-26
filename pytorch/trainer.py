import os
import os.path as osp
import time

import numpy as np
import torch
from unidecode import unidecode

from utils import (
    get_named_float_tensors,
    get_weight_decay_parameters,
    update_ema_parameters,
    compute_anisotropy,
    cosine_decay_with_warmup
)
import logger

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

        self.global_step = 0
        self.max_updates = 1e6

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)
        self.load_model_checkpoint()

        self.log_loss_scale = INITIAL_LOG_LOSS_SCALE

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
                pass
                # self.save()
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
        self.log_anisotropy()
        self.log_step()

    def forward_backward(self):
        self.opt.zero_grad()
        for _ in range(self.accumulation_steps):
            ids, length_mask, conditioning_mask = next(self.data)

            ids = ids.to(self.device)
            length_mask = length_mask.to(self.device)
            conditioning_mask = conditioning_mask.to(self.device)

            embeddings = self.model.get_embeddings(ids)
            loss_mask = torch.logical_and(length_mask, ~conditioning_mask)
            ids = ids.masked_fill(~loss_mask, -100)

            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                losses = self.diffusion.ce_score_loss(
                    model=self.model,
                    x_target=embeddings,
                    ids_target=ids,
                    length_mask=length_mask,
                    conditioning=embeddings,
                    conditioning_mask=conditioning_mask
                )

            loss = losses["loss"]

            n_elem = losses["n_elem"].item()
            logger.log_kv_mean("loss", loss.item(), n_elem)
            logger.log_kv_mean("ce", losses["ce"].item(), n_elem)
            logger.log_kv_mean("wce", losses["wce"].item(), n_elem)

            n_token, n_mask = length_mask.sum().item(), conditioning_mask.sum().item()
            logger.log_kv_mean("n_token", n_token)
            logger.log_kv_mean("p_mask", n_mask / n_token, n_token)

            if self.use_fp16:
                loss_scale = 2 ** self.log_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimise(self):
        n_loss_elem = getattr(self, "n_loss_elem", self.accumulation_steps)
        if hasattr(self, "n_loss_elem"):
            setattr(self, "n_loss_elem", 0)

        if n_loss_elem == 0:
            logger.error("tried to optimize, but number of elements in loss was 0")
            return

        self.scale_grads(1.0 / n_loss_elem)
        self.log_grad_norm()
        self.update_lr()
        if self.gradient_clipping > 0:
            self.grad_clip()
        self.opt.step()
        self.global_step += 1

    def optimize_fp16(self):
        n_loss_elem = getattr(self, "n_loss_elem", self.accumulation_steps)
        if hasattr(self, "n_loss_elem"):
            setattr(self, "n_loss_elem", 0)

        if n_loss_elem == 0:
            logger.log("tried to optimize, but number of elements in loss was 0")
            return

        if not self.grad_is_finite():
            self.log_loss_scale -= 1
            logger.log(f"found NaN, decreased log_loss_scale to {self.log_loss_scale}")
            return

        self.scale_grads(1.0 / (n_loss_elem * (2 ** self.log_loss_scale)))
        self.log_grad_norm()
        self.update_lr()
        if self.gradient_clipping > 0:
            self.grad_clip()
        self.opt.step()
        self.log_loss_scale += self.fp16_scale_growth
        self.global_step += 1

    def grad_is_finite(self):
        for param in self.model.parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    return False
        return True

    def scale_grads(self, scale):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(scale)

    def grad_clip(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

    def log_grad_norm(self):
        sq_sum = 0.0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            sq_sum += (p.grad ** 2).sum().item()
        logger.log_kv_mean("grad_norm", np.sqrt(sq_sum))

    def update_lr(self):
        lr = cosine_decay_with_warmup(self.global_step, self.learning_rate, int(self.warmup_steps),
                                      int(self.max_updates))
        [param_group.update({"lr": lr}) for param_group in self.opt.param_groups]
        logger.log_kv("lr", lr)

    def log_anisotropy(self):
        logger.log_kv("anisotropy", compute_anisotropy(self.model.embedding.weight).item())

    def log_step(self):
        logger.log_kv("step", self.global_step)
        if self.use_fp16:
            logger.log_kv("log_loss_scale", self.log_loss_scale)

    def prepare_conditioning(self):
        conditioning_ids = torch.zeros(self.sample_size, dtype=torch.int64, device=self.device)
        conditioning_mask = torch.zeros_like(conditioning_ids, dtype=torch.bool)

        if self.sample_conditioning:
            for i, text in enumerate(self.sample_conditioning):
                sample_conditioning = self.tokenizer.encode(text, bos=True, eos=False)
                sublist_len = len(sample_conditioning)
                conditioning_ids[i, :sublist_len] = torch.tensor(sample_conditioning, device=self.device)
                conditioning_mask[i, :sublist_len] = True

        return conditioning_ids, conditioning_mask

    @torch.inference_mode()
    def sample(self):
        self.model.eval()
        conditioning_ids, conditioning_mask = self.prepare_conditioning()
        conditioning = self.model.get_embeddings(conditioning_ids)

        z = torch.randn((*conditioning_mask.size(), self.model.embedding_dim), device=self.device)
        us = torch.arange(self.sample_iterations, device=z.device) / (self.sample_iterations - 1)
        ts = self.diffusion.rho_schedule(us.unsqueeze(-1))

        logger.info(f"sampling started...")
        logits = self.diffusion.sample_euler(
            model=self.model,
            x_start=z * ts[0],
            ts=ts,
            interpolate=self.model.interpolate,
            conditioning=conditioning,
            conditioning_mask=conditioning_mask,
        )
        self.model.train()

        output_ids = torch.where(conditioning_mask, conditioning_ids, logits.argmax(-1)).cpu().tolist()
        decoded = self.tokenizer.decode(output_ids)
        [logger.info(f"sample {i}:\t{unidecode(text)}") for i, text in enumerate(decoded)]
