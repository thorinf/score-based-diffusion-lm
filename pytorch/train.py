import os
import math
import random
import argparse
from typing import List

import wandb
from tqdm import tqdm
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rotary_embedding_torch import RotaryEmbedding


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
            print(f"Lines found: {len(offsets)}", end='\r')
            chunk = file.readlines(chunk_size)
    return offsets


class SentencePieceTokenizer:
    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    def __len__(self):
        return len(self.sp)

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def pad_id(self):
        return self.sp.pad_id()

    def encode(self, text):
        return self.sp.encode(text, enable_sampling=True, alpha=0.1, nbest_size=2)

    def decode(self, encoded):
        return self.sp.decode(encoded)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer: SentencePieceTokenizer):
        self.path = path
        self.tokenizer = tokenizer
        self.offsets = get_line_offsets(path)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with open(self.path, 'r', encoding='utf-8') as file:
            file.seek(self.offsets[idx])
            text = file.readline().strip()
        ids = self.tokenizer.encode(text)
        return ids


class Collate:
    def __init__(self, crop_length=-1, eos_id=-1, pad_id=-1, length_includes_pad=False):
        assert not (pad_id < 0 and length_includes_pad)
        self.crop_length = crop_length
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.length_includes_pad = length_includes_pad

    def __call__(self, batch):
        ids_end = [self.eos_id] if self.eos_id >= 0 else []
        ids_list = [torch.tensor(ids + ids_end, dtype=torch.int64) for ids in batch]
        ids = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=self.pad_id)
        lengths = torch.tensor([x.shape[0] for x in ids_list])

        if 0 < self.crop_length < ids.shape[1]:
            ids = ids[:, :self.crop_length]
            lengths = torch.minimum(lengths, torch.tensor(self.crop_length))

        if self.length_includes_pad:
            lengths = torch.full_like(lengths, lengths.max())

        return ids, lengths


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, rotary_embedding=None):
        super(MultiHeadAttention, self).__init__()
        assert (dim % num_heads == 0)
        self.model_dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_o = nn.Linear(dim, dim)

        self.rotary_emb = rotary_embedding

    def forward(self, q, k, v, mask=None):
        batch_size, seq_length, _ = q.size()

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=2)

        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = F.softmax(score, dim=-1)
        out = score @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.model_dim)
        return self.w_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, drop_prob=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=False,
            rotary_embedding=RotaryEmbedding(dim=dim // (num_heads * 2))
        )
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask, gammas=(0.0, 0.0), betas=(0.0, 0.0)):
        res1 = x
        x = self.norm1(x)
        x = (gammas[0] * x) + betas[0]
        x = self.attention(q=x, k=x, v=x, mask=mask)
        x = res1 + self.dropout1(x)

        res2 = x
        x = self.norm2(x)
        x = (gammas[1] * x) + betas[1]
        x = self.ffn(x)
        x = res2 + self.dropout2(x)
        return x


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(LearnedSinusoidalPosEmb, self).__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        freq = torch.einsum('b,d->bd', x, self.weights) * 2 * math.pi
        return torch.cat([freq.sin(), freq.cos()], dim=-1)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, target_dim, model_dim, num_layers=8, learned_sinusoidal_dim=128, dropout_prob=0.0,
                 layerdrop_prob=0.0):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layerdrop_prob = layerdrop_prob

        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(learned_sinusoidal_dim),
            nn.Linear(learned_sinusoidal_dim, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, num_layers * 4 * model_dim),
            nn.GELU(),
        )

        self.project = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.Dropout(p=dropout_prob)
        )

        self.encoder_layers = nn.ModuleList(
            TransformerEncoderLayer(
                dim=model_dim,
                hidden_dim=4 * model_dim,
                num_heads=8,
                drop_prob=dropout_prob
            )
            for _ in range(num_layers))

        self.out = nn.Linear(model_dim, target_dim)

    def forward(self, x, t, length_mask=None):
        time_emb = self.time_mlp(t)
        x = self.project(x)

        if length_mask is not None:
            x = x * length_mask.unsqueeze(-1)
            length_mask = length_mask.unsqueeze(1).unsqueeze(1)

        scaling_weights = time_emb.view(-1, self.num_layers * 4, self.model_dim).split(1, dim=1)
        for i, layer in enumerate(self.encoder_layers):
            if self.training and random.uniform(0, 1) < self.layerdrop_prob:
                continue
            gammas = scaling_weights[4 * i], scaling_weights[4 * i + 1]
            betas = scaling_weights[4 * i + 2], scaling_weights[4 * i + 3]
            x = layer(x, length_mask, gammas=gammas, betas=betas)

        return self.out(x)


class ScoreDiffusion:
    def __init__(self, score_model, interpolate=None, masking=None, sigma_min=0.1, sigma_max=100.0, sigma_data=1.0,
                 rho=7.0):
        super(ScoreDiffusion).__init__()
        self.score_model = score_model
        self.interpolate = interpolate
        self.masking = masking

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

    def rho_schedule(self, u):
        # u [0,1]
        rho_inv = 1.0 / self.rho
        sigma_max_pow_rho_inv = self.sigma_max ** rho_inv
        sigmas = (sigma_max_pow_rho_inv + u * (self.sigma_min ** rho_inv - sigma_max_pow_rho_inv)) ** self.rho
        return sigmas

    def get_snr(self, sigma):
        return sigma ** -2.0

    def get_weights(self, snr):
        return snr + 1.0 / self.sigma_data ** 2.0

    def get_scaling(self, sigma, epsilon=0.0):
        c_skip = (self.sigma_data ** 2) / (((sigma - epsilon) ** 2) + self.sigma_data ** 2)
        c_out = (sigma - epsilon) * self.sigma_data / ((sigma ** 2 + self.sigma_data ** 2) ** 0.5)
        c_in = 1 / ((sigma ** 2 + self.sigma_data ** 2) ** 0.5)
        return c_skip, c_out, c_in

    def loss_weight(self, sigma):
        return (sigma ** 2.0 + self.sigma_data ** 2.0) * torch.rsqrt(sigma * self.sigma_data)

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        c_skip, c_out, c_in = self.get_scaling(sigmas)
        rescaled_t = 0.25 * torch.log(sigmas + 1e-44)
        model_output = model(c_in.view(-1, 1, 1) * x_t, rescaled_t, **model_kwargs)
        denoised = c_out.view(-1, 1, 1) * model_output + c_skip.view(-1, 1, 1) * x_t
        return model_output, denoised

    def loss_t(self, x, t, **model_kwargs):
        z = torch.randn_like(x, device=x.device)

        target_x = x

        if self.masking is not None:
            x = self.masking(x)

        x_t = x + z * t.view(-1, 1, 1)

        _, denoised_x = self.denoise(self.score_model, x_t, t.detach(), **model_kwargs)

        snrs = self.get_snr(t)
        weights = self.get_weights(snrs)

        return ((denoised_x - target_x) ** 2.0).mean(-1) * weights.view(-1, 1), denoised_x, weights

    def compute_loss(self, x, **model_kwargs):
        rand_u = torch.rand((x.shape[0],), dtype=x.dtype, device=x.device, requires_grad=False)
        t = self.rho_schedule(rand_u)

        return self.loss_t(x, t, **model_kwargs)

    @torch.no_grad()
    def forward_diffusion(self, x, ts, return_list=False):
        x_list = []

        _, x = self.denoise(self.score_model, x, ts[0].unsqueeze(0))

        if return_list:
            x_list.append(x)

        for t in ts[1:]:
            if self.interpolate is not None:
                x = self.interpolate(x)

            t = t.unsqueeze(0)
            z = torch.randn_like(x)
            x = x + t * z
            _, x = self.denoise(self.score_model, x, t)

            if return_list:
                x_list.append(x)

        return x_list if return_list else x


class DiffusionLM(nn.Module):
    def __init__(self, num_embeddings=1000, embedding_dim=64, model_dim=512, num_layers=8, dropout_prob=0.2,
                 layerdrop_prob=0.0, loss_weights=(1.0, 1.0)):
        super(DiffusionLM, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.layerdrop_prob = layerdrop_prob
        self.loss_weights = loss_weights

        self.embedding_grad_scale = 0.1
        self.interpolate_temperature = 1.0

        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim
        )
        nn.init.normal_(self.embedding.weight, std=0.1)

        self.estimator = TransformerModel(
            input_dim=self.embedding_dim,
            target_dim=self.embedding_dim,
            model_dim=self.model_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            layerdrop_prob=layerdrop_prob
        )
        self.diffusion = ScoreDiffusion(
            score_model=self.estimator,
        )

        self.dropout = nn.Dropout(p=dropout_prob)
        self.lm_head = nn.Linear(self.embedding_dim, self.num_embeddings)

        self.loss_ce = nn.CrossEntropyLoss(reduction='none')

    def get_embeddings(self, ids):
        e = self.embedding(ids)
        e = F.normalize(e, dim=-1) * math.sqrt(self.embedding_dim)
        return e

    def get_logits(self, x):
        x = self.dropout(x)
        x = self.lm_head(x)
        return x

    def interpolate(self, x):
        logits = self.get_logits(x) / self.interpolate_temperature
        weights = logits.softmax(dim=-1)
        e = self.embedding.weight
        e = F.normalize(e, dim=-1) * math.sqrt(self.embedding_dim)
        interpolated = torch.einsum('nle,ed->nld', weights, e)
        return interpolated

    def apply_mask(self, x):
        batch_size, seq_length, _ = x.size()
        token_mask = torch.rand(batch_size, seq_length, 1, device=x.device) < 0.15
        return torch.where(token_mask, torch.randn_like(x), x)

    def cosine_similarity(self, x):
        e = self.embedding.weight
        e = F.normalize(e, dim=-1)
        x = F.normalize(x, dim=-1)
        cossim = torch.einsum('nld,ed->nle', x, e)
        return cossim

    @torch.no_grad()
    def compute_anisotropy(self):
        num_pairs = (self.num_embeddings - 1) * self.num_embeddings
        norm_embeddings = F.normalize(self.embedding.weight, dim=-1)
        return (torch.einsum('id,jd->ij', norm_embeddings, norm_embeddings).sum() - self.num_embeddings) / num_pairs

    def embedding_grad_norm(self):
        return torch.norm(self.embedding.weight.grad)

    def compute_loss(self, ids, lengths):
        x = self.get_embeddings(ids)
        x = self.embedding_grad_scale * x + (1.0 - self.embedding_grad_scale) * x.detach()

        len_mask = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        num_elems = len_mask.sum()

        loss_diff, denoised_x, weights = self.diffusion.compute_loss(x, length_mask=len_mask)
        loss_diff = loss_diff[len_mask].mean(-1)

        logits = self.get_logits(denoised_x)
        ids = ids.masked_fill(torch.logical_not(len_mask), -100)
        loss_reconstruction = self.loss_ce(logits.transpose(2, 1), ids) * weights.view(-1, 1)

        accuracy = (logits.argmax(dim=-1) == ids).float().sum() / num_elems

        loss_diff = loss_diff.mean()
        loss_reconstruction = loss_reconstruction.sum() / num_elems
        loss = self.loss_weights[0] * loss_diff + self.loss_weights[1] * loss_reconstruction

        return loss, loss_diff, loss_reconstruction, accuracy

    @torch.no_grad()
    def forward(self, z, num_steps=100):
        ts = self.diffusion.rho_schedule(torch.arange(num_steps, device=z.device) / num_steps)
        x = self.diffusion.forward_diffusion(z * ts[0], ts)
        return self.get_logits(x).argmax(dim=-1)


def linear_decay_with_warmup(step, max_learning_rate, warmup_steps, hold_steps, decay_steps, min_learning_rate=1e-8):
    if step < warmup_steps:
        return max_learning_rate * (step / warmup_steps)
    elif step < warmup_steps + hold_steps:
        return max_learning_rate
    else:
        offset = warmup_steps + hold_steps
        scale = 1 - (step - offset) / decay_steps
        return max(max_learning_rate * scale, min_learning_rate)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-decs', '--decay_steps', type=int, default=800000)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-8)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=2)

    parser.add_argument('-edim', '--embedding_dim', type=int, default=64)
    parser.add_argument('-mdim', '--model_dim', type=int, default=512)
    parser.add_argument('-numl', '--num_layers', type=int, default=8)
    parser.add_argument('-do', '--dropout_prob', type=float, default=0.1)
    parser.add_argument('-ld', '--layerdrop_prob', type=float, default=0.0)

    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-spm', '--spm_model', type=str, required=True)
    parser.add_argument('-cl', '--crop_length', type=int, default=64)
    parser.add_argument('-ngen', '--num_examples', type=int, default=8)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tokenizer = SentencePieceTokenizer(args.spm_model)

    model = DiffusionLM(
        num_embeddings=len(tokenizer),
        embedding_dim=args.embedding_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        dropout_prob=args.dropout_prob,
        layerdrop_prob=args.layerdrop_prob,
        loss_weights=(1.0, 0.1)
    )
    model.to(device)

    if os.path.exists(args.checkpoint):
        print(f"Restoring Checkpoint: {args.checkpoint}.")
        checkpoint = torch.load(args.checkpoint)
    else:
        print(f"Starting new training run: {args.checkpoint}.")
        checkpoint = {}

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()
    with torch.no_grad():
        x_T = torch.randn((args.num_examples, args.crop_length, model.embedding_dim)).to(device)
        outputs = model(x_T).tolist()
        [print(text) for text in tokenizer.decode(outputs)]

    dataset = TextDataset(path=args.data_path, tokenizer=tokenizer)
    collate = Collate(
        crop_length=args.crop_length,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        length_includes_pad=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay
    )

    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    global_step = checkpoint.get('global_step', 0)
    lr_lambda = lambda step: linear_decay_with_warmup(step, args.learning_rate, 2000, 0, args.decay_steps,
                                                      args.learning_rate * 0.1)

    wandb.init(
        project="score-based-diffusion-lm",
        config={
            'num_embeddings': model.num_embeddings,
            'embedding_dim': model.embedding_dim,
            'model_dim': model.model_dim,
            'num_layers': model.num_layers,
            'dropout_prob': model.dropout_prob,
            'layerdrop_prob': model.layerdrop_prob,
            'loss_weights': model.loss_weights
        }
    )

    for ep in range(0, args.epochs):
        model.train()
        pbar = tqdm(dataloader)
        pbar.set_description(f"epoch: {ep}")

        for idx, (ids, lengths) in enumerate(pbar):
            ids = ids.to(device)
            lengths = lengths.to(device)

            loss, loss_diff, loss_reconstruction, accuracy = model.compute_loss(ids, lengths)

            if torch.isfinite(loss):
                (loss / args.accumulation_steps).backward()
            else:
                ValueError("Loss is not finite, backward pass not computed.")

            metrics = {
                "loss": loss.item(),
                "mse": loss_diff.item(),
                "ce": loss_reconstruction.item(),
                "accuracy": accuracy.item(),
            }

            pbar.set_postfix(metrics)

            if ((idx + 1) % args.accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                optim.param_groups[0]['lr'] = lr_lambda(global_step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                torch.cuda.empty_cache()
                global_step += 1

                metrics.update({"learning_rate": optim.param_groups[0]['lr']})
                wandb.log(metrics, step=global_step)

            if ((idx + 1) % 500 == 0) or (idx + 1 == len(dataloader)):
                checkpoint = {
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }
                torch.save(checkpoint, args.checkpoint)

        model.eval()
        with torch.no_grad():
            x_T = torch.randn((args.num_examples, args.crop_length, model.embedding_dim)).to(device)
            outputs = model(x_T).tolist()
            [print(text) for text in tokenizer.decode(outputs)]

    wandb.finish()


if __name__ == "__main__":
    train()
