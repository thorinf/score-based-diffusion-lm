import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding

from diffusion import MultiStepScoreDiffusion
from utils import append_dims


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
        bsz, slen, _ = q.shape

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(bsz, slen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, slen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, slen, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # with torch.backends.cuda.sdp_kernel(enable_flash=True):
        #     out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.1 if self.training else 0.0)

        score = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = F.softmax(score, dim=-1)
        out = score @ v

        out = out.transpose(1, 2).contiguous().view(bsz, slen, self.model_dim)
        return self.w_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, drop_prob=0.0, elementwise_affine=True):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)
        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            rotary_embedding=RotaryEmbedding(dim=int(dim / (num_heads * 2)))
        )
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask):
        res = x
        x = self.norm1(x)
        x = self.attention(q=x, k=x, v=x, mask=mask)
        x = res + self.dropout1(x)

        res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = res + self.dropout2(x)
        return x


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(LearnedSinusoidalPosEmb, self).__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        freq = x @ self.weights.unsqueeze(0) * 2 * math.pi
        return torch.cat([x, freq.sin(), freq.cos()], dim=-1)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, target_dim, model_dim, num_layers=8, num_heads=16, learned_sinusoidal_dim=128,
                 dropout_prob=0.0, layerdrop_prob=0.0):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layerdrop_prob = layerdrop_prob

        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(learned_sinusoidal_dim),
            nn.Linear(learned_sinusoidal_dim + 1, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, model_dim),
        )

        self.project = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(model_dim, model_dim)
        )

        self.encoder_layers = nn.ModuleList(
            TransformerEncoderLayer(
                dim=model_dim,
                hidden_dim=4 * model_dim,
                num_heads=num_heads,
                drop_prob=dropout_prob,
                elementwise_affine=True
            )
            for _ in range(num_layers))

        self.out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(model_dim, target_dim)
        )

    @staticmethod
    def self_attention_mask(length_mask):
        return torch.logical_and(length_mask.unsqueeze(1).unsqueeze(1), length_mask.unsqueeze(1).unsqueeze(-1))

    def mixed_masking(self, length_mask, num_heads):
        # A mix of forward and backward masking, intended to combat RoPE symmetry
        mask = self.self_attention_mask(length_mask)
        mask_lower = torch.tril(mask).repeat(1, num_heads // 2, 1, 1)
        mask_upper = torch.triu(mask).repeat(1, num_heads // 2, 1, 1)
        return torch.concat([mask_lower, mask_upper], dim=1)

    def forward(self, x, t, length_mask=None):
        bsz, slen, _ = x.shape
        x = self.project(x) + self.time_mlp(append_dims(t, x.ndim))

        if length_mask is None:
            length_mask = torch.ones((bsz, slen), dtype=torch.bool, device=x.device)

        x = x * append_dims(length_mask, x.ndim)
        attention_mask = self.mixed_masking(length_mask, self.num_heads)

        for i, layer in enumerate(self.encoder_layers):
            if self.training and random.uniform(0, 1) < self.layerdrop_prob:
                continue
            x = layer(x, attention_mask)

        x = x * append_dims(length_mask, x.ndim)

        return self.out(x), x


class DiffusionLM(nn.Module):
    def __init__(self, num_embeddings=1000, embedding_dim=64, model_dim=1024, num_layers=8, num_heads=16,
                 dropout_prob=0.1, layerdrop_prob=0.0, loss_weights=(0.0, 1.0)):
        super(DiffusionLM, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.layerdrop_prob = layerdrop_prob
        self.loss_weights = loss_weights

        self.embedding_grad_scale = 1.0
        self.interpolate_temperature = 1.0
        self.label_smoothing = 0.0

        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

        self.estimator = TransformerModel(
            input_dim=self.embedding_dim,
            target_dim=self.embedding_dim,
            model_dim=self.model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            layerdrop_prob=layerdrop_prob
        )
        self.diffusion = MultiStepScoreDiffusion(sigma_min=1.0, sigma_max=10.0, sigma_data=1.0, rho=1.0)

        self.lm_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(model_dim, self.num_embeddings)
        )

        self.loss_ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=self.label_smoothing)

        self.apply(self.initialise_weights)

    @staticmethod
    def initialise_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.001)

    def get_embeddings(self, ids):
        e = self.embedding(ids)
        e = F.normalize(e, dim=-1) * math.sqrt(self.embedding_dim)
        return e

    def get_logits(self, x):
        x = self.lm_head(x)
        return x

    def interpolate(self, x, hard=False):
        logits = self.get_logits(x) / self.interpolate_temperature
        weights = logits.softmax(dim=-1)
        e = self.embedding.weight
        e = F.normalize(e, dim=-1) * math.sqrt(self.embedding_dim)
        interpolated = torch.einsum('nle,ed->nld', weights, e)
        if hard:
            return interpolated, logits, self.get_embeddings(logits.argmax(-1))
        return interpolated, logits

    def cosine_similarity(self, x):
        e = self.embedding.weight
        e = F.normalize(e, dim=-1)
        x = F.normalize(x, dim=-1)
        cossim = torch.einsum('nld,ed->nle', x, e)
        return cossim

    @torch.no_grad()
    def compute_anisotropy(self):
        num_pairs = (self.num_embeddings - 1) * self.num_embeddings
        e = self.embedding.weight
        norm_embeddings = F.normalize(e, dim=-1)
        return (torch.einsum('id,jd->ij', norm_embeddings, norm_embeddings).sum() - self.num_embeddings) / num_pairs

    def embedding_grad_norm(self):
        return torch.norm(self.embedding.weight.grad)

    def compute_loss(self, ids, length_mask, conditional_mask):
        x = self.get_embeddings(ids)
        x = self.embedding_grad_scale * x + (1.0 - self.embedding_grad_scale) * x.detach()

        diff_mask = torch.logical_and(length_mask, torch.logical_not(conditional_mask))
        num_elems = diff_mask.sum()

        loss_diff, _, latent, weights = self.diffusion.compute_loss(
            model=self.estimator,
            x_target=x,
            conditional_mask=conditional_mask,
            length_mask=length_mask
        )

        weights = weights.masked_fill(~diff_mask, 0.0)

        loss_diff = loss_diff * weights

        logits = self.get_logits(latent)
        ids = ids.masked_fill(~diff_mask, -100)
        loss_ce = self.loss_ce(logits.transpose(2, 1), ids)

        loss_ce = loss_ce * weights

        accuracy = (logits.argmax(dim=-1) == ids).float().sum() / num_elems

        loss_diff = loss_diff.sum() / num_elems
        loss_ce_pred = loss_ce.sum() / num_elems
        loss = self.loss_weights[0] * loss_diff + self.loss_weights[1] * loss_ce_pred

        return loss, loss_diff, loss_ce_pred, accuracy

    @torch.no_grad()
    def forward(self, z, num_steps=200, conditional_ids=None, final_id=None):
        ids = torch.zeros(z.shape[:2], dtype=torch.int64, device=z.device)
        conditional_mask = torch.zeros_like(ids, dtype=torch.bool)

        if conditional_ids is not None:
            for i, sublist in enumerate(conditional_ids):
                sublist_len = len(sublist)
                ids[i, :sublist_len] = torch.tensor(sublist, device=z.device)
                conditional_mask[i, :sublist_len] = True

        if final_id is not None:
            ids[:, -1] = final_id
            conditional_mask[:, -1] = True

        # Set the conditional embeddings to be the true embeddings
        z = torch.where(append_dims(conditional_mask, z.ndim), self.get_embeddings(ids), z)

        u = torch.arange(num_steps, device=z.device) / (num_steps - 1)
        u = append_dims(u, z.ndim)
        ts = self.diffusion.rho_schedule(u)

        # logits = self.diffusion.sample_euler(
        #     model=self.estimator,
        #     x_start=z * ts[0],
        #     ts=ts,
        #     conditional_mask=conditional_mask,
        #     interpolate=self.interpolate
        # )

        logits = self.diffusion.sample_iterative(
            model=self.estimator,
            x_start=z * ts[0],
            ts=ts,
            conditional_mask=conditional_mask,
            interpolate=self.interpolate
        )

        return logits.argmax(dim=-1)
