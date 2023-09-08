import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding

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


class ScoreLM(nn.Module):
    def __init__(
            self,
            num_classes: int,
            model_dim: int = 1024,
            embedding_dim: int = 1024,
            num_layers: int = 8,
            num_heads: int = 16,
            learned_sinusoidal_dim: int = 128,
            dropout_prob: float = 0.0,
            layerdrop_prob: float = 0.0
    ):
        super(ScoreLM, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.learned_sinusoidal_dim = learned_sinusoidal_dim
        self.dropout_prob = dropout_prob
        self.layerdrop_prob = layerdrop_prob
        self.interpolate_temperature = 1.0

        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)

        self.project = nn.Sequential(
            nn.Linear(self.embedding_dim, self.model_dim, bias=True),
            nn.Dropout(p=self.dropout_prob)
        )

        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(learned_sinusoidal_dim),
            nn.Linear(self.learned_sinusoidal_dim + 1, 128),
            nn.GELU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(128, self.model_dim),
        )

        self.encoder_layers = nn.ModuleList(
            TransformerEncoderLayer(
                dim=self.model_dim,
                hidden_dim=4 * self.model_dim,
                num_heads=self.num_heads,
                drop_prob=self.dropout_prob,
                elementwise_affine=True
            )
            for _ in range(num_layers))

        self.norm = nn.LayerNorm(self.model_dim)

        self.output = nn.Linear(self.model_dim, self.num_classes, bias=False)

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
        x = self.norm(x)
        x = self.output(x)
        return x

    def interpolate(self, logits):
        emb_weights = (logits / self.interpolate_temperature).softmax(dim=-1)
        norm_emb = F.normalize(self.embedding.weight, dim=-1) * math.sqrt(self.embedding_dim)
        return emb_weights @ norm_emb

    @staticmethod
    def self_attention_mask(length_mask):
        return torch.logical_and(length_mask.unsqueeze(1).unsqueeze(1), length_mask.unsqueeze(1).unsqueeze(-1))

    def forward(self, x, t, length_mask=None, conditioning=None, conditioning_mask=None):
        bsz, slen, _ = x.shape

        t = append_dims(t, x.ndim)

        if conditioning is not None and conditioning_mask is not None:
            x = torch.where(append_dims(conditioning_mask, x.ndim), conditioning, x)
            t = t.masked_fill(append_dims(conditioning_mask, t.ndim), 0.0)

        x = self.project(x) + self.time_mlp(append_dims(t, x.ndim))

        if length_mask is None:
            length_mask = torch.ones((bsz, slen), dtype=torch.bool, device=x.device)

        attention_mask = self.self_attention_mask(length_mask)

        for i, layer in enumerate(self.encoder_layers):
            if self.training and random.uniform(0, 1) < self.layerdrop_prob:
                continue
            x = layer(x, attention_mask)

        x = self.norm(x)
        x = self.output(x)

        return x
