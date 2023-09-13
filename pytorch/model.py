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

        nn.init.xavier_uniform_(self.w_q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_k.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_v.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_o.weight)

    def forward(self, q, k, v, mask=None):
        bsz, seqlen, _ = q.shape

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.1 if self.training else 0.0)

        # score = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # if mask is not None:
        #     score = score.masked_fill(mask == 0, -1e9)
        # score = F.softmax(score, dim=-1)
        # output = score @ v

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.w_o(output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, emb_dim, num_heads=8, drop_prob=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.emb1 = nn.Sequential(
            nn.Dropout(p=drop_prob),
            nn.Linear(emb_dim, 2 * dim),
        )
        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            rotary_embedding=RotaryEmbedding(dim=int(dim / num_heads))
        )
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.emb2 = nn.Sequential(
            nn.Dropout(p=drop_prob),
            nn.Linear(emb_dim, 2 * dim),
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, emb, mask=None):
        h = self.norm1(x)
        scale, shift = self.emb1(emb).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        h = self.attention(q=h, k=h, v=h, mask=mask)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        scale, shift = self.emb2(emb).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        h = self.ffn(h)
        x = x + self.dropout2(h)
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
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.001)

        self.project = nn.Linear(self.embedding_dim, self.model_dim)

        self.time_embed = nn.Sequential(
            LearnedSinusoidalPosEmb(learned_sinusoidal_dim),
            nn.Linear(self.learned_sinusoidal_dim + 1, 128),
            nn.GELU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(128, 128),
            nn.GELU()
        )

        self.encoder_layers = nn.ModuleList(
            TransformerEncoderLayer(
                dim=self.model_dim,
                hidden_dim=4 * self.model_dim,
                emb_dim=128,
                num_heads=self.num_heads,
                drop_prob=self.dropout_prob
            )
            for _ in range(num_layers))

        self.output = nn.Linear(self.model_dim, self.num_classes)

    def get_embeddings(self, ids):
        e = self.embedding(ids)
        e = F.normalize(e, dim=-1) * math.sqrt(self.embedding_dim)
        return e

    def get_logits(self, x):
        h = self.norm(x)
        output = self.output(h)
        return output

    def interpolate(self, logits):
        emb_weights = (logits / self.interpolate_temperature).softmax(dim=-1)
        norm_emb = F.normalize(self.embedding.weight, dim=-1) * math.sqrt(self.embedding_dim)
        return emb_weights @ norm_emb

    @staticmethod
    def self_attention_mask(length_mask):
        bsz, seqlen = length_mask.shape
        mask = torch.zeros(bsz, seqlen, seqlen, dtype=torch.bool, device=length_mask.device)
        mask = mask.masked_fill(length_mask.unsqueeze(1), True)
        return mask.unsqueeze(1)

    def forward(self, x, t, length_mask=None, conditioning=None, conditioning_mask=None):
        bsz, seqlen, _ = x.shape

        t = append_dims(t, x.ndim)

        if conditioning is not None and conditioning_mask is not None:
            x = torch.where(append_dims(conditioning_mask, x.ndim), conditioning, x)
            t = t.masked_fill(append_dims(conditioning_mask, t.ndim), 0.0)

        if length_mask is None:
            length_mask = torch.ones((bsz, seqlen), dtype=torch.bool, device=x.device)

        attention_mask = self.self_attention_mask(length_mask)

        emb = self.time_embed(append_dims(t, x.ndim))
        h = self.project(x)

        for i, layer in enumerate(self.encoder_layers):
            if self.training and random.uniform(0, 1) < self.layerdrop_prob:
                continue
            h = layer(h, emb=emb, mask=attention_mask)

        output = self.output(h).float()
        return output
