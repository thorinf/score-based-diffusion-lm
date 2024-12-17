import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rope
from .utils import append_dims


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FiLM(nn.Module):
    def __init__(self, dim: int, film_dim: int = None, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.wf = nn.Linear(film_dim or dim, 2 * dim)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, film: torch.Tensor) -> torch.Tensor:
        h = self._norm(x.float()).type_as(x)
        scale, shift = self.wf(film).chunk(2, dim=-1)
        return (h * scale) + shift


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads

        self.wq = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = q.shape

        q, k, v = self.wq(q), self.wk(k), self.wv(v)

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim)

        if freqs_cis is not None:
            q, k = rope.apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, dropout_prob=0.0):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(dim=dim, num_heads=num_heads)
        self.ffn = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.fnn_norm = RMSNorm(dim)
        self.attention_dropout = nn.Dropout(p=dropout_prob)
        self.fnn_dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.attention_norm(x)
        h = self.attention(h, h, h, freq_cis, mask)
        x = x + self.attention_dropout(h)

        h = self.fnn_norm(x)
        h = self.ffn(h)
        x = x + self.fnn_dropout(h)
        return x


class FiLMTransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, film_dim, num_heads=8, dropout_prob=0.0):
        super(FiLMTransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(dim=dim, num_heads=num_heads)
        self.ffn = FeedForward(dim, hidden_dim)
        self.attention_film = FiLM(dim=dim, film_dim=film_dim)
        self.fnn_film = FiLM(dim=dim, film_dim=film_dim)
        self.attention_dropout = nn.Dropout(p=dropout_prob)
        self.fnn_dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        film: torch.Tensor,
        freq_cis: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.attention_film(x, film)
        h = self.attention(h, h, h, freq_cis, mask)
        x = x + self.attention_dropout(h)

        h = self.fnn_film(x, film)
        h = self.ffn(h)
        x = x + self.fnn_dropout(h)
        return x


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(LearnedSinusoidalPosEmb, self).__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = x @ self.weights.unsqueeze(0) * 2 * math.pi
        return torch.cat([x, freq.sin(), freq.cos()], dim=-1)


class ScoreTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 1024,
        num_layers: int = 8,
        num_heads: int = 16,
        fourier_dim: int = 128,
        dropout_prob: float = 0.0,
    ):
        super(ScoreTransformer, self).__init__()
        self.input_dim = input_dim
        self.dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.fourier_dim = fourier_dim
        self.dropout_prob = dropout_prob

        self.project = nn.Linear(self.input_dim, self.dim, bias=False)

        self.time_embed = nn.Sequential(
            LearnedSinusoidalPosEmb(fourier_dim),
            nn.Linear(self.fourier_dim + 1, 128),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.encoder_layers = nn.ModuleList(
            FiLMTransformerBlock(
                dim=self.dim,
                hidden_dim=4 * self.dim,
                film_dim=128,
                num_heads=self.num_heads,
                dropout_prob=self.dropout_prob,
            )
            for _ in range(num_layers)
        )

    def self_attention_mask(self, length_mask):
        bsz, seqlen = length_mask.shape
        mask = torch.logical_and(length_mask.view(bsz, 1, 1, seqlen), length_mask.view(bsz, 1, seqlen, 1))
        return mask.expand(bsz, self.num_heads, seqlen, seqlen)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        length_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
        conditioning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        if attention_mask is None and length_mask is not None:
            attention_mask = self.self_attention_mask(length_mask)
        if attention_mask is not None:
            eye = torch.eye(seqlen, dtype=torch.bool, device=attention_mask.device)
            attention_mask = attention_mask | eye.view(1, 1, seqlen, seqlen)

        t = append_dims(t, x.ndim)
        t_seq = torch.arange(seqlen, device=x.device).unsqueeze(0)
        freq_cis = rope.compute_freqs_cis(t_seq, self.dim // self.num_heads, theta=1000)

        film = self.time_embed(t)

        if conditioning is not None and conditioning_mask is not None:
            c_mask = append_dims(conditioning_mask, x.ndim)
            x = torch.where(c_mask, conditioning, x)
            film = film.masked_fill(c_mask, 0.0)

        h = self.project(x)

        for layer in self.encoder_layers:
            h = layer(h, film, freq_cis, attention_mask)

        return h


class ScoreLM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_dim: int = 1024,
        embedding_dim: int = 1024,
        num_layers: int = 8,
        num_heads: int = 16,
        fourier_dim: int = 128,
        dropout_prob: float = 0.0,
    ):
        super(ScoreLM, self).__init__()
        self.num_classes = num_classes
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.fourier_dim = fourier_dim
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.001)

        self.score_model = ScoreTransformer(
            model_dim=self.model_dim,
            input_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            fourier_dim=self.fourier_dim,
            dropout_prob=self.dropout_prob,
        )

        self.norm = RMSNorm(self.dim)
        self.output = nn.Linear(self.dim, self.num_classes, bias=False)

    def encode(self, ids_or_weights: torch.Tensor) -> torch.Tensor:
        if ids_or_weights.dtype == torch.int64:
            e = self.embedding(ids_or_weights)
            return torch.nn.functional.normalize(e, dim=-1) * math.sqrt(e.shape[-1])
        elif ids_or_weights.dtype in (torch.float16, torch.float32, torch.float64):
            e_w = self.embedding.weight
            return ids_or_weights @ torch.nn.functional.normalize(e_w, dim=-1) * math.sqrt(e_w.shape[-1])
        else:
            raise TypeError(f"Unsupported type {ids_or_weights.dtype}. Expected int64 or float types.")

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.output(h).float()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        length_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
        conditioning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.score_model(
            x,
            t,
            length_mask,
            attention_mask,
            conditioning,
            conditioning_mask,
        )
