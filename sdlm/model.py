import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rope
from .utils import append_dims


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim=None, qk_norm=False):
        super(MultiHeadAttention, self).__init__()
        assert (dim % num_heads == 0)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else dim // num_heads
        self.qk_norm = qk_norm

        self.wq = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        if self.qk_norm:
            self.norm_q = nn.LayerNorm(self.head_dim)
            self.norm_k = nn.LayerNorm(self.head_dim)

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

        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_q(k)

        if freqs_cis is not None:
            q, k = rope.apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if mask is not None:
                mask = torch.where(mask, torch.tensor(0.0), torch.tensor(-1e4))
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # score = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # if mask is not None:
        #     score = score.masked_fill(mask == 0, -1e9)
        # score = F.softmax(score, dim=-1)
        # output = score @ v

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


class FiLM(nn.Module):
    def __init__(self, dim, film_dim=None):
        super(FiLM, self).__init__()
        self.dim = dim
        self.film_dim = film_dim or self.dim

        self.norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.wf = nn.Linear(self.film_dim, 2 * self.dim)

    def forward(self, x: torch.Tensor, film: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.wf(film).chunk(2, dim=-1)
        return (h * scale) + h


class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, film_dim, num_heads=8, dropout_prob=0.0):
        super(TransformerBlock, self).__init__()
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
            mask: Optional[torch.Tensor]
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
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = x @ self.weights.unsqueeze(0) * 2 * math.pi
        return torch.cat([x, freq.sin(), freq.cos()], dim=-1)


class SphereNorm(nn.Module):
    def __init__(self):
        super(SphereNorm, self).__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=-1) * math.sqrt(x.shape[-1])


class TokenAutoEncoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            dim: int = 1024,
            embedding_dim: int = 128
    ):
        super(TokenAutoEncoder, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)
        self.embedding_norm = SphereNorm()

        self.norm = nn.LayerNorm(self.dim)
        self.output = nn.Linear(self.dim, self.num_classes, bias=False)

    def encode(self, ids_or_weights: torch.Tensor) -> torch.Tensor:
        if ids_or_weights.dtype == torch.int64:
            return self.embedding_norm(self.embedding(ids_or_weights))
        elif ids_or_weights.dtype in (torch.float16, torch.float32, torch.float64):
            return ids_or_weights @ self.embedding_norm(self.embedding.weight)
        else:
            raise TypeError(f"Unsupported type {ids_or_weights.dtype}. Expected int64 or float types.")

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        output = self.output(h).float()
        return output

    def recode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.decode(x).softmax(-1)
        e = self.encode(w)
        return e, w


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
            nn.GELU()
        )
        self.encoder_layers = nn.ModuleList(
            TransformerBlock(
                dim=self.dim,
                hidden_dim=4 * self.dim,
                film_dim=128,
                num_heads=self.num_heads,
                dropout_prob=self.dropout_prob
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
            conditioning_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        t = append_dims(t, x.ndim)

        t_seq = torch.arange(seqlen, device=x.device).unsqueeze(0)
        freq_cis = rope.compute_freqs_cis(t_seq, self.dim // self.num_heads)

        if conditioning is not None and conditioning_mask is not None:
            x = torch.where(append_dims(conditioning_mask, x.ndim), conditioning, x)
            t = t.masked_fill(append_dims(conditioning_mask, t.ndim), 0.0)

        if attention_mask is None and length_mask is None:
            attention_mask = None
        elif attention_mask is None:
            attention_mask = self.self_attention_mask(length_mask)

        h, film = self.project(x), self.time_embed(t)

        for i, layer in enumerate(self.encoder_layers):
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

        self.autoencoder = TokenAutoEncoder(
            num_classes=self.num_classes,
            dim=self.model_dim
        )
        self.score_model = ScoreTransformer(
            model_dim=self.model_dim,
            input_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            fourier_dim=self.fourier_dim,
            dropout_prob=self.dropout_prob
        )

        self.apply(initialisation)

    def encode(self, ids_or_weights: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(ids_or_weights)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(x)

    def recode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.autoencoder.recode(x)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            length_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            conditioning: Optional[torch.Tensor] = None,
            conditioning_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.score_model(x, t, length_mask, attention_mask, conditioning, conditioning_mask)


def initialisation(module):
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(1)
        std = 1 / math.sqrt(fan_in)
        torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-std, b=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.001)
