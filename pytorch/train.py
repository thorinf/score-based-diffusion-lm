import os
import math
import random
import argparse
from typing import List, Tuple

import wandb
from tqdm import tqdm
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rotary_embedding_torch import RotaryEmbedding

torch.set_float32_matmul_precision('high')


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
            print(f"Lines found: {len(offsets):,}", end='\r')
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
        return self.sp.encode(text)

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
    def __init__(self, crop_length=-1, eos_id=-1, pad_id=-1, length_includes_pad=False, fold_size=None):
        assert not (pad_id < 0 and length_includes_pad)
        assert not (pad_id < 0 and fold_size)
        self.crop_length = crop_length
        self.fold_size = fold_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.pad_insert_rate = 0.0
        self.length_includes_pad = length_includes_pad

    def fold(self, ids):
        # pad the list for folding
        remainder = len(ids) % self.fold_size
        if remainder != 0:
            ids += [self.pad_id] * (self.fold_size - remainder)
        # fold the list
        ids = [ids[i:i + self.fold_size] for i in range(0, len(ids), self.fold_size)]
        return ids

    def generate_mask(self, length):
        conditional_mask = [False] * length
        mask_span_length = random.randint(0, length - 1)
        start_index = random.randint(0, length - mask_span_length)
        conditional_mask[start_index:start_index + mask_span_length] = [True] * mask_span_length
        # half of the masks will be completely random
        if random.random() < 0.5:
            random.shuffle(conditional_mask)
        return conditional_mask

    def process_ids(self, ids):
        # and the eos token
        if self.eos_id >= 0:
            ids.append(self.eos_id)
        # randomly insert pads into ids
        if self.pad_id >= 0 and self.pad_insert_rate > 0:
            pad_count = int(len(ids) * self.pad_insert_rate)
            pad_indices = random.sample(range(len(ids)), pad_count)
            for index in pad_indices:
                ids.insert(index, self.pad_id)
        if self.fold_size is not None:
            ids = self.fold(ids)
        # crops the length
        if 0 < self.crop_length < len(ids):
            ids = ids[:self.crop_length]
        # create a conditional mask
        conditional_mask = self.generate_mask(len(ids))
        return ids, len(ids), conditional_mask

    def __call__(self, batch):
        processed = list(map(self.process_ids, batch))
        ids, lengths, conditional_mask = zip(*processed)

        # sample a random amount of padding
        padded_lengths = [random.randint(length, max(lengths)) for length in lengths]
        lengths = torch.tensor(padded_lengths) if self.length_includes_pad else torch.tensor(lengths)

        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.int64) for x in ids],
            batch_first=True,
            padding_value=self.pad_id
        )
        conditional_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x, dtype=torch.bool) for x in conditional_mask],
            batch_first=True,
            padding_value=False
        )

        return ids, lengths, conditional_mask


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
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # with torch.backends.cuda.sdp_kernel(enable_flash=True):
        #     out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.1 if self.training else 0.0)

        score = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = F.softmax(score, dim=-1)
        out = score @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.model_dim)
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
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x):
        freq = torch.einsum('bl,d->bld', x, self.weights) * 2 * math.pi
        return torch.cat([x.unsqueeze(-1), freq.sin(), freq.cos()], dim=-1)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, target_dim, model_dim, num_layers=8, num_heads=16, learned_sinusoidal_dim=128,
                 dropout_prob=0.0, layerdrop_prob=0.0):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
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

    def forward(self, x, t, length_mask=None):
        batch_size, seq_length, _ = x.size()
        x = self.project(x) + self.time_mlp(t)

        if length_mask is not None:
            x = x * length_mask.unsqueeze(-1)
            attention_mask = self.self_attention_mask(length_mask)
        else:
            attention_mask = None

        for i, layer in enumerate(self.encoder_layers):
            if self.training and random.uniform(0, 1) < self.layerdrop_prob:
                continue
            x = layer(x, attention_mask)

        if length_mask is not None:
            x = x * length_mask.unsqueeze(-1)

        return self.out(x), x


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
        return (sigma ** 2.0 + self.sigma_data ** 2.0) / (sigma * self.sigma_data) ** 2.0

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        c_skip, c_out, c_in = self.get_scaling(sigmas)
        rescaled_t = 0.25 * torch.log(sigmas + 1e-3)
        model_output, latent = model(c_in.unsqueeze(-1) * x_t, rescaled_t, **model_kwargs)
        denoised = c_out.unsqueeze(-1) * model_output + c_skip.unsqueeze(-1) * x_t.detach()
        return model_output, denoised, latent

    def loss_t(self, x, t, **model_kwargs):
        x_target = x.detach()

        z = torch.randn_like(x, device=x.device)
        x_t = x + (z * t.unsqueeze(-1))

        model_output, x_denoised, latent = self.denoise(self.score_model, x_t, t, **model_kwargs)

        weights = self.loss_weight(t)

        return ((x_denoised - x_target) ** 2.0).mean(-1), x_denoised, latent, weights

    def compute_loss(self, x, conditional_mask, **model_kwargs):
        u = torch.rand(x.size()[:2], dtype=x.dtype, device=x.device, requires_grad=False)
        t = self.rho_schedule(u)
        t = t.masked_fill(conditional_mask, 0.0)

        return self.loss_t(x, t, **model_kwargs)

    @torch.no_grad()
    def sample_euler(self, x, ts, conditional_mask=None, return_list=False):
        if conditional_mask is None:
            conditional_mask = torch.zeros_like(x, dtype=torch.bool)

        x_list = []
        t = ts[0]

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

        return x_list if return_list else x

    @torch.no_grad()
    def sample_iterative(self, x, ts, conditional_mask=None, return_list=False):
        if conditional_mask is None:
            conditional_mask = torch.zeros_like(x, dtype=torch.bool)

        x_list = []

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

        return x_list if return_list else x


class DiffusionLM(nn.Module):
    def __init__(self, num_embeddings=1000, embedding_dim=64, model_dim=1024, num_layers=8, num_heads=16,
                 dropout_prob=0.1, layerdrop_prob=0.0, loss_weights=(1.0, 1.0)):
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
        self.diffusion = ScoreDiffusion(
            score_model=self.estimator,
            interpolate=self.interpolate,
        )

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

    def interpolate(self, x):
        logits = self.get_logits(x) / self.interpolate_temperature
        weights = logits.softmax(dim=-1)
        e = self.embedding.weight
        e = F.normalize(e, dim=-1) * math.sqrt(self.embedding_dim)
        interpolated = torch.einsum('nle,ed->nld', weights, e)
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

    def compute_loss(self, ids, lengths, conditional_mask):
        x = self.get_embeddings(ids)
        x = self.embedding_grad_scale * x + (1.0 - self.embedding_grad_scale) * x.detach()

        length_mask = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        diff_mask = torch.logical_and(length_mask, torch.logical_not(conditional_mask))
        num_elems = diff_mask.sum()

        loss_diff, x_denoised, latent, _ = self.diffusion.compute_loss(x, conditional_mask, length_mask=length_mask)

        loss_diff = loss_diff * diff_mask

        logits = self.get_logits(latent)
        ids = ids.masked_fill(~diff_mask, -100)
        loss_ce = self.loss_ce(logits.transpose(2, 1), ids)

        accuracy = (logits.argmax(dim=-1) == ids).float().sum() / num_elems

        loss_diff = loss_diff.sum() / num_elems
        loss_ce_pred = loss_ce.sum() / num_elems
        loss = self.loss_weights[0] * loss_diff + self.loss_weights[1] * loss_ce_pred

        return loss, loss_diff, loss_ce_pred, accuracy

    @torch.no_grad()
    def forward(self, z, num_steps=200, conditional_ids=None):
        ids = torch.zeros(z.shape[:2], dtype=torch.int64, device=z.device)
        conditional_mask = torch.zeros_like(ids, dtype=torch.bool)

        for i, sublist in enumerate(conditional_ids):
            sublist_len = len(sublist)
            ids[i, :sublist_len] = torch.tensor(sublist, device=z.device)
            conditional_mask[i, :sublist_len] = True

        z = torch.where(conditional_mask.unsqueeze(-1), self.get_embeddings(ids), z)

        u = torch.arange(num_steps, device=z.device).view(-1, 1, 1) / (num_steps - 1)
        ts = self.diffusion.rho_schedule(u)
        # x = self.diffusion.sample_euler(z * ts[0], ts, conditional_mask)
        x = self.diffusion.sample_iterative(z * ts[0], ts, conditional_mask)

        cossim = self.cosine_similarity(x)

        return cossim.argmax(dim=-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def linear_decay_with_warmup(step, max_learning_rate, warmup_steps, hold_steps, decay_steps, min_learning_rate=1e-8):
    scaled_lr = max_learning_rate * (step / warmup_steps)

    if step < warmup_steps:
        return scaled_lr
    elif step < warmup_steps + hold_steps:
        return max_learning_rate
    else:
        offset = warmup_steps + hold_steps
        return max(max_learning_rate * (1 - (step - offset) / decay_steps), min_learning_rate)


def cosine_decay_with_warmup(step, max_learning_rate, warmup_steps, decay_steps):
    if step < warmup_steps:
        return max_learning_rate * step / warmup_steps

    step -= warmup_steps
    decay_fraction = (math.cos(step / decay_steps * math.pi) + 1) / 2
    return decay_fraction * max_learning_rate


@torch.no_grad()
def eval_model(model, args, device, conditional_ids):
    model.eval()
    x_T = torch.randn((args.num_examples, args.crop_length, model.embedding_dim)).to(device)
    outputs = model(x_T, conditional_ids=conditional_ids).tolist()
    return outputs


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-decs', '--decay_steps', type=int, default=1e6)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=1)

    parser.add_argument('-edim', '--embedding_dim', type=int, default=128)
    parser.add_argument('-mdim', '--model_dim', type=int, default=1024)
    parser.add_argument('-numl', '--num_layers', type=int, default=8)
    parser.add_argument('-numh', '--num_heads', type=int, default=8)
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

    conditional_starts = [
        'this is a test',
        'ounce upon a time',
        'the king began thinking',
        'many people questioned the decisions of'
    ]
    conditional_ids = tokenizer.encode(conditional_starts)

    outputs = eval_model(model, args, device, conditional_ids)
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
        num_workers=4,
        pin_memory=False,
        collate_fn=collate
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    global_step = checkpoint.get('global_step', 0)
    lr_lambda = lambda step: cosine_decay_with_warmup(step, args.learning_rate, 10000, args.decay_steps)

    wandb.init(
        project="score-based-diffusion-lm",
        config={
            'num_embeddings': model.num_embeddings,
            'embedding_dim': model.embedding_dim,
            'model_dim': model.model_dim,
            'num_layers': model.num_layers,
            'dropout_prob': model.dropout_prob,
            'layerdrop_prob': model.layerdrop_prob,
            'loss_weights': model.loss_weights,
            'label_smoothing': model.label_smoothing
        }
    )
    wandb.watch(model, log_freq=100)

    num_params = count_parameters(model)
    formatted_params = "{:,}".format(num_params)
    print(f"Total number of parameters: {formatted_params}")

    for ep in range(0, args.epochs):
        model.train()
        pbar = tqdm(dataloader)
        pbar.set_description(f"epoch: {ep}")

        for idx, (ids, lengths, conditional_mask) in enumerate(pbar):

            ids, lengths, conditional_mask = ids.to(device), lengths.to(device), conditional_mask.to(device)

            loss, loss_diff, loss_ce, accuracy = model.compute_loss(ids, lengths, conditional_mask)

            (loss / args.accumulation_steps).backward()

            if ((idx + 1) % args.accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                optim.param_groups[0]['lr'] = lr_lambda(global_step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                global_step += 1

            metrics = {
                "loss": loss.item(),
                "mse": loss_diff.item(),
                "ce": loss_ce.item(),
                "accuracy": accuracy.item(),
            }
            pbar.set_postfix(metrics)

            if ((idx + 1) % args.accumulation_steps * 10 == 0) or (idx + 1 == len(dataloader)):
                metrics.update({"learning_rate": optim.param_groups[0]['lr']})
                metrics.update({"anisotropy": model.compute_anisotropy().item()})
                wandb.log(metrics, step=global_step)

            if ((idx + 1) % 500 == 0) or (idx + 1 == len(dataloader)):
                checkpoint = {
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }
                torch.save(checkpoint, args.checkpoint)

        outputs = eval_model(model, args, device, conditional_ids)
        [print(text) for text in tokenizer.decode(outputs)]

    wandb.finish()


if __name__ == "__main__":
    train()
