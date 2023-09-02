import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
import wandb

from data import SentencePieceTokenizer, TextDataset, Collate
from model import DiffusionLM
from utils import count_parameters, get_text, update_model_ema, cosine_decay_with_warmup
from monitoring import get_initialised_logger


@torch.no_grad()
def sample_model(model, size, device, conditional_ids=None, final_id=None):
    model.eval()
    x_start = torch.randn((*size, model.embedding_dim), device=device)
    outputs = model(x_start, conditional_ids=conditional_ids, final_id=final_id).tolist()
    return outputs


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-bsz', '--batch_size', type=int, default=128)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=1)
    parser.add_argument('-svi', '--log_interval', type=int, default=100)
    parser.add_argument('-lgi', '--save_interval', type=int, default=1000)
    parser.add_argument('-smi', '--sample_interval', type=int, default=10000)

    parser.add_argument('-edim', '--embedding_dim', type=int, default=128)
    parser.add_argument('-mdim', '--model_dim', type=int, default=1024)
    parser.add_argument('-nl', '--num_layers', type=int, default=8)
    parser.add_argument('-nh', '--num_heads', type=int, default=8)
    parser.add_argument('-dop', '--dropout_prob', type=float, default=0.1)
    parser.add_argument('-ldp', '--layerdrop_prob', type=float, default=0.0)

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wus', '--warmup_steps', type=int, default=1e5)
    parser.add_argument('-dcs', '--decay_steps', type=int, default=1e6)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.1)
    parser.add_argument('-ema', '--ema_momentum', type=float, default=0.9999)

    parser.add_argument('-slen', '--sequence_length', type=int, default=64)
    parser.add_argument('-nex', '--num_examples', type=int, default=8)
    parser.add_argument('-b', '--beta', type=float, default=3.0)

    parser.add_argument('-mdir', '--model_dir', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-spm', '--spm_model', type=str, required=True)

    args = parser.parse_args()

    checkpoint_path = os.path.join(args.model_dir, "checkpoint.pt")
    logfile_path = os.path.join(args.model_dir, "logfile.log")

    logger = get_initialised_logger(logfile_path=logfile_path)

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

    if os.path.exists(checkpoint_path):
        logger.info(f"Reloading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
    else:
        logger.info(f"Starting new checkpoint: {checkpoint_path}")
        checkpoint = {}

    if 'model_state_dict' in checkpoint:
        logger.info(f"Model state found in checkpoint")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    num_params = count_parameters(model)
    logger.info(f"Total number of parameters: {num_params:,}")

    ema_model = DiffusionLM(
        num_embeddings=len(tokenizer),
        embedding_dim=args.embedding_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        dropout_prob=args.dropout_prob,
        layerdrop_prob=args.layerdrop_prob,
    )
    ema_model.to(device)

    if 'ema_model_state_dict' in checkpoint:
        logger.info(f"EMA model state found in checkpoint")
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        logger.info(f"Initialising EMA model with model state")
        ema_model.load_state_dict(model.state_dict())

    conditional_starts = get_text("conditional_starts.txt")
    conditional_ids = tokenizer.encode(conditional_starts)

    output_ids = sample_model(
        model=ema_model,
        size=(8, args.sequence_length),
        device=device,
        conditional_ids=conditional_ids,
        final_id=tokenizer.eos_id
    )
    decoded = tokenizer.decode(output_ids)
    [logger.info(f"Sample {i}:\t{text}") for i, text in enumerate(decoded)]

    dataset = TextDataset(path=args.data_path, tokenizer=tokenizer)
    collate = Collate(
        crop_length=args.sequence_length,
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
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )

    if 'optimizer_state_dict' in checkpoint:
        logger.info(f"Optimizer state found in checkpoint")
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    global_step = checkpoint.get('global_step', 0)
    logger.info(f"Number of completed training steps: {global_step:,}")

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

    start_time, elapsed_iters, elapsed_tokens = time.time(), 0, 0

    for ep in range(0, args.epochs):
        model.train()

        for idx, (ids, length_mask, conditional_mask) in enumerate(dataloader):
            ids, length_mask, conditional_mask = ids.to(device), length_mask.to(device), conditional_mask.to(device)

            loss, loss_diff, loss_ce, accuracy = model.compute_loss(ids, length_mask, conditional_mask)

            (loss / args.accumulation_steps).backward()

            elapsed_iters += 1
            elapsed_tokens += length_mask.sum().item()

            if ((idx + 1) % args.accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                optim.param_groups[0]['lr'] = lr_lambda(global_step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                update_model_ema(model, ema_model, args.ema_momentum)
                global_step += 1

            if global_step % args.log_interval == 0 or (idx + 1 == len(dataloader)):
                duration = time.time() - start_time
                iter_rate, token_rate = elapsed_iters / duration, elapsed_tokens / duration
                start_time, elapsed_iters, elapsed_tokens = time.time(), 0, 0

                logger.info(
                    f"Updates: {global_step:,}, "
                    f"Loss: {loss.item():,.4e}, "
                    f"MSE: {loss_diff.item():,.4e}, "
                    f"CE: {loss_ce.item():,.4e}, "
                    f"Accuracy: {accuracy.item() * 100:.2f}%, "
                    f"Throughput: {iter_rate:,.4f} it/s or {token_rate:,.4f} tok/s"
                )

                wandb.log(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]['lr'],
                        "mse": loss_diff.item(),
                        "ce": loss_ce.item(),
                        "accuracy": accuracy.item(),
                        "anisotropy": model.compute_anisotropy().item()
                    },
                    step=global_step
                )

            if global_step % args.save_interval == 0 or (idx + 1 == len(dataloader)):
                logger.info(f"Saving checkpoint: {checkpoint_path}.")
                checkpoint = {
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }
                torch.save(checkpoint, checkpoint_path)

            if global_step % args.sample_interval == 0 or (idx + 1 == len(dataloader)):
                logger.info(f"Sampling from model started.")

                output_ids = sample_model(
                    model=ema_model,
                    size=(8, args.sequence_length),
                    device=device,
                    conditional_ids=conditional_ids,
                    final_id=tokenizer.eos_id
                )
                decoded = tokenizer.decode(output_ids)
                [logger.info(f"Sample {i}:\t{text}") for i, text in enumerate(decoded)]
                model.train()

    wandb.finish()


if __name__ == "__main__":
    train()
