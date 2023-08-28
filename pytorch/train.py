import argparse
import os

import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from data import SentencePieceTokenizer, TextDataset, Collate
from model import DiffusionLM
from utils import count_parameters, cosine_decay_with_warmup

torch.set_float32_matmul_precision('high')


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

    num_params = count_parameters(model)
    print(f"Total number of parameters: {num_params:,}")

    conditional_starts = [
        'this is a test',
        'once upon a time',
        'the king began thinking',
        'many people questioned the decisions of'
    ]
    conditional_ids = tokenizer.encode(conditional_starts)

    output_ids = eval_model(model, args, device, conditional_ids)
    decoded = tokenizer.decode(output_ids)
    [print(text) for text in decoded]

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
    print(f"Number of completed training steps: {global_step}")
    lr_lambda = lambda step: cosine_decay_with_warmup(step, args.learning_rate, 10000, args.decay_steps)

    num_params = count_parameters(model)
    formatted_params = "{:,}".format(num_params)
    print(f"Total number of parameters: {formatted_params}")

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

        output_ids = eval_model(model, args, device, conditional_ids)
        decoded = tokenizer.decode(output_ids)
        [print(text) for text in decoded]

    wandb.finish()


if __name__ == "__main__":
    train()
