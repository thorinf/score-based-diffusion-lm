import argparse
import os
import logger
from data import SentencePieceTokenizer, TextDataset, Collate, infinite_loader
from torch.utils.data import DataLoader
from model import ScoreLM
from diffusion import MultiStepScoreDiffusion
from trainer import Trainer
from utils import get_text
import wandb


def main():
    args = create_argparser().parse_args()
    logger.configure(args.model_dir)

    tokenizer = SentencePieceTokenizer(args.spm_model)

    logger.log("creating model and diffusion...")

    model = ScoreLM(
        num_classes=len(tokenizer),
        embedding_dim=args.embedding_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        dropout_prob=args.dropout_prob,
        layerdrop_prob=args.layerdrop_prob,
    )

    diffusion = MultiStepScoreDiffusion(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=1.0,
        rho=args.rho
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.log(f"total parameter count: {num_params:,}")

    wandb.init(
        name=args.model_dir,
        project=os.getenv("WANDB_PROJECT", "score_diffusion_lm"),
        dir=args.model_dir,
    )
    wandb.config.update(args.__dict__, allow_val_change=True)

    dataset = TextDataset(path=args.data_path, tokenizer=tokenizer)

    pad_sequence_value = tokenizer.pad_id if tokenizer.pad_id > 0 else tokenizer.eos_id
    collate = Collate(
        max_sequence_length=args.sequence_length,
        pad_sequence_value=pad_sequence_value,
        random_length_expansion=True,
        insert_value=tokenizer.pad_id,
        insert_rate=0.0
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate
    )
    dataloader = infinite_loader(dataloader)

    conditional_starts = get_text("conditional_starts.txt")

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        tokenizer=tokenizer,
        data=dataloader,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        ema_rate=args.ema_rate,
        model_dir=args.model_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        sample_size=(args.num_examples, args.sequence_length),
        sample_conditioning=conditional_starts,
        sample_iterations=1000,
        resume_checkpoint=True,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
    )
    trainer.run_loop()


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bsz', '--batch_size', type=int, default=128)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=1)
    parser.add_argument('-svi', '--log_interval', type=int, default=50)
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
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.1)
    parser.add_argument('-gc', '--gradient_clipping', type=float, default=-1.0)
    parser.add_argument('-ema', '--ema_rate', default="0.95, 0.9999")

    parser.add_argument('-slen', '--sequence_length', type=int, default=64)
    parser.add_argument('-smin', '--sigma_min', type=float, default=1.0)
    parser.add_argument('-smax', '--sigma_max', type=float, default=10.0)
    parser.add_argument('-rho', '--rho', type=float, default=1.0)
    parser.add_argument('-nex', '--num_examples', type=int, default=8)

    parser.add_argument('-mdir', '--model_dir', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-spm', '--spm_model', type=str, required=True)
    return parser


if __name__ == "__main__":
    main()
