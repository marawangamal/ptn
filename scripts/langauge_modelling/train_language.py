import math
import os
import re
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb
from dataloaders.shakespeare import ShakespeareDataset
from dataloaders.smiles import SmilesDataset
from ptn.dists._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput
from ptn.dists.mps_sigma_lsf import MPS_SIGMA_LSF
from toks.ctokenizer import CTokenizer

from ptn.models.modelling_nanogpt_modded import GPT, GPTConfig

import argparse

DEFAULT_SMILES_PATH = "./dataloaders/data/qm9.smi"
DEFAULT_SHAKESPEARE_PATH = "./dataloaders/data/tinyshakespeare.txt"

# Use if debugging locally (i.e., running from project root)
DEFAULT_SMILES_PATH = os.path.join("scripts", "langauge_modelling", DEFAULT_SMILES_PATH)
DEFAULT_SHAKESPEARE_PATH = os.path.join(
    "scripts", "langauge_modelling", DEFAULT_SHAKESPEARE_PATH
)


def _ensure_smiles_file(path: str):
    """Create ./data/qm9.smi by downloading from HF if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Missing SMILES file and `datasets` not installed. "
            "Run `pip install datasets` or create ./data/qm9.smi manually."
        ) from e

    print(f"[download] Creating {path} from Hugging Face yairschiff/qm9 …")
    ds = load_dataset("yairschiff/qm9", split="train")
    col = "canonical_smiles" if "canonical_smiles" in ds.column_names else "smiles"
    with open(path, "w", encoding="utf-8") as f:
        for s in ds[col]:
            if isinstance(s, str) and s:
                f.write(s + "\n")
    print(f"[download] Wrote {path}")


def _ensure_shakespeare_file(path: str):
    pass
    # raise NotImplementedError("Shakespeare dataset not implemented")


def build_exp_name(args: argparse.Namespace):
    ignore_keys = ["seed", "sample", "debug"]
    abbrev_map = {
        "lm_head_d_hidden": "md",
        "lm_head_horizon": "mh",
        "lm_head_rank": "mr",
        "lm_head": "m",
        "aux_head_d_hidden": "ad",
        "aux_head_horizon": "ah",
        "aux_head_rank": "ar",
        "aux_lambda": "al",
        "aux_head": "a",
    }
    parts = []
    for k, v in args.__dict__.items():
        if k not in ignore_keys:
            # Use abbreviation if it exists, otherwise use first letter
            abbrev = abbrev_map.get(k, k[:1])
            parts.append(f"{abbrev}{v}")

    parts = [re.sub(r"[^a-zA-Z0-9]", "", p) for p in parts]
    return "_".join(parts)


class TTLM(torch.nn.Module):  # TTLM = Tensor Train Language Model
    def __init__(self, config):
        super().__init__()
        self.mps = MPS_SIGMA_LSF(
            AbstractDisributionHeadConfig(
                d_model=1,
                d_output=config["d_vocab"],
                horizon=config["lm_head_horizon"],
                rank=config["lm_head_rank"],
            )
        )

    def forward(self, x, y=None):
        B = x.shape[0]
        x = torch.ones(B, 1, device=x.device)
        out = self.mps(x, y)
        return AbstractDisributionHeadOutput(logits=out.logits, loss=out.loss)

    def generate(self, *args, **kwargs):
        x = torch.ones(1, 1, device=next(self.parameters()).device)
        return self.mps.generate(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train NanoGPT on Shakespeare")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use_scheduler", action="store_true", help="Use scheduler")
    parser.add_argument(
        "--max_samples", type=int, default=1000, help="Max samples to use"
    )
    parser.add_argument("--model", type=str, default="gpt", help="Model type")
    parser.add_argument("--lm_head", type=str, default="stp", help="LM head type")
    parser.add_argument(
        "--lm_head_horizon", type=int, default=1, help="LM head horizon"
    )
    parser.add_argument("--lm_head_rank", type=int, default=1, help="LM head rank")
    parser.add_argument(
        "--lm_head_d_hidden", type=int, default=None, help="LM head hidden dimension"
    )
    parser.add_argument(
        "--lm_head_pos_func",
        type=str,
        default="exp",
        help="LM head positivity function",
    )
    parser.add_argument(
        "--lm_head_load_balance_lambda",
        type=float,
        default=0.0,
        help="LM head load balance lambda",
    )
    parser.add_argument("--aux_head", type=str, default=None, help="Aux head type")
    parser.add_argument(
        "--aux_head_horizon", type=int, default=2, help="Aux head horizon"
    )
    parser.add_argument("--aux_head_rank", type=int, default=8, help="Aux head rank")
    parser.add_argument("--aux_lambda", type=float, default=0.1, help="Aux loss weight")
    parser.add_argument(
        "--aux_head_d_hidden", type=int, default=None, help="Aux head hidden dimension"
    )
    parser.add_argument("--sample", action="store_true", help="Sample from model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tags", type=str, nargs="*", default=[], help="Tags for wandb logging"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--save_every", type=int, default=None, help="Save every n epochs"
    )
    parser.add_argument("--dataset", type=str, default="shakespeare", help="Dataset")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer")
    return parser.parse_args()


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            if not args.model == "ttlm":
                # Shifted
                x = input_ids[:, :-1]
                y = input_ids[:, 1:]
            else:
                # Unshifted
                x, y = input_ids, input_ids.clone()
            output = model(x, y)
            total_loss += output.loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    # Initialize wandb
    wandb.init(
        project=f"ptn-{args.dataset}",
        name=build_exp_name(args),
        config=vars(args),
        tags=args.tags,
    )
    set_seed(args.seed)
    os.makedirs(os.path.join("checkpoints", build_exp_name(args)), exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Load tokenizer
    #     tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #     tokenizer.pad_token = tokenizer.eos_token

    # Load tokenizer & download dataset if needed
    dataset_path = {
        "shakespeare": DEFAULT_SHAKESPEARE_PATH,
        "smiles": DEFAULT_SMILES_PATH,
    }[args.dataset]
    _ensure_ds = {
        "shakespeare": _ensure_shakespeare_file,
        "smiles": _ensure_smiles_file,
    }[args.dataset]
    _ensure_ds(dataset_path)
    tokenizer = CTokenizer(dataset_path)

    # Load Shakespeare dataset
    full_dataset = {
        "shakespeare": ShakespeareDataset,
        "smiles": SmilesDataset,
    }[args.dataset](
        tokenizer,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        file_path=dataset_path,
    )

    # Split dataset into train and validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # print(f"Train example: {tokenizer.decode(train_dataset[0]['input_ids'])}")
    # print(f"Val example: {tokenizer.decode(val_dataset[0]['input_ids'])}")

    # Create model
    print(f"Creating model..")
    model_kwargs = {
        "n_layers": 6,
        "n_head": 6,
        "d_model": 384,
        "d_vocab": len(tokenizer),
        "dropout": 0.2,
        "d_block": args.seq_len,
        "lm_head": args.lm_head,
        "lm_head_horizon": args.lm_head_horizon,
        "lm_head_rank": args.lm_head_rank,
        "lm_head_d_hidden": args.lm_head_d_hidden,
        "lm_head_pos_func": args.lm_head_pos_func,
        "lm_head_load_balance_lambda": args.lm_head_load_balance_lambda,
        "aux_head": args.aux_head,
        "aux_head_horizon": args.aux_head_horizon,
        "aux_head_rank": args.aux_head_rank,
        "aux_head_d_hidden": args.aux_head_d_hidden,
        "aux_lambda": args.aux_lambda,
        "debug": args.debug,
    }
    if args.model == "ttlm":
        model_kwargs["lm_head_horizon"] = args.seq_len
        model = TTLM(config=model_kwargs)
    else:
        model = GPT(GPTConfig(**model_kwargs))

    print(f"Moving model to {device}")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_dataloader)
    if args.use_scheduler:
        num_warmup_steps = 100
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (
                (step + 1) / num_warmup_steps
                if step < num_warmup_steps
                else 0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * (step - num_warmup_steps)
                        / max(1, num_training_steps - num_warmup_steps)
                    )
                )
            ),
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 1.0,
        )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    print("\n📊 Training Summary")
    print("-" * 40)
    print(f"{'Total parameters:':25} {total_params:,}")
    print(f"{'Num tokens:':25} {len(tokenizer):,}")
    print(f"{'Loss (uniform dist.):':25} {math.log(len(tokenizer)):.4f}")
    print(f"{'Num training steps:':25} {num_training_steps:,}")
    print(f"{'Using device:':25} {device}")
    print("-" * 40)

    # Log to wandb
    wandb.log({"total_parameters": total_params})
    wandb.watch(model, log="all", log_freq=100)  # log_freq = steps between logging

    # Training loop
    model.train()
    best_val_loss = float("inf")
    train_start_time = time.time()
    for epoch in range(args.epochs):
        total_loss = 0
        start_time = time.time()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(device)

            # Create input and target
            if not args.model == "ttlm":
                # Shifted
                x = input_ids[:, :-1]  # All but last token
                y = input_ids[:, 1:].clone()  # All but first token
            else:
                # Unshifted
                x, y = input_ids, input_ids.clone()

            # Forward pass
            optimizer.zero_grad()
            output = model(x, y)
            loss = output.loss

            # Backward pass
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            param_norm = sum(torch.linalg.norm(p) for p in model.parameters())

            # if grad norm very large skip
            if grad_norm > 100:
                continue

            # Scale up grad norms
            if grad_norm < 1.0 and grad_norm > 0:
                scale = 1.0 / (grad_norm + 1e-6)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if loss.isnan():
                raise ValueError("Loss is NaN")

            if batch_idx % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "epoch": epoch + 1,
                        "grad_norm": grad_norm.item(),
                        "param_norm": param_norm,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

                if hasattr(output, "loss_dict") and output.loss_dict is not None:
                    for k, v in output.loss_dict.items():
                        if isinstance(v, (list, tuple, torch.Tensor)):
                            if isinstance(v, torch.Tensor):
                                # detach, move to CPU, slice, and convert to plain list
                                sampled = v.detach().cpu().view(-1)[:500].tolist()
                            else:
                                sampled = list(v)[:500]

                            wandb.log({f"train/{k}_samples": sampled})
                        else:
                            wandb.log({f"train/{k}": v})

        avg_loss = total_loss / len(train_dataloader)

        # Evaluate on validation set
        val_loss = evaluate(model, val_dataloader, device)

        print(
            f"Epoch {epoch+1} completed. Train loss: {avg_loss:.4f} | Val loss: {val_loss:.4f} | Epoch Time: {time.time() - start_time:.2f}s | Total Time: {time.time() - train_start_time:.2f}s"
        )
        wandb.log({"train/loss": avg_loss, "val/loss": val_loss, "epoch": epoch + 1})

        # Generate sample text
        if args.sample:
            model.eval()
            with torch.no_grad():
                prompt = "VIRGILIA:"
                input_ids = (
                    torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
                )
                generated = model.generate(
                    input_ids, max_output_tokens=100, stop_token=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(
                    generated[0].tolist(), skip_special_tokens=False
                )
                print(f"\nSample generation:\n{generated_text}\n")
                wandb.log(
                    {"generated_text": wandb.Html(f"<pre>{generated_text}</pre>")}
                )
            model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                },
                f"checkpoints/{build_exp_name(args)}/model_best.pt",
            )

    print("Training completed!")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
