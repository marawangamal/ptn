import argparse
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import wandb
from ptn.models.modelling_nanogpt import GPT, GPTConfig
from ptn.dists._abc import AbstractDisributionHeadConfig
from ptn.dists.mps_sigma_lsf import MPS_SIGMA_LSF
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="ttlm")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--bpt", type=int, default=10, help="Bits per token")
parser.add_argument("--rank", type=int, default=8, help="Rank of the MPS")
parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
args = parser.parse_args()


def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------
lr = args.lr
block_size = args.seq_len
batch_size = args.batch_size
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bit_size = 2
n_bits_per_token = args.bpt
mps_rank = args.rank

# ---------------------------------------------------------------------
# 1) Train tokenizer
# ---------------------------------------------------------------------
# Initialize
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Setup trainer
trainer = trainers.BpeTrainer(
    vocab_size=bit_size**n_bits_per_token,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
)

corpus_path = "data/shakespeare/main.txt"
assert os.path.exists(corpus_path), f"File not found: {corpus_path}"

# Train
print("Vocab size before training:", tokenizer.get_vocab_size())
tokenizer.train([corpus_path], trainer)

# # Save and verify
# tokenizer.save("bpe_tokenizer.json")
print("Vocab size after training:", tokenizer.get_vocab_size())


# ---------------------------------------------------------------------
# Helpers for TTLM
# ---------------------------------------------------------------------
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches


# ---------------------------------------------------------------------
# 2) Dataset
# ---------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(
        self, path, tokenizer, block_size, n_bits_per_token=None, split="train"
    ):
        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()
        # encode entire corpus
        self.tokens = tokenizer.encode(self.text).ids
        # split into train and validation
        self.train_tokens = self.tokens[: int(len(self.tokens) * 0.9)]
        self.val_tokens = self.tokens[int(len(self.tokens) * 0.9) :]
        if split == "train":
            self.tokens = self.train_tokens
        else:
            self.tokens = self.val_tokens
        self.block_size = block_size
        self.n_bits_per_token = n_bits_per_token

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(
            self.tokens[idx + 1 : idx + 1 + self.block_size], dtype=torch.long
        )
        if self.n_bits_per_token is not None:
            x_binary = dec2bin(x, self.n_bits_per_token)
            y_binary = dec2bin(y, self.n_bits_per_token)
            return x_binary.reshape(-1).to(torch.long), y_binary.reshape(-1).to(
                torch.long
            )
        return x, y


# ---------------------------------------------------------------------
# 3) Model
# ---------------------------------------------------------------------


class TTLM(nn.Module):  # TTLM = Tensor Train Language Model
    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        print(f"config.vocab_size: {config.vocab_size}")
        word_size = math.log(config.vocab_size, bit_size)
        if word_size % 1 != 0:
            raise ValueError(
                f"vocab_size must be a power of {bit_size}, got {config.vocab_size}"
            )
        print(f"word_size: {word_size}")
        self.mps = MPS_SIGMA_LSF(
            AbstractDisributionHeadConfig(
                d_model=1,
                d_output=bit_size,
                horizon=config.block_size * int(word_size),
                rank=mps_rank,
            )
        )

    def forward(self, x, targets=None):
        B = x.shape[0]
        x = torch.ones(B, 1, device=x.device)
        out = self.mps(x, y)
        return out.logits, out.loss

    def generate(self, *args, **kwargs):
        x = torch.ones(1, 1, device=next(self.parameters()).device)
        return self.mps.generate(x)


wandb.init(project="ptn-nlp", name=f"{args.model}_shakespeare", config=vars(args))


config = GPTConfig(
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    block_size=block_size,
)
# MPS model
if args.model == "ttlm":
    model = TTLM(config)
else:
    # GPT model
    model = GPT(config)

# ---------------------------------------------------------------------
# 4) Basic training setup
# ---------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

train_dataset = TextDataset(
    "data/shakespeare/main.txt",
    tokenizer,
    block_size,
    n_bits_per_token=n_bits_per_token if args.model == "ttlm" else None,
)
val_dataset = TextDataset(
    "data/shakespeare/main.txt",
    tokenizer,
    block_size,
    n_bits_per_token=n_bits_per_token if args.model == "ttlm" else None,
    split="val",
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
num_iterations = len(train_loader) * args.epochs

# ---------------------------------------------------------------------
# 5) Training loop
# ---------------------------------------------------------------------
model.train()
print(f"Training {args.model} model on {device} for {args.epochs} epochs")
print(f"Total number of iterations: {num_iterations}")
for epoch in range(args.epochs):
    total_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (x, y) in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, targets=y)  # many GPT impls return both
        if loss is None:  # if model doesn't return loss internally
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        optimizer.step()

        wandb.log({"train/loss": loss.item()})

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    # sample from the model
    prefix = torch.tensor(tokenizer.encode("VINCENTIO").ids).reshape(1, -1)
    x = model.generate(prefix, max_new_tokens=block_size)
    if args.model == "ttlm":
        x = bin2dec(x.reshape(x.size(0), -1, n_bits_per_token), n_bits_per_token)
    print(f"Sample: {tokenizer.decode(x[0].tolist())}")
    validation_loss = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1}/{args.epochs} | Val loss: {validation_loss:.4f}")
    wandb.log({"val/loss": validation_loss})

# ---------------------------------------------------------------------
# 6) Save checkpoint
# ---------------------------------------------------------------------
torch.save(model.state_dict(), f"checkpoints/{args.model}_shakespeare.pt")
print(f"✅ Training complete. Model saved to checkpoints/{args.model}_shakespeare.pt")
