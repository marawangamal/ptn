import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from ptn.models.modelling_nanogpt import GPT, GPTConfig
from ptn.dists._abc import AbstractDisributionHeadConfig
from ptn.dists.mps_sigma_lsf import MPS_SIGMA_LSF
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import os

# Initialize
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Setup trainer
trainer = trainers.BpeTrainer(
    vocab_size=2**10, special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
)

corpus_path = "data/shakespeare/main.txt"
assert os.path.exists(corpus_path), f"File not found: {corpus_path}"

# Train
print("Vocab size before training:", tokenizer.get_vocab_size())
tokenizer.train([corpus_path], trainer)

# Save and verify
tokenizer.save("bpe_tokenizer.json")
print("Vocab size after training:", tokenizer.get_vocab_size())


def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


# # Example
# NUM_BITS_PER_TOKEN = 10
# d = torch.randint(0, 16, (3, 6))
# b = dec2bin(d, NUM_BITS_PER_TOKEN)
# d_rec = bin2dec(b, NUM_BITS_PER_TOKEN)

# ---------------------------------------------------------------------
# Run training using MPS
# ---------------------------------------------------------------------

#  --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0

# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------
lr = 3e-4
block_size = 32
batch_size = 12
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bit_size = 2
n_bits_per_token = 10
mps_rank = 8
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 1) Load the trained tokenizer
# ---------------------------------------------------------------------
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")


# ---------------------------------------------------------------------
# 2) Dataset and DataLoader
# ---------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, path, tokenizer, block_size, n_bits_per_token=None):
        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()
        # encode entire corpus
        self.tokens = tokenizer.encode(self.text).ids
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
# 3) Instantiate model
# ---------------------------------------------------------------------


class TTModel(nn.Module):
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

    def generate(self):
        x = torch.ones(1, 1, device=next(self.parameters()).device)
        return self.mps.generate(x)


config = GPTConfig(
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    block_size=block_size,
)
# MPS model
model = TTModel(config)

# # GPT model
# model = GPT(config)

# ---------------------------------------------------------------------
# 4) Basic training setup
# ---------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
# model = torch.compile(model) # requires PyTorch 2.0

train_dataset = TextDataset(
    "data/shakespeare/main.txt", tokenizer, block_size, n_bits_per_token=10
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------
# 5) Training loop
# ---------------------------------------------------------------------
epochs = 5
model.train()
print(f"Training model on {device} for {epochs} epochs")
for epoch in range(epochs):
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

        total_loss += loss.item()
        if i % 50 == 0:
            # sample from the model
            xb = model.generate()
            xd = bin2dec(xb.reshape(xb.size(0), -1, n_bits_per_token), n_bits_per_token)
            print(
                f"Epoch {epoch+1} Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}"
            )
            print(f"Sample: {tokenizer.decode(xd[0].tolist())}")
            pass

        pbar.set_postfix(loss=loss.item())

# ---------------------------------------------------------------------
# 6) Save checkpoint
# ---------------------------------------------------------------------
torch.save(model.state_dict(), "gpt_shakespeare.pt")
print("✅ Training complete. Model saved to gpt_shakespeare.pt")
