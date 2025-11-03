import os
import certifi
import torch
from tqdm import tqdm
import torchvision
from torchvision import transforms
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ptn.dists import dists
from ptn.dists._abc import AbstractDisributionHeadConfig

# Set the SSL certificate file for secure downloads
os.environ["SSL_CERT_FILE"] = certifi.where()


class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        seq_len=256,
        max_samples=None,
        file_path="dataloaders/data/tinyshakespeare.txt",
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Read Shakespeare text
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)

        # Limit samples if specified
        if max_samples:
            tokens = tokens[: max_samples * seq_len]

        # Create sequences
        self.sequences = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            seq = tokens[i : i + seq_len]
            if len(seq) == seq_len:
                self.sequences.append(seq)
        # for i in range(0, len(tokens) - seq_len, seq_len // 2):  # 50% overlap
        #     seq = tokens[i : i + seq_len]
        #     if len(seq) == seq_len:
        #         self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {"input_ids": torch.tensor(seq, dtype=torch.long)}


class BPETokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.eos_token_id = tokenizer.token_to_id(self.eos_token)
        self.pad_token_id = tokenizer.token_to_id(self.pad_token)

    def __len__(self):
        return self.vocab_size

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text).ids

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids)

    def get_vocab_size(self):
        return self.vocab_size


def get_tokenizer(corpus_path, bit_size, n_bits_per_token):
    # You seem to want a specific vocab size tied to your later bin-coding:
    vocab_size = bit_size**n_bits_per_token  # e.g., 2**8 = 256, 2**16 = 65536

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
    )

    tok.train([corpus_path], trainer)
    return BPETokenizerWrapper(tok)


def train_model(model, dataset, batch_size=32, lr=1e-3, n_epochs=50):
    dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dvc)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    grads = []
    losses = []
    print(f"\n\nTraining model...")
    print(f"Num batches: {len(dataloader)}")
    for epoch in range(n_epochs):
        train_losses = []
        for i, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader)):
            x = batch["input_ids"]
            x = x.to(dvc)
            output = model(torch.ones(x.size(0), 1), x)
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            grads.append(torch.nn.utils.clip_grad_norm(model.parameters(), 1.0))
            train_losses.append(output.loss.item())

        loss_avg = sum(train_losses) / len(train_losses)
        losses.append(loss_avg)

        # Generate sample
        # Decimal
        y = model.generate(torch.ones(1, 1))
        y_str = tokenizer.decode(y[0].tolist())
        y_ids = [str(k) for k in y[0][:5].tolist()]

        print(
            f"[Epoch{epoch + 1}/{n_epochs}][{i + 1}/{len(dataloader)}] Loss: {loss_avg:.2f} | {' '.join(y_ids)} | {repr(y_str)}"
        )


if __name__ == "__main__":
    # Hyperparameters
    # Data
    file_path = "../data/shakespeare/main.txt"
    n_samples = 200_000
    # Model
    horizon = 32  # i.e. sequence length
    batch_size = 32
    d_output = 512
    d_model = 1
    rank = 4

    tokenizer = get_tokenizer(
        corpus_path=file_path, bit_size=d_output, n_bits_per_token=1
    )
    dataset = ShakespeareDataset(
        tokenizer, seq_len=horizon, max_samples=n_samples + 1, file_path=file_path
    )
    print("Num. Tokens:", tokenizer.get_vocab_size())

    # Hyperparameters

    model = dists["mps_sigma_lsf"](
        AbstractDisributionHeadConfig(
            d_model=d_model,
            d_output=d_output,
            horizon=horizon,
            rank=rank,
            pos_func="abs",
            mode="direct",
            init_method="ortho",
        )
    )

    train_model(model, dataset, batch_size)
