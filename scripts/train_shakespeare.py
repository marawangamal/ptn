import os
import certifi
import torch
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

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


def compute_loss(model, dataloader, batch_size=32):
    model.eval()
    total_loss = 0
    num_batches = 0
    dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dvc)
    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"]
            x = x.to(dvc)
            dummy_input = torch.ones(x.size(0), 1, device=dvc)
            output = model(dummy_input, x)
            total_loss += output.loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_model(
    model, train_dataset, val_dataset, batch_size=32, lr=1e-3, n_epochs=50, resume=True
):
    dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto resume training from last checkpoint
    ckpt_path = "checkpoints/shakespeare/model_last.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    if os.path.exists(ckpt_path) and resume:
        model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
        print(f"Loaded model from {ckpt_path}")

    model.to(dvc)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    grads = []
    train_losses = []
    val_losses = []

    print(f"\n\nTraining model...")
    print(f"Num batches: {len(train_dataloader)}")
    print(f"Device: {dvc}")
    dummy_input = torch.ones(batch_size, 1, device=dvc)
    for epoch in range(n_epochs):
        train_losses = []
        for i, batch in tqdm(
            enumerate(train_dataloader), leave=False, total=len(train_dataloader)
        ):
            x = batch["input_ids"]
            x = x.to(dvc)
            output = model(dummy_input, x)
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            grads.append(torch.nn.utils.clip_grad_norm(model.parameters(), 1.0))
            train_losses.append(output.loss.item())

        # Compute avg train loss
        train_loss = sum(train_losses) / len(train_losses)
        train_losses.append(train_loss)

        # Compute avg val loss
        val_loss = compute_loss(model, val_dataloader)

        # Generate sample
        # Decimal
        y = model.generate(dummy_input)
        y_str = tokenizer.decode(y[0].tolist())
        y_ids = [str(k) for k in y[0][:5].tolist()]

        # Log results
        print(
            f"[Epoch {epoch + 1}/{n_epochs}] Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f} | {' '.join(y_ids)} | {repr(y_str)}"
        )

        # Save checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "epoch": epoch,
            },
            ckpt_path,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Shakespeare model.")
    # Data
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/shakespeare/main.txt",
        help="Path to the Shakespeare data.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=200_000, help="Number of samples to use."
    )
    # Model
    parser.add_argument(
        "--horizon", type=int, default=32, help="Sequence length (horizon)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size to use for training."
    )
    parser.add_argument("--d_output", type=int, default=512, help="Output dimension.")
    parser.add_argument("--d_model", type=int, default=1, help="Model dimension.")
    parser.add_argument("--rank", type=int, default=4, help="Model rank.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--n_epochs", type=int, default=50, help="Number of epochs to train for."
    )

    args = parser.parse_args()

    file_path = args.file_path
    n_samples = args.n_samples
    horizon = args.horizon
    batch_size = args.batch_size
    d_output = args.d_output
    d_model = args.d_model
    rank = args.rank
    lr = args.lr
    n_epochs = args.n_epochs

    tokenizer = get_tokenizer(
        corpus_path=file_path, bit_size=d_output, n_bits_per_token=1
    )
    dataset = ShakespeareDataset(
        tokenizer, seq_len=horizon, max_samples=n_samples + 1, file_path=file_path
    )
    # Split dataset into train and val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
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

    train_model(model, train_dataset, val_dataset, batch_size, lr=lr, n_epochs=n_epochs)
