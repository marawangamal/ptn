import torch
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
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
        for i in range(0, len(tokens) - seq_len, seq_len // 2):  # 50% overlap
            seq = tokens[i : i + seq_len]
            if len(seq) == seq_len:
                self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {"input_ids": torch.tensor(seq, dtype=torch.long)}
