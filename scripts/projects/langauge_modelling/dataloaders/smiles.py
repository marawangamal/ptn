# dataloaders/smiles.py
import os
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

DEFAULT_SMILES_PATH = "./dataloaders/data/qm9.smi"


def _ensure_smiles_file(path: str):
    """Download QM9 SMILES from Hugging Face if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return path

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "SMILES file missing and `datasets` not installed. "
            "Run `pip install datasets`."
        ) from e

    print(f"[download] Creating {path} from Hugging Face (yairschiff/qm9)…")
    ds = load_dataset("yairschiff/qm9", split="train")
    col = "canonical_smiles" if "canonical_smiles" in ds.column_names else "smiles"
    with open(path, "w", encoding="utf-8") as f:
        for s in ds[col]:
            if isinstance(s, str) and s:
                f.write(s + "\n")
    print(f"[download] Wrote {path}")
    return path


class SmilesDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        seq_len: int = 256,
        max_samples: int = None,
        file_path: str = DEFAULT_SMILES_PATH,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # ensure file exists (download if needed)
        file_path = _ensure_smiles_file(file_path)

        # read whole corpus
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # tokenize once
        tokens = self.tokenizer.encode(text)

        if max_samples:
            tokens = tokens[: max_samples * seq_len]

        # overlapping windows like ShakespeareDataset
        self.sequences = []
        step = max(1, seq_len // 2)
        for i in range(0, len(tokens) - seq_len, step):
            seq = tokens[i : i + seq_len]
            if len(seq) == seq_len:
                self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {"input_ids": torch.tensor(seq, dtype=torch.long)}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = SmilesDataset(tokenizer, max_samples=1000)
    print(f"Encoded: {dataset[0]['input_ids']}")
    print(f"Decoded: {tokenizer.decode(dataset[0]['input_ids'])}")
