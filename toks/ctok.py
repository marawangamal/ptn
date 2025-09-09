class CTokenizer:
    """
    Simple character-level tokenizer for SMILES.

    API (HF-like subset):
      - encode(text: str, add_eos: bool=True) -> List[int]
      - decode(ids: List[int], skip_special_tokens: bool=True) -> str
      - __len__() -> vocab size
      - pad_token(_id), eos_token(_id), unk_token(_id)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

        # special tokens
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        # build vocab from entire file content (keeps '\n')
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        charset = set(text)  # includes '\n' since we used read()
        # make sure we keep specials first, then sorted chars
        self.tokens = [self.pad_token, self.eos_token, self.unk_token] + sorted(charset)
        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}

        self.pad_token_id = self.token2id[self.pad_token]
        self.eos_token_id = self.token2id[self.eos_token]
        self.unk_token_id = self.token2id[self.unk_token]

        print(f"[SmilesCharTokenizer] vocab={len(self.tokens)} from {filepath}")

    def encode(self, text: str, add_eos: bool = True):
        ids = [self.token2id.get(ch, self.unk_token_id) for ch in text]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids, skip_special_tokens: bool = True):
        out = []
        for i in ids:
            tok = self.id2token.get(int(i), self.unk_token)
            if skip_special_tokens and tok in {self.pad_token, self.eos_token}:
                continue
            out.append(tok if tok != self.unk_token else "?")
        return "".join(out)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return f"CTokenizer(tokens={self.tokens})"


if __name__ == "__main__":
    # quick smoke test
    tok = CTokenizer("./dataloaders/data/qm9.smi")
    demo = "CCO\nc1ccccc1"
    ids = tok.encode(demo)
    print("Encoded (first 30):", ids[:30], "…")
    print("Decoded (first 60 chars):", tok.decode(ids)[:60])
