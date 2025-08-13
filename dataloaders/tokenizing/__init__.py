import torch
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, encoder, decoder, vocab_size, name=None):
        self.encode = encoder
        self.decode = decoder
        self.vocab_size = vocab_size
        self.name = name

    def tokenize(self, data_list):
        """
        Takes a list of prefix-target pairs, tokenizes and concatenates them
        """
        out = []
        prefix_len = len(self.encode(data_list[0][0]))
        target_len = len(self.encode(data_list[0][1]))
        same_len = True

        for prefix, target in data_list:
            prefix = torch.tensor(self.encode(prefix))
            target = torch.tensor(self.encode(target))
            if not (len(prefix) == prefix_len and len(target) == target_len):
                same_len = False
            seq = torch.concatenate([prefix, target], dim=-1).long()
            out.append(seq)

        # Check if all prefixes and all targets have the same length
        if not same_len:
            print("Not all prefixes or targets have the same length!!")
        else:
            print("Equal sequence lengths!")

        return out, prefix_len, target_len


def get_tokenizer(model_name):
    t = AutoTokenizer.from_pretrained(model_name)
    tokenizer = Tokenizer(
        encoder=t.encode, decoder=t.decode, vocab_size=50257, name="gpt2"
    )
    return tokenizer
