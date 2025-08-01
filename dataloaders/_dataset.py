"""This script is used to prepares HF datasets for pre-training.

Downloads, Tokenizes, and Chunks HF LM datasets. This process is CPU intensive, so we use
multiple processes to speed it up.

Example:
    >> salloc --cpus-per-task=64 --mem=64G
    >> HF_HOME=$SCRATCH/huggingface python dataloaders/prepare_hf_ds.py --dataset wikitext --subset wikitext-2-raw-v1 --split "train[:10000]"
"""

# TODO:
# - Is attn mask valid with the chunking?

import os
import argparse
from typing import Union
import datasets
from transformers import AutoTokenizer


# NOTE: `num_proc` is used in the cache signature, ensure it is the same in the
# training script. Even if you don't use it, it must be present. Otherwise, the
# cache will be invalidated and the dataset will be re-processed.


class BaseDataset:
    def __init__(self, dataset, subset, split, tokenizer, max_length):
        self.dataset = dataset
        self.subset = subset
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.max_length = max_length
        self.num_proc = 32
        self.batch_size = 1024
        self.ds = self.construct_ds()

    def construct_ds(self):
        raise NotImplementedError

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.max_length:
            total_length = (total_length // self.max_length) * self.max_length
        # Split by chunks of block_size.
        result = {
            k: [
                t[i : i + self.max_length]
                for i in range(0, total_length, self.max_length)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--max_length", type=int, default=2048)
#     p.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM-135M")
#     p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
#     p.add_argument("--split", type=str, default="train")
#     p.add_argument("--subset", type=str, default="sample-10BT")
#     args = p.parse_args()
#     ds = Fineweb(**vars(args))

#     print("Number of samples:", len(ds))
#     print("Features:", ds.column_names)
