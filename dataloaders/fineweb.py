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

from dataloaders._dataset import BaseDataset


# NOTE: `num_proc` is used in the cache signature, ensure it is the same in the
# training script. Even if you don't use it, it must be present. Otherwise, the
# cache will be invalidated and the dataset will be re-processed.


class Fineweb(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def construct_ds(self):
        tok = AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True)
        cache = os.environ.get("HF_HOME", "data")

        def tokenize(examples):
            return tok(examples["text"])

        ds = datasets.load_dataset(
            self.dataset,
            self.subset,
            split=self.split,
            cache_dir=cache,
        )

        # Filter empty texts
        ds = ds.filter(
            lambda x: x["text"] and x["text"].strip(),
            num_proc=NUM_PROC,  # type: ignore
            desc="Filter empty",  # type: ignore
        ).shuffle(seed=42)

        # Tokenize
        cols = ds.column_names
        ds = ds.map(
            tokenize,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc="Tokenize",
            load_from_cache_file=True,
            remove_columns=list(cols),
        )

        # Chunk
        ds = ds.map(
            self.group_texts,
            batched=True,
            batch_size=self.batch_size,
            num_proc=self.num_proc,
            desc="Chunk",
            load_from_cache_file=True,
        )

        return ds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset", type=str, default="sample-10BT")
    args = p.parse_args()
    ds = Fineweb(**vars(args)).ds

    print("Number of samples:", len(ds))
    print("Features:", ds.column_names)
