"""This script is used to prepares HF datasets for pre-training.

Downloads, Tokenizes, and Chunks HF LM datasets. This process is CPU intensive, so we use
multiple processes to speed it up.

Example:
    >> salloc --cpus-per-task=64 --mem=64G
    >>  HF_HOME=$SCRATCH/huggingface python dataloaders/prepare_hf_ds.py --dataset wikitext --subset wikitext-2-raw-v1 --split "train[:10000]"
"""

# TODO:
# - Is attn mask valid with the chunking?

import multiprocessing
import os
import argparse
from typing import Union
import datasets
from transformers import AutoTokenizer
import hashlib, base64


def stable_fingerprint(s: str) -> str:
    # 64-bit hex string, plenty for cache keys
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def get_dataset(
    dataset,
    subset,
    split,
    tokenizer,
    max_len,
    column_names=["text"],
    num_proc=32,
    batch_size=1024,
) -> Union[datasets.Dataset, datasets.DatasetDict]:
    cache = os.environ.get("HF_HOME", "data")
    fingerprint_str = f"{dataset}-{subset}-{split}-{tokenizer}-{max_len}-{column_names}"
    fingerprint = stable_fingerprint(fingerprint_str)
    tok = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    print(f"Fingerprint: {fingerprint}")
    print(f"Fingerprint string: {fingerprint_str}")

    def tokenize(examples):
        return tok(examples["text"])

    def group_texts(examples, max_length=max_len):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Load dataset
    ds = datasets.load_dataset(
        dataset,
        subset,
        split=split,
        cache_dir=cache,
    )

    # Create a new column with the text
    ds = ds.map(
        lambda x: {"text": " ".join(x[col] for col in column_names)},
        num_proc=num_proc,
        desc="Create text column",
        load_from_cache_file=True,
        new_fingerprint=fingerprint + "_1_concat",
    )

    # Filter empty texts
    ds = ds.filter(
        lambda x: x["text"] and x["text"].strip(),
        num_proc=num_proc,  # type: ignore
        desc="Filter empty",  # type: ignore
        load_from_cache_file=True,
        new_fingerprint=fingerprint + "_2_filter",
    ).shuffle(seed=42)

    # Tokenize
    cols = ds.column_names
    ds = ds.map(
        tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Tokenize",
        remove_columns=list(cols),
        load_from_cache_file=True,
        new_fingerprint=fingerprint + "_3_tokenize",
    )

    # Chunk
    ds = ds.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Chunk",
        load_from_cache_file=True,
        new_fingerprint=fingerprint + "_4_chunk",
    )

    return ds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset", type=str, default="sample-10BT")
    p.add_argument("--column_names", type=str, nargs="*", default=["text"])
    args = p.parse_args()
    ds = get_dataset(**vars(args))

    print("Number of samples:", len(ds))
    print("Features:", ds.column_names)
