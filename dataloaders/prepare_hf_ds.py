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


def get_dataset(
    dataset, subset, split, tokenizer, max_length, column_names=["text"]
) -> Union[datasets.Dataset, datasets.DatasetDict]:

    tok = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    cache = os.environ.get("HF_HOME", "data")
    print(
        f"Args: dataset={dataset}, subset={subset}, split={split}, tokenizer={tokenizer}, max_length={max_length}, cache={cache}"
    )

    # ---------------------------------------------------------------------------
    # Hard-coded parallelism & batching parameters.
    # These speed up HF `Dataset.map`/`filter` dramatically while keeping the
    # function signature unchanged.
    # ---------------------------------------------------------------------------
    BATCH_SIZE = 1024
    NUM_PROC = 32

    ds = datasets.load_dataset(
        dataset,
        subset,
        split=split,
        cache_dir=cache,
    )

    def tokenize(examples):
        return tok(examples["text"])

    def group_texts(examples, max_length=max_length):
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

    # Create a new column with the text
    ds = ds.map(
        lambda x: {"text": " ".join(x[col] for col in column_names)},
        num_proc=NUM_PROC,
        desc="Create text column",
        load_from_cache_file=True,
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
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        desc="Tokenize",
        load_from_cache_file=True,
        remove_columns=list(cols),
    )

    # Chunk
    ds = ds.map(
        group_texts,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        desc="Chunk",
        load_from_cache_file=True,
    )

    return ds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset", type=str, default="sample-10BT")
    args = p.parse_args()
    ds = get_dataset(**vars(args))

    print("Number of samples:", len(ds))
    print("Features:", ds.column_names)
