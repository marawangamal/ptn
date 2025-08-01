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


# NOTE: `num_proc`is used in the cache signature, ensure it is the same in the
# training script. Even if you don't use it, it must be present. Otherwise, the
# cache will be invalidated and the dataset will be re-processed.
def get_dataset(
    dataset, subset, split, tokenizer, max_length
) -> Union[datasets.Dataset, datasets.DatasetDict]:

    tok = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    cache = os.environ.get("HF_HOME", "data")
    print(
        f"Args: dataset={dataset}, subset={subset}, split={split}, tokenizer={tokenizer}, max_length={max_length}, cache={cache}"
    )

    ds = datasets.load_dataset(
        dataset,
        subset,
        split=split,
        # ===== DEBUG =====
        # "wikitext",
        # "wikitext-2-raw-v1",
        # split="train[:10000]",
        # ==================
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

    # Filter empty texts
    ds = ds.filter(
        lambda x: x["text"] and x["text"].strip(),
        # num_proc=num_proc,  # type: ignore
        desc="Filter empty",  # type: ignore
    ).shuffle(seed=42)

    # Tokenize
    cols = ds.column_names
    ds = ds.map(
        tokenize,
        batched=True,
        # batch_size=batch_size,
        # num_proc=num_proc,
        desc="Tokenize",
        load_from_cache_file=True,
        remove_columns=list(cols),
    )

    # Chunk
    ds = ds.map(
        group_texts,
        batched=True,
        # batch_size=batch_size,
        # num_proc=num_proc,
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
