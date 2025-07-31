import multiprocessing
import os
import argparse
import datasets
from transformers import AutoTokenizer


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    ds = datasets.load_dataset(
        # args.dataset
        # args.subset
        # split = args.split
        # ===== DEBUG =====
        "wikitext",
        "wikitext-2-raw-v1",
        split="train[:10000]",
        # ==================
        cache_dir=args.cache,
    )

    def tokenize(examples):
        return tokenizer(examples["text"])

    def group_texts(examples, max_length=args.max_length):
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
        num_proc=args.num_proc,  # type: ignore
        desc="Filter empty",  # type: ignore
    ).shuffle(seed=42)

    # Tokenize
    cols = ds.column_names
    ds = ds.map(
        tokenize,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="Tokenize",
        load_from_cache_file=True,
        remove_columns=list(cols),
    )

    # Chunk
    ds = ds.map(
        group_texts,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="Chunk",
        load_from_cache_file=True,
    )

    return ds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--cache", type=str, default=os.environ.get("HF_HOME", "data"))
    p.add_argument("--num_proc", type=int, default=multiprocessing.cpu_count())
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--subset", type=str, default="sample-10BT")
    args = p.parse_args()
    ds = main(args)

    print("Number of samples:", len(ds))
    print("Features:", ds.column_names)
