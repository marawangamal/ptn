"""This script is used to prepare the FineWeb dataset for training.

Tokenization is CPU intensive, so we use multiple processes to speed it up.
Example:
    >> salloc --cpus-per-task=64 --mem=64G
    >> python prepare_fineweb.py
"""

import argparse
import os
import datasets
from transformers import AutoTokenizer
import multiprocessing


def group_texts(examples, max_length):
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


def process_dataset(max_length, tokenizer, output_path, num_proc=None, batch_size=1000):
    num_proc = num_proc or max(1, multiprocessing.cpu_count())
    print(f"Using num_proc={num_proc}, batch_size={batch_size}")

    # 1. Load dataset (streaming disabled for now)
    dataset = datasets.load_dataset(
        "HuggingFaceFW/fineweb", "sample-10BT", split="train"
    )

    # 2. Filter empty texts (parallel)
    dataset = dataset.filter(
        lambda x: x["text"] and x["text"].strip(),
        num_proc=num_proc,
        desc="Filtering empty texts",
    )

    # 3. Shuffle for randomness
    dataset = dataset.shuffle(seed=42)

    # 4. Tokenize (batched + parallel)
    cols = dataset.column_names
    dataset = dataset.map(
        lambda x: tokenizer(x["text"]),
        remove_columns=list(cols),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Tokenizing text",
    )

    # 5. Group into fixed-length chunks (batched + parallel)
    dataset = dataset.map(
        lambda x: group_texts(x, max_length),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Grouping into blocks",
    )

    # 6. Save to disk (Arrow format)
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    return dataset


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument(
        "--tokenizer_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )
    p.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.environ.get("HF_HOME", "data"), "processed", "fineweb"),
    )
    p.add_argument("--num_proc", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1000)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    # Avoid double parallelism if using multiple processes
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dataset = process_dataset(
        args.max_length,
        tokenizer,
        args.output_path,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
    )

    # visualize
    print(dataset)
    print(dataset[0])
