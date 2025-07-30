import argparse
import os
import datasets
from transformers import AutoTokenizer


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


def process_dataset(max_length, tokenizer, output_path):
    dataset = datasets.load_dataset(
        "HuggingFaceFW/fineweb", "sample-10BT", split="train"
    )
    dataset = dataset.filter(lambda x: x["text"] and x["text"].strip() != "")
    dataset = dataset.shuffle(seed=42)
    cols = dataset.column_names
    # Tokenize
    dataset = dataset.map(
        lambda x: tokenizer(x["text"]),
        remove_columns=cols,
        batched=True,
        desc="Tokenizing text",
    )

    # Group instead of padding/truncation
    dataset = dataset.map(
        lambda x: group_texts(x, max_length),
        batched=True,
        desc="Groupinging text",
    )

    # save (saves arrow format ds)
    dataset.save_to_disk(output_path)
    return dataset


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--tokenizer_name", type=str, default="gpt2")
    p.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.environ.get("HF_HOME", "data"), "processed", "fineweb"),
    )
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    dataset = process_dataset(args.max_length, tokenizer, args.output_path)

    # visualize
    print(dataset)
    print(dataset[0])
