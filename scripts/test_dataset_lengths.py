#!/usr/bin/env python3
"""Test script to check sequence lengths in the prepared dataset."""

import torch
from transformers.data.data_collator import default_data_collator
from dataloaders.prepare_hf_ds import get_dataset, DS_KWARGS
from tqdm import tqdm

ds = get_dataset(
    tokenizer="meta-llama/Llama-3.2-3B-Instruct", max_len=512, **DS_KWARGS["omi:1m"]
)

ds = ds.train_test_split(test_size=0.1)
ds["val"] = ds["test"]
print(ds)

dl = torch.utils.data.DataLoader(
    ds["train"],
    # ds,
    batch_size=8,
    collate_fn=default_data_collator,
)


# ENUMERATE DATASET
# for i, ex in enumerate(tqdm(ds)):
#     if len(ex["input_ids"]) != 512:
#         print(f"Found unexpected length {len(ex['input_ids'])} at index {i}")
#         break

# ENUMERATE DATALOADER
ex = next(iter(dl))
print(ex.keys())

for i, ex in enumerate(tqdm(dl)):
    # print(len(ex["input_ids"][0]))
    # print(ex["input_ids"].shape)
    if any(len(ex[k][0]) != 512 for k in ex.keys()):
        lens = [(k, len(ex[k][0])) for k in ex.keys()]
        print("|".join(f"{k}: {v}" for k, v in lens))
        break
    # if len(ex["input_ids"][0]) != 512:
    #     print(f"[INPUTS] Found unexpected length {len(ex['input_ids'])} at index {i}")
    #     break
    # elif len(ex["labels"][0]) != 512:
    #     print(f"[LABELS] Found unexpected length {len(ex['labels'])} at index {i}")
    #     break
    # elif len(ex["attention_mask"][0]) != 512:
    #     print(
    #         f"[MASK] Found unexpected length {len(ex['attention_mask'])} at index {i}"
    #     )
    #     break
else:
    print("All sequences have length 512")
