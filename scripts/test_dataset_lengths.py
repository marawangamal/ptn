#!/usr/bin/env python3
"""Test script to check sequence lengths in the prepared dataset."""

import sys
from dataloaders.prepare_hf_ds import get_dataset, DS_KWARGS
from tqdm import tqdm

ds = get_dataset(
    tokenizer="meta-llama/Llama-3.2-3B-Instruct", max_len=512, **DS_KWARGS["omi:1m"]
)
for i, ex in enumerate(tqdm(ds)):
    if len(ex["input_ids"]) != 512:
        print(f"Found unexpected length {len(ex['input_ids'])} at index {i}")
        break
else:
    print("All sequences have length 512")
