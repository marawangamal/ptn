#!/usr/bin/env python3
"""
Test script for the MultiTokenLlama model.
"""

import torch
from transformers import AutoTokenizer
from mtp.mthf import MultiTokenHFConfig, MultiTokenHF

sample_kwargs = {
    "max_new_tokens": 32,
    "do_sample": True,
    "top_k": 50,
}
model_name = "distilbert/distilgpt2"


def test_model():
    # Create model
    config = MultiTokenHFConfig(model_name=model_name, horizon=1, pretrained=True)
    model = MultiTokenHF(config)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Test data
    inputs = tokenizer("Hello world this is a test sentence:", return_tensors="pt")
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Input tokens: {inputs['input_ids']}")
    print(f"Decoded input: {tokenizer.decode(inputs['input_ids'][0])}")

    # Test forward pass with debugging
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
        print(f"Loss: {outputs['loss'].item():.4f}")
        print(f"Logits shape: {outputs['logits'].shape}")

    # Test generation
    generated = model.generate(inputs["input_ids"], **sample_kwargs)
    print(f"Generated: {tokenizer.decode(generated[0])}")

    # Test save/load
    model.save_pretrained("./test_model", safe_serialization=False)
    loaded_model = MultiTokenHF.from_pretrained("./test_model")

    # Test generation with loaded model
    generated = loaded_model.generate(inputs["input_ids"], **sample_kwargs)
    print(f"[Loaded model] Generated: {tokenizer.decode(generated[0])}")

    # Test generation with loaded model
    print("Save/load: OK")

    print("✅ All tests passed!")


if __name__ == "__main__":
    test_model()
