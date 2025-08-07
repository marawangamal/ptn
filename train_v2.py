#!/usr/bin/env python
# finetune_l3_omi_gsm.py
#
# • QLoRA fine-tune of meta-llama/Llama-3.2-3B-Instruct on OpenMathInstruct-2
# • Automatically launches LM-Eval-Harness on GSM8K after training
# Author: You :-)

import argparse, os, json, torch, subprocess, tempfile
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
p.add_argument("--dataset_name", default="nvidia/OpenMathInstruct-2")
p.add_argument("--split", default="train_1M")
p.add_argument("--max_length", type=int, default=2048)
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--warmup_steps", type=int, default=1000)
p.add_argument("--max_epochs", type=int, default=1)
p.add_argument("--devices", type=int, default=1)
p.add_argument("--output_dir", default="./checkpoints")
args = p.parse_args()

# ---------- Model & Tokenizer ----------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
tokenizer.padding_side, tokenizer.truncation_side = "left", "left"
# Llama 3.2 chat template needs no manual BOS/EOS once you call apply_chat_template

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_cfg,
    trust_remote_code=True,
)
peft_cfg = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
model = get_peft_model(model, peft_cfg)
model.gradient_checkpointing_enable()


# ---------- Dataset ----------
def _format(example):
    # Build llama-chat style prompt: user -> assistant
    messages = [
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["generated_solution"]},
    ]
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example


ds = load_dataset(args.dataset_name, split=args.split, streaming=True)
ds = ds.map(_format).shuffle(buffer_size=10_000)
ds = ds.take(1_000_000)  # stream first 1 M samples; adjust as needed


def _tokenize(batch):
    toks = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    toks["labels"] = toks["input_ids"].clone()
    return toks


dl = torch.utils.data.DataLoader(
    ds.map(_tokenize, batched=True, remove_columns=ds.features),
    batch_size=args.batch_size,
    collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    num_workers=4,
)


# ---------- Lightning Module ----------
class LitLlama(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **x):
        return self.model(**x).loss

    def training_step(self, batch, _):
        loss = self(**batch)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        sched = get_cosine_schedule_with_warmup(
            opt, args.warmup_steps, args.max_epochs * len(dl)
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    devices=args.devices,
    accelerator="gpu",
    gradient_clip_val=1.0,
    accumulate_grad_batches=2,
    default_root_dir=args.output_dir,
    precision="bf16-mixed",
)
trainer.fit(LitLlama(model), train_dataloaders=dl)

# Save merged adapter + base
final_path = os.path.join(args.output_dir, "merged")
model.save_pretrained(final_path, safe_serialization=True)
tokenizer.save_pretrained(final_path)

# ---------- Post-train GSM8K eval ----------
print("\n▶ Running LM-Eval-Harness on GSM8K…")
cmd = [
    "lm_eval",
    "--model",
    "hf",
    "--model_args",
    f"pretrained={final_path},dtype=float16",
    "--tasks",
    "gsm8k",
    "--batch_size",
    "8",
    "--output_path",
    os.path.join(args.output_dir, "gsm8k.json"),
]
subprocess.run(cmd, check=True)
print("✓ Done — see gsm8k.json for accuracy.")
