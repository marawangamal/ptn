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


# --- callbacks.py (or just paste above the main if you prefer) ----------------
import subprocess, json, random, torch, pytorch_lightning as pl
from pathlib import Path


class PeriodicSample(pl.Callback):
    def __init__(self, prompts, tokenizer, every_steps=200, max_new=128, **gen_kwargs):
        self.prompts = prompts
        self.tok = tokenizer
        self.every = every_steps
        self.max_new = max_new
        self.gen_kwargs = dict(temperature=0.2, do_sample=True, **gen_kwargs)

    def on_train_batch_end(self, trainer, pl_module, *_):
        step = trainer.global_step
        if step == 0 or step % self.every:
            return

        pl_module.eval()
        with torch.no_grad():
            prompt = random.choice(self.prompts)
            inputs = self.tok(prompt, return_tensors="pt").to(pl_module.device)
            out_ids = pl_module.model.generate(
                **inputs, max_new_tokens=self.max_new, **self.gen_kwargs
            )
            completion = self.tok.decode(
                out_ids[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
            )
            print(f"[{step}] {prompt} -> {completion}")
        pl_module.train()
        trainer.logger.log_text(
            key="samples",
            columns=["step", "prompt", "completion"],
            data=[[step, prompt.strip(), completion.strip()]],
        )


# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
p.add_argument("--dataset_name", default="nvidia/OpenMathInstruct-2")
p.add_argument("--split", default="train_1M")
p.add_argument("--max_length", type=int, default=512)
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--warmup_steps", type=int, default=1000)
p.add_argument("--max_epochs", type=int, default=1)
p.add_argument("--output_dir", default="./checkpoints")
p.add_argument("--max_steps", type=int, default=20_000)

args = p.parse_args()

# ---------- Model & Tokenizer ----------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
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


# make sure we have a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


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
cols = ds.column_names
ds = ds.map(_format).shuffle(buffer_size=10_000)
ds = ds.take(1_000_000)  # stream first 1 M samples; adjust as needed


def _tokenize(batch):
    toks = tokenizer(
        batch["text"],
        truncation=True,
        max_length=args.max_length,
    )
    return toks


dl = torch.utils.data.DataLoader(
    ds.map(_tokenize, batched=True, remove_columns=cols + ["text"]),
    batch_size=args.batch_size,
    collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    num_workers=4,
)


# ---------- Lightning Module ----------
class LitLlama(pl.LightningModule):
    def __init__(self, model, max_steps):
        super().__init__()
        self.model = model
        self.max_steps = max_steps

    def forward(self, **x):
        return self.model(**x).loss

    def training_step(self, batch, _):
        loss = self(**batch)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        sched = get_cosine_schedule_with_warmup(
            opt, args.warmup_steps, args.max_epochs * self.max_steps
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


# choose some fixed validation prompts once
val_prompts = [
    "A train travels 120 miles at 60 mph. How long does the trip take?",
    "If 3x + 2 = 14, what is x?",
    "Tom has 5 apples and eats 2. How many are left?",
]

callbacks = [
    PeriodicSample(val_prompts, tokenizer, every_steps=500),  # ← step freq
]


trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    # devices=args.devices,
    # accelerator="gpu",
    gradient_clip_val=1.0,
    accumulate_grad_batches=2,
    default_root_dir=args.output_dir,
    precision="bf16-mixed",
)
trainer.fit(LitLlama(model, args.max_steps), train_dataloaders=dl)

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
