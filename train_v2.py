#!/usr/bin/env python
# finetune_l3_omi_gsm.py
#
# • QLoRA fine-tune of meta-llama/Llama-3.2-3B-Instruct on OpenMathInstruct-2
# • Automatically launches LM-Eval-Harness on GSM8K after training
# Author: You :-)

from pytorch_lightning.callbacks import ModelCheckpoint
import argparse, os, json, torch, subprocess
from typing import List
import pytorch_lightning as pl
import lm_eval
from lm_eval import simple_evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)


# --- callbacks.py (or just paste above the main if you prefer) ----------------
import subprocess, json, random, torch, pytorch_lightning as pl
from pathlib import Path


class PeriodicSample(pl.Callback):
    def __init__(self, prompts, tokenizer, every_steps=200, max_new=128, **gen_kwargs):
        self.prompts = prompts
        self.tok = tokenizer
        self.every_steps = every_steps
        self.max_new = max_new
        self.gen_kwargs = dict(temperature=0.2, do_sample=True, **gen_kwargs)

    def on_train_batch_end(self, trainer, pl_module, *_):
        step = trainer.global_step
        if step == 0 or step % self.every_steps:
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


class PeriodicEvaluate(pl.Callback):
    def __init__(self, evals, batch_size, limit=None, every_steps=100):
        self.evals = evals
        self.batch_size = batch_size
        self.limit = limit
        self.every_steps = every_steps

    def _run_eval(self, pl_module, logger_prefix=None):
        print(
            f"Running eval on {self.evals} with batch size {self.batch_size} and limit {self.limit}"
        )
        output = simple_evaluate(
            model=lm_eval.models.huggingface.HFLM(pretrained=pl_module.model),
            tasks=self.evals,
            batch_size=self.batch_size,
            limit=self.limit,
        )
        for task_name in output["results"].keys():
            # Example output:
            # >> output['results']['gsm8k_cot']
            # {'alias': 'gsm8k_cot', 'exact_match,strict-match': np.float64(0.5),
            # 'exact_match_stderr,strict-match': 0.16666666666666666,
            # 'exact_match,flexible-extract': np.float64(0.5),
            # 'exact_match_stderr,flexible-extract': 0.16666666666666666}
            for metric_name in output["results"][task_name].keys():
                # if metric is a number, log it
                if isinstance(output["results"][task_name][metric_name], (int, float)):
                    pl_module.log(
                        f"eval/{task_name}/{metric_name}",
                        output["results"][task_name][metric_name],
                        sync_dist=False,
                    )
                    print(
                        f"[{logger_prefix}] (LMEvalsCallback) {task_name}/{metric_name} = {output['results'][task_name][metric_name]}"
                    )

    def on_train_batch_end(self, trainer, pl_module, *_):
        step = trainer.global_step
        if step == 0 or step % self.every_steps:
            return

        pl_module.eval()
        with torch.no_grad():
            self._run_eval(pl_module, logger_prefix=f"[{step}]")
        pl_module.train()


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


DATASET_KWARGS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
    },
    "omi:1m": {
        "path": "nvidia/OpenMathInstruct-2",
        "split": "train_1M",
    },
}

RENAMED_COLUMNS = {
    "omi:1m": {
        "problem": "question",
        "generated_solution": "answer",
    },
}

# ---------- CONSTANTS ----------
OUTPUT_DIR = "./checkpoints"

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
p.add_argument("--dataset", default="gsm8k")
p.add_argument("--max_length", type=int, default=512)
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--warmup_steps", type=int, default=100)
p.add_argument("--max_epochs", type=int, default=1)
p.add_argument("--max_steps", type=int, default=None)
p.add_argument("--accumulate_grad_batches", type=int, default=2)
# eval
p.add_argument("--eval_every", type=int, default=100)
p.add_argument("--eval_batch_size", type=int, default=8)
p.add_argument("--eval_limit", type=int, default=16)
p.add_argument("--eval_evals", type=str, default="gsm8k_cot")
args = p.parse_args()

# ---------- Model & Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# make sure we have a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


# ---------- Dataset ----------
def _format(example):
    # Build llama-chat style prompt: user -> assistant
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example


def _tokenize(batch):
    toks = tokenizer(
        batch["text"],
        truncation=True,
        max_length=args.max_length,
    )
    return toks


ds = load_dataset(**DATASET_KWARGS[args.dataset])
ds = ds.rename_columns(RENAMED_COLUMNS.get(args.dataset, {}))
cols = ds.column_names
ds = ds.map(_format)
ds = ds.map(_tokenize, batched=True, remove_columns=cols + ["text"])
dl = torch.utils.data.DataLoader(
    ds,
    batch_size=args.batch_size,
    collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    num_workers=4,
)

# choose some fixed validation prompts once
val_prompts = [
    "A train travels 120 miles at 60 mph. How long does the trip take?",
]
callbacks: List[pl.Callback] = [
    PeriodicSample(val_prompts, tokenizer, every_steps=args.eval_every),  # ← step freq
    PeriodicEvaluate(
        args.eval_evals,
        args.eval_batch_size,
        limit=args.eval_limit,
        every_steps=args.eval_every,
    ),
]


max_steps = args.max_steps
if max_steps is None:
    max_steps = (
        args.max_epochs
        * len(dl)
        / (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        / args.accumulate_grad_batches
    )

trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    gradient_clip_val=1.0,
    accumulate_grad_batches=args.accumulate_grad_batches,
    default_root_dir=OUTPUT_DIR,
    max_steps=max_steps,
    callbacks=callbacks,
)
trainer.fit(LitLlama(model, max_steps), train_dataloaders=dl)

# # Save merged adapter + base
# final_path = os.path.join(OUTPUT_DIR, "merged")
# model.save_pretrained(final_path, safe_serialization=True)
# tokenizer.save_pretrained(final_path)

# ---------- Post-train GSM8K eval ----------
# print("\n▶ Running LM-Eval-Harness on GSM8K…")
# cmd = [
#     "lm_eval",
#     "--model",
#     "hf",
#     "--model_args",
#     f"pretrained={final_path},dtype=float16",
#     "--tasks",
#     "gsm8k",
#     "--batch_size",
#     "8",
#     "--output_path",
#     os.path.join(OUTPUT_DIR, "gsm8k.json"),
# ]
# subprocess.run(cmd, check=True)
# print("✓ Done — see gsm8k.json for accuracy.")
