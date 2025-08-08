#!/usr/bin/env python
# train_v2.py
#
# To evaluate a ckpt, run:
# accelerate launch -m lm_eval --model hf --tasks lambada_openai,arc_easy  --batch_size 16

import os
import re
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import argparse, torch
from typing import List
import pytorch_lightning as pl
import lm_eval
from lm_eval import simple_evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


# --- callbacks.py (or just paste above the main if you prefer) ----------------
import subprocess, json, random, torch, pytorch_lightning as pl
from pathlib import Path

import wandb

from mtp.mthf.modelling_mthf import MultiTokenHF, MultiTokenHFConfig


# ---------- Constants ----------
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

OUTPUT_DIR = "./experiments"


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
            columns = ["text"]
            data = [[completion]]
            trainer.logger.log_text(key="samples", columns=columns, data=data)
        pl_module.train()


class PeriodicEvaluate(pl.Callback):
    def __init__(
        self, evals, batch_size, limit=None, every_steps=100, val_on_start=True
    ):
        self.evals = evals
        self.batch_size = batch_size
        self.limit = limit
        self.every_steps = every_steps
        self.val_on_start = val_on_start

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
        if (step == 0 and not self.val_on_start) or (step % self.every_steps):
            return

        pl_module.eval()
        with torch.no_grad():
            self._run_eval(pl_module, logger_prefix=f"{step}")
        pl_module.train()


# ---------- Lightning Module ----------
class LitLlama(pl.LightningModule):
    def __init__(self, model, max_steps, lr=2e-5, warmup_ratio=0.05):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.max_steps = max_steps

    def forward(self, **x):
        return self.model(**x).loss

    def training_step(self, batch, _):
        loss = self(**batch)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        wr = self.hparams["warmup_ratio"]
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["lr"], weight_decay=0
        )
        warmup_steps = int(wr * self.hparams["max_steps"])
        warmup_factor = lambda st: wr + (1 - wr) * (st / max(warmup_steps, 1))
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, warmup_factor)
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams["max_steps"] - warmup_steps,
            eta_min=0.1 * self.hparams["lr"],  # end at 10% of lr
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_scheduler, cos_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    def on_after_backward(self) -> None:
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.detach().norm(2)
                    for p in self.parameters()
                    if p.grad is not None
                ]
            ),
            2,
        )
        self.log("grad_norm", norm, on_step=True, on_epoch=False)


# ------------- Utils -------------------
def get_econfig_name(args: argparse.Namespace):
    ignore_keys = [
        "eval_every",
        "eval_batch_size",
        "eval_limit",
        "eval_evals",
        "tags",
        "ckpt_every",
    ]
    parts = [f"{k[:1]}{v}" for k, v in args.__dict__.items() if k not in ignore_keys]
    # remove special characters
    parts = [re.sub(r"[^a-zA-Z0-9]", "", p) for p in parts]
    return "_".join(parts)


def lookup_ckpt(args: argparse.Namespace):
    ckpt_path = f"{OUTPUT_DIR}/{get_econfig_name(args)}/last.ckpt"
    if not os.path.exists(ckpt_path):
        return None
    return ckpt_path


def lookup_wandb_run(args: argparse.Namespace):
    run_name = get_econfig_name(args)
    runs = wandb.Api(timeout=15).runs("mtl")
    matches = [r for r in runs if r.name == run_name]
    matches.sort(key=lambda x: x.created_at, reverse=True)
    if len(matches) == 0:
        return None
    return matches[0].id


# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
p.add_argument("--dataset", default="omi:1m")
p.add_argument("--max_length", type=int, default=512)
p.add_argument("--batch_size", type=int, default=8)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--max_epochs", type=int, default=1)
p.add_argument("--max_steps", type=int, default=None)
p.add_argument("--accumulate_grad_batches", type=int, default=1)
# eval
p.add_argument("--eval_every", type=int, default=5000)
p.add_argument("--eval_batch_size", type=int, default=64)
p.add_argument("--eval_limit", type=int, default=None)
p.add_argument("--eval_evals", type=str, default="gsm8k_cot")
p.add_argument("--tags", type=str, nargs="*", default=[])
# ckpt
p.add_argument("--ckpt_every", type=int, default=1000)
args = p.parse_args()

# ---------- Setup ----------------------
os.makedirs(os.path.join(OUTPUT_DIR, get_econfig_name(args)), exist_ok=True)

# ---------- Model & Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
# model = MultiTokenHF(
#     MultiTokenHFConfig(
#         model_name=args.model,
#         model_head="stp",
#         horizon=1,
#         loss_type="mhead",
#         pretrained=True,
#         lambda_mhead=0.0,
#     )
# )

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


ds = load_dataset(**DATASET_KWARGS[args.dataset])  # type: ignore
ds = ds.rename_columns(RENAMED_COLUMNS.get(args.dataset, {}))
cols = ds.column_names
ds = ds.map(_format, desc="Formatting")  # type: ignore
ds = ds.map(_tokenize, batched=True, remove_columns=cols + ["text"], desc="Tokenizing")  # type: ignore
dl = torch.utils.data.DataLoader(
    ds,  # type: ignore
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
    LearningRateMonitor(logging_interval="step"),  # Built-in LR monitoring
    ModelCheckpoint(  # save last
        dirpath=f"{OUTPUT_DIR}/{get_econfig_name(args)}",
        filename="last",
        every_n_train_steps=args.ckpt_every,
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

# hacky way to maybe auto resume
resume_ckpt = lookup_ckpt(args)
wandb_id = None
if resume_ckpt is not None:
    print(f"[INFO] Resuming from checkpoint {resume_ckpt}.")
    resume_ckpt = lookup_ckpt(args)
    wandb_id = lookup_wandb_run(args)

trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    gradient_clip_val=1.0,
    accumulate_grad_batches=args.accumulate_grad_batches,
    default_root_dir=OUTPUT_DIR,
    max_steps=max_steps,
    callbacks=callbacks,
    accelerator="auto",
    logger=WandbLogger(
        project="mtl",
        tags=args.tags,
        name=get_econfig_name(args),
        id=wandb_id,
        resume="allow",
    ),
)
trainer.fit(
    LitLlama(model, max_steps, lr=args.lr), train_dataloaders=dl, ckpt_path=resume_ckpt
)

# ---------- Save HF-compatible checkpoint ----------
if trainer.is_global_zero:
    final_dir = os.path.join(OUTPUT_DIR, get_econfig_name(args), "hf")
    os.makedirs(final_dir, exist_ok=True)
    # Save base model weights (already updated during training) + tokenizer
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    print(f"[INFO] Saved HF checkpoint to {final_dir}")
    print(
        "Run eval with: accelerate launch -m lm_eval --model hf --tasks gsm8k_cot --batch_size 16 --model_args pretrained="
        + final_dir
    )

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
