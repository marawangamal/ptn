#!/usr/bin/env python
# train_v2.py
#
# Example:
# WANDB_CACHE_DIR=$SCRATCH/wandb HF_HOME=$SCRATCH/huggingface python train_v2.py --tags tamia --eval_limit 50 --lr 1e-7 --lambda_mhead 0.1 --model_head multihead --horizon 2 --max_steps 1000
# To evaluate a ckpt, run:
# accelerate launch -m lm_eval --model hf --model_args pretrained=experiments/mmetallamaLlama323BInstruct_domi1m_m512_b8_l1e07_m1_mNone_a1/hf --tasks gsm8k_cot  --batch_size 64

# TODO:
# [ ] Change to loss_dict
# [ ] Ignore prefix targets (set to -100)


import os
import re
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import argparse, torch
from typing import Callable, List, Optional
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

from ctn.mthf.modelling_mthf import MultiTokenHF, MultiTokenHFConfig


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
    def __init__(
        self,
        prompts,
        tokenizer,
        every_steps=200,
        max_new=128,
        val_on_start=True,
        **gen_kwargs,
    ):
        self.prompts = prompts
        self.tok = tokenizer
        self.every_steps = every_steps
        self.max_new = max_new
        self.gen_kwargs = dict(**gen_kwargs)
        self.val_on_start = val_on_start

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *_):
        step = trainer.global_step
        if (step == 0 and not self.val_on_start) or (step % self.every_steps):
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
            print(f"[{step}] (PeriodicSample) {prompt} -> {completion}")
            columns = ["text"]
            data = [[completion]]
            trainer.logger.log_text(key="samples", columns=columns, data=data)
        pl_module.train()


class PeriodicEvaluate(pl.Callback):
    def __init__(
        self,
        evals,
        batch_size,
        limit=None,
        every_steps=100,
        val_on_start=True,
    ):
        self.evals = evals
        self.batch_size = batch_size
        self.limit = limit
        self.every_steps = every_steps
        self.val_on_start = val_on_start

    def _run_eval(self, pl_module, logger_prefix=None, **kwargs):
        print(
            f"Running eval on {self.evals} with batch size {self.batch_size} and limit {self.limit}"
        )
        output = simple_evaluate(
            model=lm_eval.models.huggingface.HFLM(pretrained=pl_module.model),
            tasks=self.evals,
            batch_size=self.batch_size,
            **kwargs,
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

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *_):
        step = trainer.global_step
        if (step == 0 and not self.val_on_start) or (step % self.every_steps):
            return
        pl_module.eval()
        with torch.no_grad():
            self._run_eval(pl_module, logger_prefix=f"{step}", limit=self.limit)
        pl_module.train()


# class BestHFCheckpoint(pl.Callback):
#     def __init__(self, output_dir, eval_metric, tokenizer):
#         self.output_dir = output_dir
#         self.eval_metric = eval_metric
#         self.tokenizer = tokenizer
#         self.best_metric = 0.0

#     def on_train_batch_end(self, trainer, pl_module, *_):
#         # Check if we have the metric we're monitoring
#         if self.eval_metric in trainer.logged_metrics:
#             current_value = trainer.logged_metrics[self.eval_metric]

#             # Save if this is the best value so far
#             if current_value > self.best_metric:
#                 self.best_metric = current_value
#                 final_dir = os.path.join(self.output_dir, "hf_best")

#                 # Save HF checkpoint using shared function
#                 save_hf_checkpoint(pl_module.model, self.tokenizer, final_dir)
#                 print(
#                     f"[INFO] Saved best HF checkpoint to {final_dir} (metric: {self.best_metric:.4f})"
#                 )


# ---------- Lightning Module ----------
class LitLlama(pl.LightningModule):
    def __init__(self, model, max_steps, lr=2e-5, warmup_ratio=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.max_steps = max_steps
        print(f"[INFO] Warmup ratio: {warmup_ratio} | Max steps: {max_steps}")

    def forward(self, **x):
        return self.model(**x)

    def _grad_vector(self, module, max_params=1000000):
        """Get gradient vector for a module, sampling parameters if needed to avoid OOM."""
        all_grads = []

        for p in module.parameters():
            if p.grad is not None:
                grad_flat = p.grad.detach().flatten()
                all_grads.append(grad_flat)

        # Concatenate all gradients
        full_grad = torch.cat(all_grads)

        # If too large, sample a subset deterministically
        if full_grad.numel() > max_params:
            # Use evenly spaced indices for deterministic sampling
            step = full_grad.numel() // max_params
            indices = torch.arange(0, full_grad.numel(), step)[:max_params]
            return full_grad[indices]

        return full_grad

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)

        # Only compute gradient analysis if enabled and conditions are met
        if (
            self.model is not None
            and hasattr(self.model, "backbone")
            and outputs.loss_main is not None
            and outputs.loss_aux is not None
        ):
            # Only log every 50 steps to save time and memory
            if batch_idx % 50 == 0:
                # Get grads for main loss
                self.zero_grad()
                outputs.loss_main.backward(retain_graph=True)
                g_main = self._grad_vector(self.model.backbone, 10_000)

                # Get grads for aux loss
                self.zero_grad()
                outputs.loss_aux.backward(retain_graph=True)
                g_aux = self._grad_vector(self.model.backbone, 10_000)

                # Cosine similarity & ratio
                cos = torch.cosine_similarity(g_main, g_aux, dim=0).item()
                ratio = (g_aux.norm() / (g_main.norm() + 1e-12)).item()

                self.log("grad_cosine_main_aux", cos, prog_bar=True)
                self.log("grad_ratio_main_aux", ratio, prog_bar=False)
                # Clean up
                self.zero_grad()

        if (
            hasattr(outputs, "loss_main")
            and hasattr(outputs, "loss_aux")
            and outputs.loss_main is not None
            and outputs.loss_aux is not None
        ):
            self.log("loss_main", outputs.loss_main)
            self.log("loss_aux", outputs.loss_aux)

        self.log("loss", outputs.loss)
        return outputs.loss

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

        # # Linear decay scheduler
        # def linear_decay_factor(step):
        #     if step < warmup_steps:
        #         return 1.0  # During warmup, keep full LR
        #     else:
        #         # Linear decay from 1.0 to 0.1 over the remaining steps
        #         decay_steps = self.hparams["max_steps"] - warmup_steps
        #         decay_progress = (step - warmup_steps) / max(decay_steps, 1)
        #         return max(
        #             0.1, 1.0 - 0.9 * decay_progress
        #         )  # Decay to 10% of original LR
        # linear_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, linear_decay_factor)

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


def save_hf_checkpoint(model, tokenizer, output_dir):
    """Save a HuggingFace-compatible checkpoint with all necessary configuration files."""
    os.makedirs(output_dir, exist_ok=True)

    # Teach HF how to import your custom classes
    model.config.model_type = "multi_token_hfmodel"
    model.config.architectures = ["MultiTokenHF"]
    model.config.auto_map = {
        "AutoConfig": "configuration_multi_token_hfmodel.MultiTokenHFConfig",
        "AutoModelForCausalLM": "modeling_multi_token_hfmodel.MultiTokenHF",
    }
    # IMPORTANT: prevent __init__ from re-downloading the base on load
    model.config.pretrained = False  # ← add this line

    # Write the minimal loader stubs next to the weights
    open(os.path.join(output_dir, "configuration_multi_token_hfmodel.py"), "w").write(
        "from transformers.configuration_utils import PretrainedConfig\n\n"
        "class MultiTokenHFConfig(PretrainedConfig):\n"
        "    model_type = 'multi_token_hfmodel'\n"
        "    def __init__(self, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
    )

    open(os.path.join(output_dir, "modeling_multi_token_hfmodel.py"), "w").write(
        "from ctn.mthf.modelling_mthf import MultiTokenHF, MultiTokenHFConfig\n"
    )

    # Save base model weights + tokenizer
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir, safe_serialization=True)


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
# model
p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
p.add_argument("--tokenizer", default=None)
p.add_argument("--model_head", type=str, default="stp")
p.add_argument("--horizon", type=int, default=1)
p.add_argument("--lambda_mhead", type=float, default=0.0)
# data
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
tokenizer = AutoTokenizer.from_pretrained(
    args.model if args.tokenizer is None else args.tokenizer, use_fast=True
)
model = MultiTokenHF(
    MultiTokenHFConfig(
        model_name=args.model,
        model_head=args.model_head,
        horizon=args.horizon,
        lambda_mhead=args.lambda_mhead,
        loss_type="joint",
        pretrained=True,
    )
)

# NOTE: Would be a better api experience to load like this:
# model = MultiTokenHF.from_pretrained(
#     "experiments/mmetallamaLlama323BInstruct_domi1m_m512_b8_l1e07_m1_mNone_a1/hf"
# )
# model = AutoModelForCausalLM.from_pretrained(args.model)

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
    log_every_n_steps=10,
)
trainer.fit(
    LitLlama(model, max_steps, lr=args.lr),
    train_dataloaders=dl,
    ckpt_path=resume_ckpt,
)
print("Done training.")

# if trainer.is_global_zero:
#     final_dir = os.path.join(OUTPUT_DIR, get_econfig_name(args), "hf")
#     save_hf_checkpoint(model, tokenizer, final_dir)
#     print(f"[INFO] Saved HF checkpoint to {final_dir}")
#     print(
#         "[INFO] Run eval with: accelerate launch -m lm_eval --model hf --tasks gsm8k_cot --batch_size 64 --model_args pretrained="
#         + final_dir
#         + ",trust_remote_code=true"
#     )
