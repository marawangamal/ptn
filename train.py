"""Minimal training script for SmolLM.

Usage:
    python train_smol.py --model_name distilbert/distilgpt2 --dataset_name wikitext --max_length 32 --epochs 1 --batch_size 1
    python train_smol.py --datasets wikipedia --max_num_samples 50 --batch_size 1 --max_length 8 --epochs 10 --model distilbert/distilgpt2
"""

# TODO:
# Perf improvements
# [x] use cosine annealing lr scheduler (test on wikitext)
# [x] add validation set
# [x] add qualitative evaluation / logging (ie., x="the world is")
# [ ] log norm = torch.clip_grad_norm_(1.0)

# Runtime improvements
# [ ] copy data onto $SLURM_TMPDIR


import os
import re
import argparse
from datetime import timedelta
from typing import Literal

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities import rank_zero_only
import datasets
import wandb
from datasets import load_from_disk
import lm_eval
from lm_eval import simple_evaluate
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from orion.client import report_objective
from pytorch_lightning.utilities import rank_zero_only

from mtp.mthf import MultiTokenHFConfig, MultiTokenHF

EXPERIMENTS_DIR = "experiments"

PRETRAINING_DS_CONFIG = {
    "fineweb": {
        "load_from_disk_path": os.path.join(
            os.environ.get("HF_HOME", "data"), "processed", "fineweb"
        ),
        # "path": "HuggingFaceFW/fineweb",
        # "name": "sample-10BT",
        # "split": "train",
        # "streaming": True,
    },
    # small ds for testing
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-v1",
        "split": "test",
        "streaming": True,
    },
}


class LitLM(pl.LightningModule):
    def __init__(
        self,
        model_name,
        model_head,
        vocab_size,
        horizon,
        max_steps,
        lr=5e-5,
        scheduler: Literal["none", "cosine"] = "none",
    ):
        super().__init__()
        self.save_hyperparameters()
        config = MultiTokenHFConfig(
            model_name=model_name,
            model_head=model_head,
            vocab_size=vocab_size,
            horizon=horizon,
        )
        self.model = MultiTokenHF(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_after_backward(self) -> None:
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.log("grad_norm", norm, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        if (
            self.hparams["scheduler"] == "cosine"
            and self.hparams["max_steps"] is not None
        ):
            print(
                f"[INFO] Using cosine annealing with {self.hparams['max_steps']} steps."
            )
            opt = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
            warmup_steps = int(0.01 * self.hparams["max_steps"])
            warmup_factor = lambda st: 0.05 + 0.95 * (st / max(warmup_steps, 1))
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, warmup_factor)
            cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=self.hparams["max_steps"] - warmup_steps,
                eta_min=0.1 * self.hparams["lr"],
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

        else:
            return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])


class LMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        dataset_name,
        batch_size,
        max_length,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.max_length:
            total_length = (total_length // self.max_length) * self.max_length
        # Split by chunks of block_size.
        result = {
            k: [
                t[i : i + self.max_length]
                for i in range(0, total_length, self.max_length)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def setup(self, stage=None):
        if PRETRAINING_DS_CONFIG[self.dataset_name].get("load_from_disk_path"):
            self.dataset = load_from_disk(
                PRETRAINING_DS_CONFIG[self.dataset_name]["load_from_disk_path"]
            )
        else:
            self.dataset = datasets.load_dataset(
                **PRETRAINING_DS_CONFIG[self.dataset_name]
            )
            self.dataset = self.dataset.filter(
                lambda x: x["text"] and x["text"].strip() != ""
            )
            self.dataset = self.dataset.shuffle(seed=42)

            # Tokenize
            self.dataset = self.dataset.map(
                lambda x: self.tokenizer(x["text"]),
                remove_columns=["text"],
                batched=True,
            )

            # Group instead of padding/truncation
            self.dataset = self.dataset.map(
                lambda x: self.group_texts(x),
                batched=True,
            )

        self.dataset = self.dataset.train_test_split(test_size=0.1)
        self.dataset["val"] = self.dataset["test"]

    def train_dataloader(self):
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        return DataLoader(
            self.dataset["train"], batch_size=self.batch_size, collate_fn=collator
        )

    def val_dataloader(self):
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        return DataLoader(
            self.dataset["val"], batch_size=self.batch_size, collate_fn=collator
        )


class HellaSwagEvalCallback(pl.Callback):
    def __init__(self, model_name, val_check_interval=1, device=None):

        super().__init__()
        self.model_name = model_name
        self.val_check_interval = val_check_interval
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    @rank_zero_only
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (batch_idx + 1) % self.val_check_interval == 0:
            print(
                f"\n[HellaSwagEvalCallback] Evaluating on HellaSwag at batch {batch_idx+1}..."
            )
            results = simple_evaluate(
                model=lm_eval.models.huggingface.HFLM(pretrained=pl_module.model),
                tasks=["hellaswag"],
                num_fewshot=0,
                batch_size=2,
                gen_kwargs={"max_new_tokens": 40},
            )
            if (
                results
                and results.get("results")
                and results["results"].get("hellaswag")
            ):
                print(
                    f"[HellaSwagEvalCallback] HellaSwag results: {results['results']['hellaswag']}"
                )
                acc = results["results"]["hellaswag"].get("acc,none")
                acc_norm = results["results"]["hellaswag"].get("acc_norm,none")
                if acc_norm is not None:
                    log_dict = {
                        "eval/hellaswag_acc": acc,
                        "eval/hellaswag_acc_norm": acc_norm,
                        "batch": batch_idx + 1,
                    }
                    if (
                        hasattr(trainer.logger, "log_metrics")
                        and trainer.logger.log_metrics is not None
                    ):
                        trainer.logger.log_metrics(log_dict, step=batch_idx + 1)
            else:
                print("[HellaSwagEvalCallback] HellaSwag results not available.")


class SampleEvalCallback(pl.Callback):
    def __init__(
        self, tokenizer, val_check_interval=1, prefix="Hello, I'm a language model,"
    ):
        super().__init__()
        self.val_check_interval = val_check_interval
        self.prefix = prefix
        self.tokenizer = tokenizer

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (batch_idx + 1) % self.val_check_interval == 0:
            outputs = pl_module.model.generate(
                self.tokenizer.encode(self.prefix, return_tensors="pt").to(
                    pl_module.device
                ),
                max_new_tokens=32,
                do_sample=True,
                top_k=50,
            )
            columns = ["text"]
            data = [[self.tokenizer.decode(outputs[0])]]
            trainer.logger.log_text(
                key="samples", columns=columns, data=data, step=batch_idx + 1
            )


class OrionCallback(pl.Callback):
    """
    Track the best value of `monitor` and send it to Orion at the end of fit().

    Args
    ----
    monitor : str
        Metric key in ``trainer.callback_metrics`` to track
        (e.g. "val_loss_epoch", "eval/hellaswag_acc_norm").
    mode : {"min", "max"}
        Whether a lower or higher value is better.
    """

    def __init__(self, monitor: str = "val_loss_epoch", mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.minimize = mode == "min"
        self.best = float("inf") if self.minimize else -float("inf")

    # ---------- helpers -----------------------------------------------------
    def _is_better(self, current):
        if current is None:
            return False
        return (current < self.best) if self.minimize else (current > self.best)

    # ---------- Lightning hooks --------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        """Update best after every validation epoch."""
        current = trainer.callback_metrics.get(self.monitor)
        if self._is_better(current):
            self.best = current

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        """Send the score to Orion once training is finished."""
        value = self.best.item() if torch.is_tensor(self.best) else float(self.best)
        report_objective(value)
        print(f"[Orion] reported {self.monitor} = {value:.5f}")


def get_econfig_name(args: argparse.Namespace):
    ignore_keys = [
        "disable_auto_resume",
        "disable_evals",
        "val_check_interval",
        "tags",
    ]
    parts = [f"{k[:1]}{v}" for k, v in args.__dict__.items() if k not in ignore_keys]
    # remove special characters
    parts = [re.sub(r"[^a-zA-Z0-9]", "", p) for p in parts]
    return "_".join(parts)


def lookup_ckpt(args: argparse.Namespace):
    ckpt_path = f"{EXPERIMENTS_DIR}/{get_econfig_name(args)}/last.ckpt"
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


def main():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--model_head", type=str, default="stp")
    p.add_argument("--horizon", type=int, default=1)
    # data
    p.add_argument("--dataset_name", type=str, default="fineweb")
    # optimization
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    p.add_argument("--lr", type=float, default=None)
    # misc (untracked)
    p.add_argument("--disable_auto_resume", action="store_true")
    p.add_argument("--disable_evals", action="store_true")
    p.add_argument("--val_check_interval", type=int, default=5000)
    p.add_argument("--limit_train_batches", type=int, default=None)  # used for hpo
    p.add_argument("--limit_val_batches", type=int, default=None)  # used for hpo
    p.add_argument("--tags", type=str, nargs="*", default=[])
    args = p.parse_args()

    # data
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    dm = LMDataModule(
        tokenizer,
        args.dataset_name,
        args.batch_size,
        args.max_length,
    )

    # model
    max_steps = args.limit_train_batches
    if max_steps is None:
        dm.setup()
        max_steps = args.epochs * len(dm.train_dataloader())
    model = LitLM(
        args.model_name,
        model_head=args.model_head,
        vocab_size=tokenizer.vocab_size,
        horizon=args.horizon,
        max_steps=max_steps,
        scheduler=args.scheduler,
    )

    # maybe auto resume
    resume_ckpt = lookup_ckpt(args)
    wandb_id = None
    if not (args.disable_auto_resume or resume_ckpt is None):
        print(f"[INFO] Resuming from checkpoint {resume_ckpt}.")
        resume_ckpt = lookup_ckpt(args)
        wandb_id = lookup_wandb_run(args)

    # trainer callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(  # save last
            dirpath=f"{EXPERIMENTS_DIR}/{get_econfig_name(args)}",
            filename="last",
            every_n_train_steps=1000,
        ),
        OrionCallback(
            monitor="val_loss_epoch",
        ),
    ]

    # Add evals
    if not args.disable_evals:
        callbacks.extend(
            [
                HellaSwagEvalCallback(
                    args.model_name,
                    val_check_interval=args.val_check_interval,
                ),
                SampleEvalCallback(
                    tokenizer,
                    val_check_interval=args.val_check_interval,
                ),
                ModelCheckpoint(  # save best ckpt according to eval
                    dirpath=f"{EXPERIMENTS_DIR}/{get_econfig_name(args)}",
                    filename="best",
                    monitor="eval/hellaswag_acc_norm",
                    mode="max",
                    save_top_k=1,
                ),
            ]
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=WandbLogger(
            project="mtl",
            name=get_econfig_name(args),
            id=wandb_id,
            resume="allow",
            tags=args.tags,
        ),
        default_root_dir=f"{EXPERIMENTS_DIR}/{get_econfig_name(args)}",
        val_check_interval=args.val_check_interval,
        # used for hpo
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    # Tune lr
    if not resume_ckpt and args.lr is None:
        # skip lr tuning if resuming from checkpoint
        tuner = Tuner(trainer)
        tuner.lr_find(model, dm)

    # NOTE: Add callback after lr tuning to avoid issues
    trainer.callbacks.extend(callbacks)
    trainer.fit(model, dm, ckpt_path=resume_ckpt)  # for auto resume, not for saving
    print("Done!")

    # # Report to orion
    # for cb in trainer.callbacks:
    #     if isinstance(cb, ModelCheckpoint) and cb.monitor == "val_loss":
    #         val_loss = cb.best_model_score
    #         report_objective(val_loss)
    #         print(f"Best val_loss: {val_loss}")
    #         print(f"Best checkpoint path: {cb.best_model_path}")


if __name__ == "__main__":
    main()


# Epoch 0:   2%|██▉       | 5337/315209 [54:08<52:23:38,  1.64it/s, v_num=v874, train_loss_step=4.630]^C
# Epoch 0:  21%|████████████████▋           | 16418/78803 [2:55:56<11:08:31,  1.56it/s, v_num=m7ey, train_loss_step=3.710]slurmstepd: error: container_p_join: open failed for /var/opt/slurm/localstorage/7233753/.ns: No such file or directory

# Single gpu
# Epoch 0:   0%|▏      | 171/157605 [01:39<25:29:40,  1.72it/s, v_num=06tw, train_loss_step=6.970]
# BFloat16 and Float
