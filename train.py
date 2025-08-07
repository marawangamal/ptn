"""Minimal training script for SmolLM.

Usage:
    python train_smol.py --model_name distilbert/distilgpt2 --dataset_name wikitext --max_length 32 --epochs 1 --batch_size 1
    python train_smol.py --datasets wikipedia --max_num_samples 50 --batch_size 1 --max_length 8 --epochs 10 --model distilbert/distilgpt2
"""

# TODO:
# Performance improvements
# [x] (All) exclude prefix prediction from loss
# [x] (Multihead) Add H-1 heads and Aux loss function
# [ ] (All) Add option for non-chunking (use PAD or EOS)
# [ ] (MuToR Specific) add bi-directionl attention for prefixes


# Runtime improvements
# [ ] copy data onto $SLURM_TMPDIR
# [ ] use `torch.compile`

# Misc:
# [ ] Add evals (ARC, PIQA, etc.) See https://arxiv.org/pdf/2203.15556

import os
import re
import argparse
from typing import Any, List, Literal

import torch
import wandb
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

import datasets
from transformers import AutoTokenizer
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from dataloaders import get_dataset
import lm_eval
from lm_eval import simple_evaluate

from mtp.mthf import MultiTokenHFConfig, MultiTokenHF


EXPERIMENTS_DIR = "experiments"
DS_KWARGS = {  # presets for diff datasets
    "omi:1m": {
        "dataset_name": "nvidia/OpenMathInstruct-2",
        "split": "train_1M",
        "subset": "",
        "column_names": ["problem", "generated_solution"],
    },
    "fineweb": {
        "dataset_name": "HuggingFaceFW/fineweb",
        "subset": "sample-10BT",
        "split": "train",
        "column_names": ["text"],
    },
    "wikitext": {
        "dataset_name": "wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": "train[:10000]",
        "column_names": ["text"],
    },
    "gsm8k": {
        "dataset_name": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "column_names": ["question", "answer"],
    },
}

# print("-" * 100)
# print(f"HF_HOME: {os.environ.get('HF_HOME', 'data')}")
# print("-" * 100)


class LitLM(pl.LightningModule):
    def __init__(
        self,
        model_name,
        model_head,
        horizon,
        max_steps,
        lr=5e-5,
        scheduler: Literal["none", "cosine"] = "none",
        loss_type: Literal["joint", "mhead"] = "mhead",
        pretrained: bool = False,
        warmup_ratio: float = 0.05,
        lambda_mhead: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = None

    def configure_model(self) -> None:
        if self.model is None:
            config = MultiTokenHFConfig(
                model_name=self.hparams["model_name"],
                model_head=self.hparams["model_head"],
                horizon=self.hparams["horizon"],
                loss_type=self.hparams["loss_type"],
                pretrained=self.hparams["pretrained"],
                lambda_mhead=self.hparams["lambda_mhead"],
            )
            self.model = MultiTokenHF(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
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

        else:
            return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])


class LMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name,
        dataset_name,
        subset,
        split,
        batch_size,
        max_length,
        split_ratio=0.1,
        column_names=["text"],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length
        self.split_ratio = split_ratio
        self.column_names = column_names

        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        ds = get_dataset(
            dataset=self.dataset_name,
            subset=self.subset,
            split=self.split,
            tokenizer=self.tokenizer_name,
            max_len=self.max_length,
            column_names=self.column_names,
        )

        self.dataset = ds.train_test_split(test_size=self.split_ratio)
        self.dataset["val"] = self.dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["val"],
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
        )

    # def state_dict(self):
    #     # track whatever you want here
    #     state = {"current_train_batch_index": self.current_train_batch_index}
    #     return state

    # def load_state_dict(self, state_dict):
    #     # restore the state based on what you tracked in (def state_dict)
    #     self.current_train_batch_index = state_dict["current_train_batch_index"]


class LMEvalsCallback(pl.Callback):
    def __init__(
        self,
        model_name,
        evals,
        batch_size,
        val_check_interval=1,
        val_on_start=False,
        device=None,
        limit=None,
    ):

        super().__init__()
        self.model_name = model_name
        self.evals = evals
        self.batch_size = batch_size
        self.val_check_interval = val_check_interval
        self.val_on_start = val_on_start
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.limit = limit
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    def _run_eval(self, pl_module, logger_prefix=None):
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

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (batch_idx + 1) % self.val_check_interval == 0 or (
            batch_idx == 0 and self.val_on_start
        ):
            self._run_eval(
                pl_module=pl_module,
                logger_prefix=f"Batch {batch_idx + 1}",
            )

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if (
            hasattr(trainer, "current_epoch")
            and hasattr(trainer, "max_epochs")
            and trainer.current_epoch is not None
            and trainer.max_epochs is not None
            and trainer.current_epoch == trainer.max_epochs - 1
        ):
            self._run_eval(
                pl_module=pl_module,
                logger_prefix="Epoch End",
            )


class SampleEvalCallback(pl.Callback):
    def __init__(
        self,
        tokenizer,
        val_check_interval=1,
        val_on_start=False,
        prefix="Hello, I'm a language model,",
    ):
        super().__init__()
        self.val_check_interval = val_check_interval
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.val_on_start = val_on_start

    def _run_eval(self, trainer, pl_module, logger_prefix="N/A"):
        outputs = pl_module.model.generate(
            self.tokenizer.encode(self.prefix, return_tensors="pt").to(
                pl_module.device
            ),
            max_new_tokens=64,
            do_sample=False,
        )
        columns = ["text"]
        data = [[self.tokenizer.decode(outputs[0])]]
        print(f"[{logger_prefix}] (SampleEvalCallback) Generated sample: {data[0][0]}")
        trainer.logger.log_text(key="samples", columns=columns, data=data)

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (batch_idx + 1) % self.val_check_interval == 0 or (
            batch_idx == 0 and self.val_on_start
        ):
            self._run_eval(
                trainer=trainer,
                pl_module=pl_module,
                logger_prefix=f"Batch {batch_idx + 1}",
            )

    # @rank_zero_only
    # def on_train_batch_end(
    #     self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    # ):
    #     if (batch_idx + 1) % self.val_check_interval == 0:
    #         self._run_eval(trainer, pl_module, logger_prefix=f"Batch {batch_idx}")

    # @rank_zero_only
    # def on_train_start(self, trainer, pl_module):
    #     if self.val_on_start:
    #         self._run_eval(trainer, pl_module, logger_prefix="Start")


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

    # @rank_zero_only
    # def on_fit_end(self, trainer, pl_module):
    #     """Send the score to Orion once training is finished."""
    #     value = self.best.item() if torch.is_tensor(self.best) else float(self.best)
    #     report_objective(value)
    #     print(f"[Orion] reported {self.monitor} = {value:.5f}")


def get_econfig_name(args: argparse.Namespace):
    ignore_keys = [
        "disable_auto_resume",
        "disable_evals",
        "val_check_interval",
        "val_on_start",
        "ckpt_interval",
        "tags",
        "evals",
        "column_names",
        # ds metadata
        "subset",
        "split",
        "prepare_ds",
        "fast_dev_run",
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
    p.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--model_head", type=str, default="stp")
    p.add_argument("--lambda_mhead", type=float, default=0.1)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--loss_type", type=str, default="mhead", choices=["joint", "mhead"])
    # data
    p.add_argument("--dataset", type=str, default="fineweb", choices=DS_KWARGS.keys())
    # optimization
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    # misc (untracked)
    p.add_argument("--disable_auto_resume", action="store_true")
    p.add_argument("--disable_evals", action="store_true")
    p.add_argument("--val_check_interval", type=int, default=5000)
    p.add_argument("--val_on_start", action="store_true")
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--limit_train_batches", type=int, default=None)  # used for hpo
    p.add_argument("--limit_val_batches", type=int, default=None)  # used for hpo
    p.add_argument("--tags", type=str, nargs="*", default=[])
    p.add_argument("--evals", type=str, nargs="*", default=["hellaswag"])
    p.add_argument("--fast_dev_run", action="store_true")
    args = p.parse_args()

    # data
    dm = LMDataModule(
        tokenizer_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        **DS_KWARGS[args.dataset],
    )

    # model
    max_steps = args.limit_train_batches  # BUG: incorrect, seeing full cosine curve
    if max_steps is None:
        dm.setup()
        max_steps = (
            args.epochs
            * len(dm.train_dataloader())
            / (torch.cuda.device_count() if torch.cuda.is_available() else 1)
            / args.accumulate_grad_batches
        )
    model = LitLM(
        model_name=args.model,
        model_head=args.model_head,
        horizon=args.horizon,
        max_steps=max_steps,
        scheduler=args.scheduler,
        lr=args.lr,
        loss_type=args.loss_type,
        pretrained=args.pretrained,
        lambda_mhead=args.lambda_mhead,
    )

    # hacky way to maybe auto resume
    resume_ckpt = lookup_ckpt(args)
    wandb_id = None
    if not (args.disable_auto_resume or resume_ckpt is None):
        print(f"[INFO] Resuming from checkpoint {resume_ckpt}.")
        resume_ckpt = lookup_ckpt(args)
        wandb_id = lookup_wandb_run(args)

    # trainer callbacks
    callbacks: List[pl.Callback] = [
        LearningRateMonitor(logging_interval="step")
        # ModelCheckpoint(  # save last
        #     dirpath=f"{EXPERIMENTS_DIR}/{get_econfig_name(args)}",
        #     filename="last",
        #     every_n_train_steps=args.ckpt_interval,
        # ),
        # OrionCallback(
        #     monitor="val_loss_epoch",
        # ),
    ]

    # Add evals
    if not args.disable_evals:
        callbacks.extend(
            [
                LMEvalsCallback(
                    args.model,
                    val_check_interval=args.val_check_interval,
                    limit=args.limit_val_batches,
                    evals=args.evals,
                    batch_size=args.batch_size,
                    val_on_start=args.val_on_start,
                ),
                SampleEvalCallback(
                    dm.tokenizer,
                    val_check_interval=args.val_check_interval,
                    val_on_start=args.val_on_start,
                ),
                # ModelCheckpoint(  # save best ckpt according to eval
                #     dirpath=f"{EXPERIMENTS_DIR}/{get_econfig_name(args)}",
                #     filename="best",
                #     monitor="eval/hellaswag_acc_norm",
                #     mode="max",
                #     save_top_k=1,
                # ),
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
        accumulate_grad_batches=args.accumulate_grad_batches,
        fast_dev_run=args.fast_dev_run,
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
